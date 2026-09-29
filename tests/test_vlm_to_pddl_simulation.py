#!/usr/bin/env python
"""
Test Script: End-to-End VLM-to-PDDL with Behavior Tree & RRT Simulation Execution
================================================================================
This script integrates:
1. Environment initialization: Robosuite `TaskSorting` with Panda robot.
2. Vision Pipeline (Qwen 2.5-VL 3B): Grounding & Ambiguity Resolution from camera frame.
3. Reasoning Pipeline (Llama 3): PDDL problem JSON generation & validation.
4. Symbolic Planning: Fast Downward planner (`PDDLenv`) solving the manipulation plan.
5. Execution Layer: `py_trees` Behavior Tree compiler (`build_bt_from_pddl_plan`).
6. Motion Planning: `TaskSpaceRRT` for obstacle-avoidance trajectory generation.
7. Real-Time Visualization: Live dual-camera OpenCV GUI display (Frontal + Top-Down)
   and high-resolution video recording (`videos/simulation_trial_<idx>_<timestamp>.mp4`).
"""

import os
import sys
import time
import re
from datetime import datetime
import argparse

# Eagerly import torch and PPO before any OpenGL/EGL context is created by MuJoCo/robosuite
import torch
from stable_baselines3 import PPO

import cv2
import imageio
import numpy as np
import robosuite as suite
import py_trees

# Ensure workspace root is in sys.path
curr_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(curr_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Custom environment, adapter, VLM, PDDL, and Behavior Tree modules
import src.envs  # registers TaskSorting
from src.envs import RobosuiteEnvAdapter
from src.ambiguityres.vlm_model import ambresFewShotPrompt, AmbresStructured, SceneGrounding
from src.llm2pddl.domains import Domain, PDDLenv
from src.behaviorTree import (
    build_bt_from_pddl_plan,
    render_bt,
    export_presentation_bt,
)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end VLM to PDDL with BT & RRT Simulation")
    parser.add_argument(
        "--task",
        type=str,
        default="Put the red can in the sorting bin, then put the yellow cube in the pot. Then take the blue can and put it in the sorting bin.",
        help="Natural language task instruction",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=1200,
        help="Maximum Behavior Tree ticks for execution",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable live OpenCV GUI window (useful for headless / CI environments)",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM/LLM query and use deterministic pre-validated plan for rapid simulation testing",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default=None,
        help="Optional custom filename for output video (defaults to simulation_trial_<idx>_<timestamp>.mp4)",
    )
    parser.add_argument(
        "--render-bt",
        action="store_true",
        help="Render and save the generated Behavior Tree diagram (PNG/SVG) into the images folder",
    )
    parser.add_argument(
        "--bt-image-name",
        type=str,
        default="behavior_tree",
        help="Custom filename for the rendered Behavior Tree image in the images folder (default: behavior_tree)",
    )
    parser.add_argument(
        "--export-presentation-bt",
        action="store_true",
        help="Export presentation-ready Behavior Tree diagrams (PDF, SVG, PNG, DOT) for LaTeX-Beamer",
    )
    parser.add_argument(
        "--presentation-bt-name",
        type=str,
        default="behavior_tree_main_scenario",
        help="Base filename for exported presentation Behavior Tree diagram (default: behavior_tree_main_scenario)",
    )
    parser.add_argument(
        "--no-ppo",
        action="store_true",
        help="Disable PPO micro-manipulation policy and use pure heuristic descend-grasp-lift",
    )
    parser.add_argument(
        "--ppo-model-path",
        type=str,
        default="models/ppo_hybrid_lift.zip",
        help="Path to trained PPO model checkpoint for hybrid pick skill",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("End-to-End VLM -> PDDL -> Behavior Tree -> RRT Simulation")
    print("=" * 70)
    print(f"Task Instruction: '{args.task}'")
    
    # -------------------------------------------------------------------------
    # 1. Initialize Robosuite Environment
    # -------------------------------------------------------------------------
    print("\n[Step 1] Initializing Robosuite Environment (TaskSorting)...")
    env = suite.make(
        env_name="TaskSorting",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["top_down_vlm", "frontview"],
        camera_heights=512,
        camera_widths=512,
    )
    obs = env.reset()
    
    # Capture and save top-down camera frame for VLM grounding
    rgb_top_down = np.flipud(obs["top_down_vlm_image"])
    exp_dir = os.path.join(repo_root, "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    image_path = os.path.abspath(os.path.join(exp_dir, "test_vlm2pddl_camera_top_down.jpg"))
    cv2.imwrite(image_path, cv2.cvtColor(rgb_top_down, cv2.COLOR_RGB2BGR))
    print(f"Captured top-down camera frame saved to: {image_path}")

    # -------------------------------------------------------------------------
    # 2. VLM Grounding & PDDL Problem Generation
    # -------------------------------------------------------------------------
    domain_path = os.path.join(repo_root, "domains", "manipulation", "domain.pddl")
    with open(domain_path, "r") as f:
        domain_pddl = f.read()

    plan = None

    if not args.skip_vlm:
        print("\n[Step 2] Initializing vision pipeline (Qwen 2.5-VL 3B)...")
        vision_pipeline = ambresFewShotPrompt(vlm_name="qwen2.5vl:3b")
        vision_pipeline.reset_chat()

        print("Initializing reasoning pipeline (Llama 3)...")
        reasoning_pipeline = AmbresStructured(vlm_name="llama3:latest")

        print(f"\n[Step 3] Resolving Ambiguity and Grounding Objects with VLM...")
        final_objects = ["red can", "yellow cube", "blue can"]
        final_locations = ["sorting_bin", "pot"]
        try:
            input_data = {
                "task_description": args.task,
                "image_path": image_path,
            }
            result = vision_pipeline.handle_query_dict(input_data)

            if result.get("task_ambiguous"):
                print(f"\n[Robot asks]: {result['clarifying_question']}")
                try:
                    user_clarification = input("[Your response]: ")
                except EOFError:
                    user_clarification = "Pick the red can, yellow cube, and blue can."
                    print(f"[Automated response]: {user_clarification}")
                result = vision_pipeline.handle_response(user_clarification)

            sg = SceneGrounding(
                task_objects=result.get("task_objects", []),
                target_locations=result.get("target_locations", [])
            )
            if sg.task_objects:
                final_objects = sg.task_objects
            if sg.target_locations:
                final_locations = sg.target_locations
        except Exception as err:
            print(f"\n⚠️  VLM Query encountered error ({err}). Falling back to grounded objects: {final_objects}")

        print(f"\n[Grounded Objects]: {final_objects}")
        print(f"[Grounded Locations]: {final_locations}")

        print("\n🎯 [Final Grounded Entities for PDDL Problem Synthesis]:")
        print("   📦 Objects (type 'obj'):")
        for idx, name in enumerate(final_objects, 1):
            pddl_name = name.strip().replace(" ", "_")
            print(f"      ({idx}) Natural Name: '{name}'  ==> PDDL Symbol: '{pddl_name}'")
        print("   📍 Locations (type 'location'):")
        for idx, name in enumerate(final_locations, 1):
            pddl_name = name.strip().replace(" ", "_")
            print(f"      ({idx}) Natural Name: '{name}'  ==> PDDL Symbol: '{pddl_name}'")

        print("\n[Step 4] Generating Problem PDDL JSON via Reasoning Pipeline...")
        try:
            initial_state_desc = "All objects are currently resting on the table. The robot's arm is empty."
            problem_obj = reasoning_pipeline.generate_problem_json(
                initial_state_desc=initial_state_desc,
                task_description=args.task,
                task_objects=final_objects,
                domain_pddl=domain_pddl,
                target_locations=final_locations,
            )
            problem_pddl = problem_obj.to_pddl()
            print("\n✅ PDDL Problem Generated and Validated Successfully!")
        except Exception as e:
            print(f"\n❌ Validation Failed ({e}), falling back to deterministic problem.")
            problem_pddl = None
    else:
        print("\n[Step 2-4] Skipping VLM query (--skip-vlm specified). Using direct PDDL problem.")
        problem_pddl = None

    # Fallback / deterministic problem if VLM was skipped or validation failed
    if not problem_pddl:
        problem_pddl = """(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_can blue_can yellow_cube - obj
    sorting_bin pot - location
  )
  (:init
    (on-table red_can)
    (on-table blue_can)
    (on-table yellow_cube)
  )
  (:goal
    (and 
      (on red_can sorting_bin)
      (on yellow_cube pot)
      (on blue_can sorting_bin)
    )
  )
)"""

    # Save trial to experiments/generated_problems folder
    gen_problems_dir = os.path.join(repo_root, "experiments", "generated_problems")
    os.makedirs(gen_problems_dir, exist_ok=True)
    existing_pddl = [f for f in os.listdir(gen_problems_dir) if f.endswith(".pddl") and f != "problem.pddl"]
    pddl_trial_num = len(existing_pddl) + 1
    pddl_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_pddl_path = os.path.join(gen_problems_dir, f"problem_trial_{pddl_trial_num:03d}_{pddl_timestamp}.pddl")
    with open(trial_pddl_path, "w") as f:
        f.write(problem_pddl)
    print(f"📁 Saved trial problem file to: {trial_pddl_path}")

    # Save canonical latest problem file in experiments/generated_problems
    with open(os.path.join(gen_problems_dir, "problem.pddl"), "w") as f:
        f.write(problem_pddl)

    # -------------------------------------------------------------------------
    # 5. Fast Downward Symbolic Planning
    # -------------------------------------------------------------------------
    print("\n[Step 5] Solving PDDL Plan with Fast Downward...")
    fast_downward_path = os.path.join(repo_root, "downward", "fast-downward.py")
    pddl_env = PDDLenv(
        fast_downward_path=fast_downward_path,
        time_limit=10,
        planning_algorithm=PDDLenv.SUB_OPTIMAL_ALIAS,
    )

    plan, succ, msg = pddl_env.search_plan(
        domain_pddl=domain_pddl,
        problem_pddl=problem_pddl,
    )

    if succ and plan:
        print("\n--- Fast Downward Solution Plan ---")
        print(plan.strip())
        print("-----------------------------------")
    else:
        print(f"\n[ERROR] Failed to solve PDDL plan: {msg}")
        env.close()
        return

    # -------------------------------------------------------------------------
    # 6. Build Behavior Tree Execution Layer
    # -------------------------------------------------------------------------
    print("\n[Step 6] Compiling PDDL Plan into Behavior Tree...")
    env_adapter = RobosuiteEnvAdapter(raw_env=env)
    env_adapter.use_ppo = not args.no_ppo
    env_adapter.ppo_model_path = args.ppo_model_path
    if env_adapter.use_ppo:
        from src.behaviorTree.skills import HybridPPOPickUpSkill
        cached = HybridPPOPickUpSkill.get_model(env_adapter.ppo_model_path)
        if cached is not None:
            print(f"  - Pre-cached PPO model: {env_adapter.ppo_model_path}")
        else:
            print(f"  - Warning: PPO model checkpoint not found at {env_adapter.ppo_model_path}")
    pick_mode = "Hybrid TaskSpaceRRT + PPO" if env_adapter.use_ppo else "Pure Heuristic (MotionPlanning)"
    print(f"Pick Skill Architecture: {pick_mode}")

    root_node = build_bt_from_pddl_plan(
        plan=plan,
        env=env_adapter,
    )

    bt_tree = py_trees.trees.BehaviourTree(root_node)
    bt_tree.setup(timeout=15, env=env_adapter)

    print("\n--- Behavior Tree Structure ---")
    print(py_trees.display.ascii_tree(root_node))
    print("-------------------------------")

    # Render Behavior Tree diagram if requested
    if args.render_bt:
        images_dir = os.path.join(repo_root, "Docs", "pictures")
        render_bt(root=root_node, name=args.bt_image_name, target_dir=images_dir)

    # Export presentation-grade Behavior Tree diagram for LaTeX-Beamer if requested
    if args.export_presentation_bt:
        images_dir = os.path.join(repo_root, "Docs", "pictures")
        print("\n[BehaviorTree] Exporting presentation-ready diagram artifacts...")
        paths = export_presentation_bt(
            root=root_node,
            name=args.presentation_bt_name,
            target_dir=images_dir,
            sync_main_scenario_bt=True,
        )
        print(f"✅ Successfully exported presentation diagrams to {images_dir}:")
        for fmt, p in paths.items():
            print(f"  - [{fmt.upper()}] {p}")

    # -------------------------------------------------------------------------
    # 7. Simulation Execution with RRT Motion Planning & Dual-Camera Visualization
    # -------------------------------------------------------------------------
    print("\n[Step 7] Executing Behavior Tree with TaskSpaceRRT Motion Planning...")

    video_dir = os.path.join(repo_root, "videos")
    os.makedirs(video_dir, exist_ok=True)

    if args.video_name:
        video_filename = args.video_name if args.video_name.endswith(".mp4") else f"{args.video_name}.mp4"
    else:
        # Determine next sequential trial index
        trial_indices = []
        for f in os.listdir(video_dir):
            match = re.match(r"simulation_trial_(\d+)_", f)
            if match:
                trial_indices.append(int(match.group(1)))
        trial_num = max(trial_indices) + 1 if trial_indices else 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"simulation_trial_{trial_num:03d}_{timestamp}.mp4"

    video_path = os.path.join(video_dir, video_filename)
    print(f"  - Target video recording path: {video_path}")

    # Initialize high-quality MP4 video writer with faststart and yuv420p for universal player compatibility
    writer = imageio.get_writer(
        video_path,
        fps=20,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart"],
    )

    window_name = "VLM Robot Execution: Frontal | Top-Down (TaskSpaceRRT)"
    gui_active = False
    if not args.no_gui:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1024, 512)
            gui_active = True
        except Exception as e:
            print(f"Warning: could not open GUI window ({e}). Running without live display.")

    tick_count = 0
    max_ticks = args.max_ticks

    try:
        while tick_count < max_ticks:
            bt_tree.tick()
            status = root_node.status

            # Render frontview and top_down_vlm cameras
            try:
                frame_front = env.sim.render(height=512, width=512, camera_name="frontview")
                frame_top = env.sim.render(height=512, width=512, camera_name="top_down_vlm")

                frame_front_rgb = np.ascontiguousarray(np.flipud(frame_front))
                frame_top_rgb = np.ascontiguousarray(np.flipud(frame_top))

                # Combine cameras side-by-side (1024x512)
                combined_rgb = np.hstack([frame_front_rgb, frame_top_rgb])
                combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)

                # Find currently active leaf node
                active_leaf = None
                for node in root_node.iterate():
                    if (
                        node.status == py_trees.common.Status.RUNNING
                        and isinstance(node, py_trees.behaviour.Behaviour)
                        and not isinstance(node, py_trees.composites.Composite)
                        and not isinstance(node, py_trees.decorators.Decorator)
                    ):
                        active_leaf = node
                        break

                action_desc = "Idle"
                if active_leaf is not None:
                    stage = getattr(active_leaf, "stage", "")
                    node_args = getattr(active_leaf, "args", {})
                    target_obj = node_args.get("object", "")
                    target_asset = node_args.get("asset", "")
                    detail = f"{target_obj}" + (f" -> {target_asset}" if target_asset else "")
                    action_desc = f"{active_leaf.name} [{detail}] ({stage})"

                # Render translucent informational HUD overlay
                overlay = combined_bgr.copy()
                cv2.rectangle(overlay, (0, 0), (combined_bgr.shape[1], 50), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.75, combined_bgr, 0.25, 0, combined_bgr)

                # Left HUD: Tick count & status
                status_str = str(status).replace("Status.", "")
                status_color = (0, 255, 0) if status == py_trees.common.Status.SUCCESS else (0, 255, 255)
                cv2.putText(combined_bgr, f"Tick: {tick_count:04d} | Status: {status_str}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

                # Right HUD: Active skill & motion planning stage
                cv2.putText(combined_bgr, f"Skill: {action_desc}", (420, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                # Camera view labels
                cv2.putText(combined_bgr, "Frontal View", (20, combined_bgr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(combined_bgr, "Top-Down (VLM) View", (532, combined_bgr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

                # Save frame to video
                writer.append_data(cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB))

                # Display live GUI window
                if gui_active:
                    cv2.imshow(window_name, combined_bgr)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC to interrupt
                        print("\n[User Interrupt] Execution halted by ESC key.")
                        break
            except Exception as exc:
                print(f"[Warning] Frame rendering error at tick {tick_count}: {exc}")

            tick_count += 1
            if tick_count % 30 == 0 or status != py_trees.common.Status.RUNNING:
                print(f"  [Tick {tick_count:04d}] Status: {status} | Action: {action_desc}")

            if status == py_trees.common.Status.SUCCESS:
                print(f"\n✅ [SUCCESS] Behavior Tree completed entire plan successfully at tick {tick_count}!")
                break
            elif status == py_trees.common.Status.FAILURE:
                print(f"\n❌ [FAILURE] Behavior Tree returned FAILURE at tick {tick_count}.")
                break
    except KeyboardInterrupt:
        print("\n[User Interrupt] KeyboardInterrupt received. Finalizing video...")
    finally:
        if writer is not None:
            writer.close()
            print(f"  - Execution video saved and finalized: {video_path}")
        if gui_active:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        env.close()

    print("\n" + "=" * 70)
    print("Execution Finished!")
    print(f"  - Total simulation ticks: {tick_count}")
    print(f"  - Final BT Status       : {status}")
    print(f"  - Execution video saved : {video_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
