#!/usr/bin/env python
"""
Test Script: Perception & Ambiguity Resolution (VLM Visual Grounding)
====================================================================
Isolates only the Perception and Ambiguity Resolution (Ambres) pipeline:
1. Takes as input:
   - A natural language task prompt (--prompt).
   - An image (--image), or captures a top-down view directly from the TaskSorting simulation.
2. Runs the Ambiguity Resolution & Visual Grounding loop (Qwen 2.5-VL):
   - Queries the VLM with prompt + image.
   - Detects task ambiguity and engages the user in clarification if needed.
   - Resolves entities into `SceneGrounding` (Manipulable Objects and Target Locations).
3. Produces visual segmentation and recognition overlays:
   - Panel 1: Raw VLM camera input frame.
   - Panel 2: Color-coded instance segmentation mask.
   - Panel 3: Object recognition & visual grounding overlay (masks, bounding boxes, labels).
4. Outputs the grounded symbols (natural names and normalized PDDL symbols) in the terminal.
"""

from __future__ import annotations

import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image

# Ensure workspace root is in sys.path
curr_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(curr_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Custom environment and VLM modules
import src.envs  # registers TaskSorting
from src.ambiguityres.vlm_model import ambresFewShotPrompt, SceneGrounding


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perception and Ambiguity Resolution Test with Segmentation and Recognition"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Put the red can in the sorting bin, then put the yellow cube in the pot. Then take the blue can and put it in the sorting bin.",
        help="Natural language task prompt",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an existing input image. If omitted, captures top-down frame from TaskSorting simulation.",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="qwen2.5vl:3b",
        help="VLM model identifier (default: qwen2.5vl:3b)",
    )
    parser.add_argument(
        "--ambiguous",
        action="store_true",
        help="Use an intentionally ambiguous task prompt to test the interactive clarification flow",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiments/perception_segmentation_recognition.jpg",
        help="Destination path for the composite visualization image",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable interactive cv2.imshow GUI display window",
    )
    return parser.parse_args()


# Palette for visualization (BGR format)
PALETTE = {
    "red_can": (0, 0, 230),         # Vivid Red
    "blue_can": (230, 80, 0),        # Vivid Blue
    "green_can": (0, 200, 50),       # Vivid Green
    "yellow_cube": (0, 215, 255),    # Vivid Yellow
    "purple_cube": (200, 0, 180),    # Vivid Purple
    "sorting_bin": (255, 140, 0),    # Cyan / Steel
    "pot": (0, 165, 255),            # Orange
    "default": (180, 180, 180),      # Neutral Gray
}


def capture_from_sim():
    """Spins up TaskSorting environment and captures top-down RGB + Instance Segmentation."""
    import robosuite as suite
    print("🌍 Launching Robosuite TaskSorting environment to capture scene observation...")
    env = suite.make(
        env_name="TaskSorting",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["top_down_vlm"],
        camera_segmentations="instance",
        camera_heights=512,
        camera_widths=512,
    )
    obs = env.reset()

    # Robosuite camera images are vertically inverted relative to standard cv2 coordinates
    rgb_img = np.flipud(obs["top_down_vlm_image"])
    seg_img = np.flipud(obs["top_down_vlm_segmentation_instance"].squeeze(-1))

    # Instance ID to name mapping: {id: name}
    id2name = {i + 1: inst for i, inst in enumerate(list(env.model.instances_to_ids.keys()))}

    # Physical scene objects
    scene_objects = [obj.name for obj in getattr(env, "objects", [])]

    env.close()
    return rgb_img, seg_img, id2name, scene_objects


def create_segmentation_colormap(seg_img, id2name):
    """Converts 2D integer instance segmentation into an aesthetic RGB color mask."""
    h, w = seg_img.shape
    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)

    for inst_id, name in id2name.items():
        mask = (seg_img == inst_id)
        if not np.any(mask):
            continue
        clean = name.lower()
        if any(ign in clean for ign in ["panda", "gripper", "rethink", "robot", "mount"]):
            color = (50, 50, 55)  # Muted dark slate for robot body
        elif "table" in clean or "floor" in clean:
            color = (15, 15, 18)  # Deep neutral for table/floor
        elif "bowl" in clean or "pot" in clean:
            color = PALETTE["pot"]
        else:
            color = None
            for key, val in PALETTE.items():
                if key in clean:
                    color = val
                    break
            if color is None:
                np.random.seed(inst_id * 17)
                color = tuple(int(c) for c in np.random.randint(60, 230, size=3))

        colored_seg[mask] = color

    return colored_seg


def extract_instances_from_segmentation(seg_img, id2name):
    """Extracts bounding boxes and masks for each recognized object instance (excluding robot/background)."""
    instances = {}
    for inst_id, name in id2name.items():
        clean = name.lower()
        # Exclude robot arm and table structures from bounding-box detection
        if any(ign in clean for ign in ["panda", "gripper", "rethink", "robot", "table", "floor", "mount"]):
            continue

        mask = (seg_img == inst_id).astype(np.uint8)
        if np.sum(mask) < 20:  # Ignore trivial or hidden artifacts
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 15]
        if not valid_contours:
            continue

        # Get combined bounding box of all valid contours belonging to this instance
        all_pts = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_pts)
        instances[name] = {
            "mask": mask,
            "bbox": (x, y, w, h),
            "area": int(np.sum(mask)),
            "center": (x + w // 2, y + h // 2),
        }
    return instances


def draw_recognition_overlay(base_bgr, instances, grounded_objects, grounded_locations):
    """Overlays transparent segmentation masks, bounding boxes, and grounded tags."""
    overlay = base_bgr.copy()
    mask_blend = base_bgr.copy()

    # Pre-clean grounded names for flexible matching
    norm_objs = [o.lower().replace(" ", "_") for o in grounded_objects]
    norm_locs = [l.lower().replace(" ", "_") for l in grounded_locations]

    for name, data in instances.items():
        clean_name = name.lower()
        # Canonical display name
        display_name = "pot" if "bowl" in clean_name else name

        # Find matching color
        if "pot" in display_name or "bowl" in clean_name:
            color = PALETTE["pot"]
        else:
            color = PALETTE.get("default")
            for key, val in PALETTE.items():
                if key in clean_name:
                    color = val
                    break

        # Check if grounded as task object or target location
        is_obj = any(k in clean_name or clean_name in k for k in norm_objs)
        is_loc = any(k in clean_name or clean_name in k for k in norm_locs) or (
            ("bowl" in clean_name or "pot" in clean_name) and any("pot" in k for k in norm_locs)
        )

        # 1. Fill mask
        mask = data["mask"] > 0
        mask_blend[mask] = color

        # 2. Draw contour boundary
        contours, _ = cv2.findContours(data["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)

        # 3. Draw Bounding Box & Pill Label
        x, y, w, h = data["bbox"]
        box_thickness = 2 if (is_obj or is_loc) else 1
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, box_thickness, cv2.LINE_AA)

        # Tag text & badge color
        if is_obj:
            tag = f"[OBJ] {display_name}"
            badge_bg = (0, 130, 0)       # Dark Green
            badge_fg = (255, 255, 255)
        elif is_loc:
            tag = f"[LOC] {display_name}"
            badge_bg = (160, 80, 0)      # Deep Blue
            badge_fg = (255, 255, 255)
        else:
            tag = f"[DISTRACTOR] {display_name}"
            badge_bg = (45, 45, 50)      # Subtle Gray
            badge_fg = (190, 190, 190)

        # Draw label badge
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.44
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(tag, font, font_scale, thickness)

        label_y = max(y - 6, th + 6)
        cv2.rectangle(
            overlay,
            (x, label_y - th - 4),
            (x + tw + 6, label_y + baseline + 1),
            badge_bg,
            -1,
        )
        cv2.rectangle(
            overlay,
            (x, label_y - th - 4),
            (x + tw + 6, label_y + baseline + 1),
            color,
            1,
        )
        cv2.putText(
            overlay,
            tag,
            (x + 3, label_y - 2),
            font,
            font_scale,
            badge_fg,
            thickness,
            cv2.LINE_AA,
        )

    # Alpha blend mask layer (35% opacity)
    cv2.addWeighted(mask_blend, 0.35, overlay, 0.65, 0, overlay)
    return overlay


def build_composite_dashboard(raw_bgr, seg_bgr, rec_bgr, prompt: str, ambiguity_info: dict):
    """Stitches panels into a clean, presentation-ready 3-panel visualization dashboard."""
    h, w, _ = raw_bgr.shape

    # Panel labels
    def add_panel_label(img, title):
        banner = img.copy()
        cv2.rectangle(banner, (0, 0), (w, 32), (30, 30, 30), -1)
        cv2.putText(
            banner,
            title,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return banner

    p1 = add_panel_label(raw_bgr, "1. VLM Input Frame (Raw)")
    p2 = add_panel_label(seg_bgr, "2. Instance Segmentation")
    p3 = add_panel_label(rec_bgr, "3. Recognition & Visual Grounding")

    # Combine side by side
    panels = np.hstack([p1, p2, p3])

    # Add top dashboard header
    header_h = 80
    total_w = panels.shape[1]
    header = np.full((header_h, total_w, 3), 22, dtype=np.uint8)

    # Title
    cv2.putText(
        header,
        "VLM PERCEPTION & VISUAL GROUNDING DASHBOARD (QWEN 2.5-VL)",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    # Full prompt line (no truncation)
    prompt_str = f"Task Command: \"{prompt}\""
    cv2.putText(
        header,
        prompt_str,
        (16, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    # Ambiguity status badge
    amb_status = "AMBIGUOUS (RESOLVED VIA CLARIFICATION)" if ambiguity_info.get("was_ambiguous") else "UNAMBIGUOUS SPECIFICATION"
    amb_color = (0, 165, 255) if ambiguity_info.get("was_ambiguous") else (0, 210, 0)
    objs_str = ", ".join(ambiguity_info.get("task_objects", []))
    locs_str = ", ".join(ambiguity_info.get("target_locations", []))
    status_text = f"Status: {amb_status}   |   Grounded Task Objects: [{objs_str}]   |   Target Locations: [{locs_str}]"
    cv2.putText(
        header,
        status_text,
        (16, 69),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        amb_color,
        1,
        cv2.LINE_AA,
    )

    composite = np.vstack([header, panels])
    return composite


def build_recognition_dashboard(raw_bgr, rec_bgr, prompt: str, ambiguity_info: dict):
    """Creates a clean 2-panel presentation dashboard showing Input RGB Frame -> Visual Grounding Output."""
    h, w, _ = raw_bgr.shape

    def add_panel_label(img, title):
        banner = img.copy()
        cv2.rectangle(banner, (0, 0), (w, 32), (25, 25, 28), -1)
        cv2.putText(
            banner,
            title,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return banner

    p1 = add_panel_label(raw_bgr, "(a) Input: Raw Overhead Camera Observation")
    p2 = add_panel_label(rec_bgr, "(b) Output: Visual Recognition & Grounding (Qwen 2.5-VL)")

    panels = np.hstack([p1, p2])

    header_h = 76
    total_w = panels.shape[1]
    header = np.full((header_h, total_w, 3), 20, dtype=np.uint8)

    cv2.putText(
        header,
        "VLM VISUAL RECOGNITION & SYMBOLIC GROUNDING",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    prompt_str = f"Task Command: \"{prompt}\""
    cv2.putText(
        header,
        prompt_str,
        (16, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    amb_status = "AMBIGUOUS (RESOLVED VIA CLARIFICATION)" if ambiguity_info.get("was_ambiguous") else "UNAMBIGUOUS SPECIFICATION"
    amb_color = (0, 165, 255) if ambiguity_info.get("was_ambiguous") else (0, 210, 0)
    objs_str = ", ".join(ambiguity_info.get("task_objects", []))
    locs_str = ", ".join(ambiguity_info.get("target_locations", []))
    status_text = f"Status: {amb_status}   |   Grounded Task Objects: [{objs_str}]   |   Target Locations: [{locs_str}]"
    cv2.putText(
        header,
        status_text,
        (16, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        amb_color,
        1,
        cv2.LINE_AA,
    )

    composite = np.vstack([header, panels])
    return composite


def main():
    args = parse_args()

    # Determine prompt
    task_prompt = args.prompt
    if args.ambiguous:
        task_prompt = "Put the can in the bin."

    print("\n" + "=" * 76)
    print(" 🤖 VLM PERCEPTION & AMBIGUITY RESOLUTION TEST PIPELINE")
    print("=" * 76)
    print(f" • Model:          {args.vlm}")
    print(f" • Initial Prompt: \"{task_prompt}\"")
    print("=" * 76 + "\n")

    # 1. Acquire Image & Segmentation
    exp_dir = os.path.join(repo_root, "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    temp_image_path = os.path.join(exp_dir, "perception_input_camera.jpg")

    if args.image and os.path.exists(args.image):
        print(f"🖼️ Loading provided input image: {args.image}")
        raw_bgr = cv2.imread(args.image)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(temp_image_path, raw_bgr)
        # Fallback dummy segmentation if standalone image
        gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        seg_img = thresh.astype(np.int32)
        id2name = {255: "segmented_region"}
        scene_objects = ["scene_image"]
    else:
        raw_rgb, seg_img, id2name, scene_objects = capture_from_sim()
        raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_image_path, raw_bgr)
        print(f"📸 Captured camera frame saved to: {temp_image_path}")

    # Display detected scene objects in simulator
    print(f"\n🌍 [Physical Scene Entities ({len(scene_objects)})]:")
    for idx, name in enumerate(scene_objects, 1):
        print(f"   {idx}. {name}")

    # 2. Run VLM Perception & Ambres Loop
    print(f"\n🧠 Querying VLM ({args.vlm}) for Visual Grounding & Ambiguity Analysis...")
    vision_pipeline = ambresFewShotPrompt(vlm_name=args.vlm)
    vision_pipeline.reset_chat()

    input_data = {
        "task_description": task_prompt,
        "image_path": temp_image_path,
    }

    result = vision_pipeline.handle_query_dict(input_data)

    was_ambiguous = bool(result.get("task_ambiguous"))

    # Initial SceneGrounding parse
    scene_grounding = SceneGrounding(
        task_objects=result.get("task_objects", []),
        target_locations=result.get("target_locations", []),
    )

    print("\n🔍 [Initial VLM Scene Grounding]:")
    print(f"   • Ambiguous Task?      {was_ambiguous}")
    print(f"   • Manipulable Objects: {scene_grounding.task_objects}")
    print(f"   • Target Locations:    {scene_grounding.target_locations}")

    # 3. Interactive Clarification Loop if Ambiguous
    if was_ambiguous:
        question = result.get("clarifying_question", "Could you please specify which object and location you mean?")
        print("\n" + "─" * 70)
        print(" ❓ AMBIGUITY DETECTED BY VLM")
        print("─" * 70)
        print(f" [Robot Asks]: {question}")
        print("─" * 70)

        # Accept user clarification
        user_response = input(" [Your Clarification]: ").strip()
        if not user_response:
            user_response = "The red can into the sorting bin."
            print(f" [Default Used]: {user_response}")

        print("\n🔄 Updating Visual Grounding with clarification...")
        result = vision_pipeline.handle_response(user_response)
        scene_grounding = SceneGrounding(
            task_objects=result.get("task_objects", []),
            target_locations=result.get("target_locations", []),
        )

    # Deduplicate while preserving order
    final_objects = list(dict.fromkeys(scene_grounding.task_objects))
    final_locations = list(dict.fromkeys(scene_grounding.target_locations))

    # 4. Terminal Output of Grounded Symbols
    print("\n" + "=" * 76)
    print(" 🎯 FINAL GROUNDED SYMBOLS OUTPUT (SceneGrounding Schema)")
    print("=" * 76)

    print("\n📦 MANIPULABLE OBJECTS (PDDL Type: 'obj'):")
    if final_objects:
        for idx, name in enumerate(final_objects, 1):
            pddl_symbol = name.strip().replace(" ", "_")
            print(f"   ({idx}) Natural Name: \"{name}\"  ==>  PDDL Symbol: \033[92m{pddl_symbol}\033[0m")
    else:
        print("   (None identified)")

    print("\n📍 TARGET RECEPTACLES / LOCATIONS (PDDL Type: 'location'):")
    if final_locations:
        for idx, name in enumerate(final_locations, 1):
            pddl_symbol = name.strip().replace(" ", "_")
            print(f"   ({idx}) Natural Name: \"{name}\"  ==>  PDDL Symbol: \033[94m{pddl_symbol}\033[0m")
    else:
        print("   (None identified)")
    print("=" * 76)

    # 5. Image Segmentation & Recognition Visualization
    print("\n🎨 Generating Segmentation & Recognition Visual Dashboard...")
    seg_bgr = create_segmentation_colormap(seg_img, id2name)
    instances = extract_instances_from_segmentation(seg_img, id2name)
    rec_bgr = draw_recognition_overlay(raw_bgr, instances, final_objects, final_locations)

    ambiguity_info = {
        "was_ambiguous": was_ambiguous,
        "task_objects": final_objects,
        "target_locations": final_locations,
    }
    dashboard = build_composite_dashboard(raw_bgr, seg_bgr, rec_bgr, task_prompt, ambiguity_info)

    # Save output
    save_path = os.path.abspath(args.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, dashboard)
    print(f"💾 Visual perception dashboard saved to: \033[96m{save_path}\033[0m")

    # Also automatically save presentation-ready assets (both PDF and PNG) to Docs/pictures/
    docs_pic_dir = os.path.join(repo_root, "Docs", "pictures")
    os.makedirs(docs_pic_dir, exist_ok=True)
    
    # 2-panel presentation dashboard (Raw RGB -> Grounded Output)
    dashboard_2panel = build_recognition_dashboard(raw_bgr, rec_bgr, task_prompt, ambiguity_info)
    pres_2panel_png = os.path.join(docs_pic_dir, "perception_recognition_grounding.png")
    cv2.imwrite(pres_2panel_png, dashboard_2panel)
    Image.fromarray(cv2.cvtColor(dashboard_2panel, cv2.COLOR_BGR2RGB)).save(
        os.path.join(docs_pic_dir, "perception_recognition_grounding.pdf"), "PDF", resolution=150.0
    )

    # Single overlay panel
    pres_overlay_png = os.path.join(docs_pic_dir, "perception_grounding_overlay.png")
    cv2.imwrite(pres_overlay_png, rec_bgr)
    Image.fromarray(cv2.cvtColor(rec_bgr, cv2.COLOR_BGR2RGB)).save(
        os.path.join(docs_pic_dir, "perception_grounding_overlay.pdf"), "PDF", resolution=150.0
    )

    # Raw camera frame
    cv2.imwrite(os.path.join(docs_pic_dir, "perception_raw_camera.png"), raw_bgr)
    Image.fromarray(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)).save(
        os.path.join(docs_pic_dir, "perception_raw_camera.pdf"), "PDF", resolution=150.0
    )

    # 3-panel composite (for completeness)
    cv2.imwrite(os.path.join(docs_pic_dir, "perception_segmentation_recognition.png"), dashboard)
    Image.fromarray(cv2.cvtColor(dashboard, cv2.COLOR_BGR2RGB)).save(
        os.path.join(docs_pic_dir, "perception_segmentation_recognition.pdf"), "PDF", resolution=150.0
    )
    print(f"🖼️  Presentation figure updated: \033[96m{os.path.join(docs_pic_dir, 'perception_recognition_grounding.pdf')}\033[0m")

    # Interactive GUI display
    if not args.no_gui:
        has_display = os.environ.get("DISPLAY") is not None
        if has_display:
            window_name = "VLM Perception & Ambiguity Resolution"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1440, 520)
            cv2.imshow(window_name, dashboard)
            print("\n👀 Showing visualization window. Press any key or 'q' to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("ℹ️ Headless environment detected (no $DISPLAY); skipped live window.")

    print("\n✨ Perception & Ambiguity Resolution test completed successfully.\n")


if __name__ == "__main__":
    main()
