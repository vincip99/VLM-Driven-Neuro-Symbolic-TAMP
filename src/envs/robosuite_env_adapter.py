"""
RobosuiteEnvAdapter
===================
Wraps robomimic's `EnvRobosuite` (loaded from a BC checkpoint) and exposes the
interface that BT skill nodes expect:

    get_obs()                          → dict  (policy-ready obs)
    step(action)                       → (obs, reward, done, info)
    reset()                            → dict
    is_grasped(obj_name)               → bool
    is_in_bin(obj_name, bin_name)      → bool
    get_camera_frame(camera_name)      → np.ndarray  HxWx3 uint8
    get_scene_description()            → dict  (for VLM prompt)

The adapter loads the env directly from the checkpoint's env_meta so the
observation space, controller, and robot config are guaranteed to match what
the policy was trained on.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Gripper-closure threshold (Panda-specific)
# Panda finger joints: fully open ~0.04 each → sum ~0.08
#                      gripping an object    → sum < 0.06
# ---------------------------------------------------------------------------
_GRIPPER_CLOSED_THRESHOLD = 0.078   # sum of both finger qpos (m) (open ~0.08, grasping object ~0.07)
_LIFT_MARGIN = 0.035                # metres above table surface to count as "lifted" (table surface ~0.80)
_BIN_XY_HALF = 0.14               # half-width of the bin footprint (m)


class RobosuiteEnvAdapter:
    """
    Thin wrapper around robomimic's EnvRobosuite.

    Parameters
    ----------
    ckpt_path : str
        Path to the robomimic checkpoint (.pth).  The environment is
        reconstructed from the env_meta stored inside the checkpoint so obs
        keys, controller, and robot match the trained policy exactly.
    render_offscreen : bool
        If True, enables off-screen rendering for camera captures.
        Requires MUJOCO_GL to be set (e.g. 'egl') before importing mujoco.
    verbose : bool
        Print extra info about the env on creation.
    """

    def __init__(self, ckpt_path: Optional[str] = None, raw_env: Optional[Any] = None, render_offscreen: bool = True, render_onscreen: bool = False, verbose: bool = True):
        self.ckpt_path = ckpt_path
        self._env = None          # robomimic EnvRobosuite instance
        self._base_env = None     # underlying robosuite environment
        self._table_height: float = 0.8
        self._obj_body_ids: dict = {}   # name → mujoco body id
        self._bin_pos: Optional[np.ndarray] = None

        if raw_env is not None:
            self._base_env = raw_env
            self._env = raw_env
            self._table_height = float(getattr(self._base_env, "table_offset", np.array([0, 0, 0.8]))[2])
            self._refresh_body_ids()
        elif ckpt_path is not None:
            self._build_env(render_offscreen=render_offscreen, render_onscreen=render_onscreen, verbose=verbose)

    # ── construction ──────────────────────────────────────────────────────

    def _build_env(self, render_offscreen: bool, render_onscreen: bool, verbose: bool):
        import robomimic.utils.obs_utils as ObsUtils
        from robomimic.utils.file_utils import (
            env_from_checkpoint,
            config_from_checkpoint,
            load_dict_from_checkpoint,
        )

        # Load the raw checkpoint dict once so we can init ObsUtils
        # before env_from_checkpoint (which doesn't do it itself)
        ckpt_dict = load_dict_from_checkpoint(self.ckpt_path)
        
        # --- OVERRIDE ENV NAME ---
        # We override the environment name in the checkpoint metadata here
        # so that it simulates TaskSorting instead of the training environment (PickPlaceCan).
        if "env_metadata" in ckpt_dict:
            ckpt_dict["env_metadata"]["env_name"] = "TaskSorting"
            if "env_kwargs" in ckpt_dict["env_metadata"]:
                ckpt_dict["env_metadata"]["env_kwargs"]["env_name"] = "TaskSorting"

        algo_name = ckpt_dict["algo_name"]
        config, _ = config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=False)

        # MUST be called before env.get_observation() so OBS_KEYS_TO_MODALITIES is set
        ObsUtils.initialize_obs_utils_with_config(config)

        env, _ = env_from_checkpoint(
            ckpt_dict=ckpt_dict,
            render=render_onscreen,
            render_offscreen=render_offscreen,
            verbose=verbose,
        )
        self._env = env
        self._base_env = env.env      # the raw robosuite env
        self._table_height = float(getattr(self._base_env, "table_offset",
                                           np.array([0, 0, 0.8]))[2])
        if verbose:
            print(f"[RobosuiteEnvAdapter] env type : {type(self._base_env).__name__}")
            print(f"[RobosuiteEnvAdapter] table_height: {self._table_height:.3f} m")

    def _refresh_body_ids(self):
        """Cache mujoco body IDs for all scene objects after each reset."""
        sim = self._base_env.sim
        self._obj_body_ids.clear()

        # PickPlaceCan objects ─────────────────────────────────────────────
        if hasattr(self._base_env, "obj_body_id"):
            for k, v in self._base_env.obj_body_id.items():
                if not k.startswith("Visual"):
                    self._obj_body_ids[k] = v

        # Resolve bin positions ────────────────────────────────────────────
        # PickPlaceCan stores bin1_pos / bin2_pos set in _reset_internal
        if hasattr(self._base_env, "bin1_pos"):
            self._bin_pos = np.array(self._base_env.bin1_pos)

        # TaskSorting objects ──────────────────────────────────────────────
        for attr, key in [
            ("red_can_body_id", "red_can"),
            ("blue_can_body_id", "blue_can"),
            ("green_can_body_id", "green_can"),
            ("sorting_bin_id", "sorting_bin"),
            ("bowl_id", "bowl"),
            ("yellow_cube_body_id", "yellow_cube"),
            ("purple_cube_body_id", "purple_cube"),
        ]:
            if hasattr(self._base_env, attr):
                self._obj_body_ids[key] = getattr(self._base_env, attr)

        # Dynamic registration for any objects in TaskSorting / custom envs
        if hasattr(self._base_env, "objects"):
            for obj in self._base_env.objects:
                try:
                    bid = sim.model.body_name2id(obj.root_body)
                    self._obj_body_ids[obj.name] = bid
                except Exception:
                    pass

        # Register aliases for TaskSorting container bodies
        if "bowl" in self._obj_body_ids:
            self._obj_body_ids["pot"] = self._obj_body_ids["bowl"]
        if "sorting_bin" in self._obj_body_ids:
            self._obj_body_ids["bin"] = self._obj_body_ids["sorting_bin"]

    # ── public interface ───────────────────────────────────────────────────

    def reset(self) -> dict:
        obs = self._env.reset()
        self._refresh_body_ids()
        self._init_obj_z = {}
        for name, bid in self._obj_body_ids.items():
            try:
                self._init_obj_z[name] = float(self._base_env.sim.data.body_xpos[bid][2])
            except Exception:
                pass
        return obs

    def get_obs(self) -> dict:
        """Return current observation dict (policy-ready format)."""
        if hasattr(self._env, "get_observation"):
            return self._env.get_observation()
        if hasattr(self._base_env, "_get_observations"):
            return self._base_env._get_observations()
        return {}

    def step(self, action: np.ndarray):
        """Apply action to the simulation."""
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, info

    # ── termination checks ────────────────────────────────────────────────

    def is_grasped(self, obj_name: str) -> bool:
        """
        Return True when the object has been lifted above the table surface by the gripper.
        """
        body_id = self._resolve_body_id(obj_name)
        if body_id is None:
            return False

        # An object already resting inside a target container is placed, not held by the gripper
        if self.is_in_bin(obj_name, "bin") or self.is_in_bin(obj_name, "pot") or self.is_in_bin(obj_name, "sorting_bin"):
            return False

        obj_pos = np.array(self._base_env.sim.data.body_xpos[body_id])

        # Verify robot gripper proximity (< 0.12m)
        try:
            eef_site_id = self._base_env.robots[0].eef_site_id["right"]
            eef_pos = self._base_env.sim.data.site_xpos[eef_site_id]
            if np.linalg.norm(eef_pos - obj_pos) > 0.12:
                return False
        except Exception:
            pass

        # Verify object has been lifted above its resting height
        init_z = getattr(self, "_init_obj_z", {}).get(obj_name, None)
        if init_z is not None:
            return bool(obj_pos[2] > init_z + 0.02)

        margin = 0.045 if "can" in str(obj_name).lower() or "cylinder" in str(obj_name).lower() else _LIFT_MARGIN
        return bool(obj_pos[2] > self._table_height + margin)

    def is_in_bin(self, obj_name: str, bin_name: str = "bin") -> bool:
        """
        Return True when *obj_name* is located within the XY footprint of the
        bin and is at a reasonable height (resting in the bin, not floating).
        """
        body_id = self._resolve_body_id(obj_name)
        if body_id is None:
            return False

        obj_pos = np.array(self._base_env.sim.data.body_xpos[body_id])

        # Resolve bin position ─────────────────────────────────────────────
        bin_pos = self._resolve_bin_pos(bin_name)
        if bin_pos is None:
            return False

        xy_dist = np.abs(obj_pos[:2] - bin_pos[:2])
        # Bin footprint is ~0.15m half-width, pot with handles is ~0.16m
        radius = 0.18 if str(bin_name).lower() in ["pot", "bowl"] else 0.16
        in_xy = np.all(xy_dist < radius)
        # Object should be below the top of the container area (not floating high above)
        in_z = obj_pos[2] < (bin_pos[2] + 0.25)
        return bool(in_xy and in_z)

    # ── camera / VLM helpers ──────────────────────────────────────────────

    def get_camera_frame(self, camera_name: str = "agentview",
                          height: int = 256, width: int = 256) -> np.ndarray:
        """
        Return an HxWx3 uint8 RGB frame from the named camera.
        Falls back to 'frontview' if the requested camera is unavailable.
        """
        try:
            frame = self._base_env.sim.render(
                height=height, width=width, camera_name=camera_name
            )
            # robosuite render() returns RGB, origin at top-left
            return frame
        except Exception as exc:
            print(f"[RobosuiteEnvAdapter] render failed ({exc}), trying frontview")
            try:
                return self._base_env.sim.render(
                    height=height, width=width, camera_name="frontview"
                )
            except Exception:
                return np.zeros((height, width, 3), dtype=np.uint8)

    def get_scene_description(self) -> dict:
        """
        Build a scene description dict suitable for the VLM prompt.
        Object positions are taken from the live simulation state.
        """
        sim = self._base_env.sim
        objects_meta = {}
        for name, bid in self._obj_body_ids.items():
            pos = sim.data.body_xpos[bid].tolist()
            objects_meta[name] = {"position": pos}

        bin_pos = self._bin_pos.tolist() if self._bin_pos is not None else [0, 0, 0]

        return {
            "semantic_map_locations": {
                "table_area": {"position": [0.0, 0.0, self._table_height]},
                "bin_area":   {"position": bin_pos},
            },
            "objects_metadata": objects_meta,
            "assets_metadata": {"bin": {"position": bin_pos}},
            "asset_object_relations": {
                "table": list(objects_meta.keys()),
                "bin":   [],
            },
            "location_asset_relations": {
                "table_area": ["table"],
                "bin_area":   ["bin"],
            },
        }

    # ── internal helpers ──────────────────────────────────────────────────

    def _resolve_body_id(self, obj_name: str) -> Optional[int]:
        """Look up a body ID by object name (fuzzy match if needed)."""
        name = str(obj_name).lower().strip()
        if name in ["pot", "bowl", "potwithhandles"]:
            for k in ["bowl", "pot"]:
                if k in self._obj_body_ids:
                    return self._obj_body_ids[k]
        if name in ["bin", "sorting_bin", "sorting bin"]:
            for k in ["sorting_bin", "bin"]:
                if k in self._obj_body_ids:
                    return self._obj_body_ids[k]
        if name in self._obj_body_ids:
            return self._obj_body_ids[name]
        if obj_name in self._obj_body_ids:
            return self._obj_body_ids[obj_name]
        # fuzzy: match any key that starts with obj_name or vice-versa
        for k, v in self._obj_body_ids.items():
            if name.startswith(k) or k.startswith(name) or obj_name.startswith(k) or k.startswith(obj_name):
                return v
        # last resort: ask mujoco directly
        try:
            return self._base_env.sim.model.body_name2id(obj_name)
        except Exception:
            print(f"[RobosuiteEnvAdapter] body '{obj_name}' not found — "
                  f"known: {list(self._obj_body_ids)}")
            return None

    def _resolve_bin_pos(self, bin_name: str = "bin") -> Optional[np.ndarray]:
        """Return the XYZ position of the bin/container body."""
        name = str(bin_name).lower().strip()

        # Check pot / bowl first
        if name in ["pot", "bowl", "potwithhandles"]:
            for k in ["bowl", "pot"]:
                if k in self._obj_body_ids:
                    return np.array(self._base_env.sim.data.body_xpos[self._obj_body_ids[k]])
            if hasattr(self._base_env, "bowl_id"):
                return np.array(self._base_env.sim.data.body_xpos[self._base_env.bowl_id])

        # Check bin / sorting_bin
        if name in ["bin", "sorting_bin", "sorting bin"]:
            for k in ["sorting_bin", "bin"]:
                if k in self._obj_body_ids:
                    return np.array(self._base_env.sim.data.body_xpos[self._obj_body_ids[k]])
            if hasattr(self._base_env, "sorting_bin_id"):
                return np.array(self._base_env.sim.data.body_xpos[self._base_env.sorting_bin_id])

        # Direct dictionary match
        if name in self._obj_body_ids:
            return np.array(self._base_env.sim.data.body_xpos[self._obj_body_ids[name]])

        # TaskSorting general fallback
        if "sorting_bin" in self._obj_body_ids and name in ["bin", "sorting_bin"]:
            return np.array(self._base_env.sim.data.body_xpos[self._obj_body_ids["sorting_bin"]])

        # PickPlaceCan
        if self._bin_pos is not None:
            return self._bin_pos

        # Try direct mujoco lookup
        for candidate in [bin_name, "bin1", "bin2", "VisualBin", "bowl", "sorting_bin"]:
            try:
                bid = self._base_env.sim.model.body_name2id(candidate)
                return np.array(self._base_env.sim.data.body_xpos[bid])
            except Exception:
                continue
        return None
