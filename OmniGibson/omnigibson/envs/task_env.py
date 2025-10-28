import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import math
import numpy as np
import torch as th
import cv2

sys.path.insert(0, "BEHAVIOR-1K/OmniGibson")

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
from omnigibson.macros import gm
from omnigibson.envs.env_wrapper import EnvironmentWrapper

from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES, generate_basic_environment_config
from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    load_available_tasks,
    generate_robot_config,
    get_task_relevant_room_types,
)
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.tasks.task_factory import get_sub_tasks
from omnigibson.robots import BaseRobot
from rich.table import Table

from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    generate_basic_environment_config,
    flatten_obs_dict,
    TASK_NAMES_TO_INDICES,
)

from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    write_video,
)

gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True


class TaskCombination:
    """
    Managing and executing a sequence of subtasks in combination.
    """

    def __init__(
            self, tasks: list, bonus_completed_subtask: float = 10.0, sparse_early_sub_goals: bool = False
    ) -> None:
        """
        Initialize the TaskCombination with a sequence of subtasks.
        Args:
            tasks: List of task instances to be executed sequentially. Each must implement
            `reset(env)` and `step(env, action)` methods.
            bonus_completed_subtask: Reward bonus given after completing each subtask.
            sparse_early_sub_goals: If True, suppresses dense rewards.
        """
        self.tasks = tasks
        self.current_index = 0
        self.bonus_completed_subtask = bonus_completed_subtask
        self.sparse_early_sub_goals = sparse_early_sub_goals

    def reset(self, env) -> None:
        """
        Reset all subtasks and start from the first one.
        Args:
            env: The environment instance to which all subtasks belong.

        Returns:
            None
        """
        self.current_index = 0
        for task in self.tasks:
            task.reset(env)

    def step(self, env, action) -> tuple[float, bool, dict]:
        """
        Perform one step in the current active subtask.
        Args:
            env: The environment in which the tasks operate.
            action: The action to perform, passed directly to the current subtask's `step` method.

        Returns:
            reward : The reward from the current subtask, possibly modified by `sparse_early_sub_goals`
            or replaced by `bonus_completed_subtask` upon completion.
            done : True if all subtasks have been completed, False otherwise.
            info : Additional information dictionary, typically propagated from the active subtask.

        """
        if self.current_index >= len(self.tasks):
            return 0.0, True, {"done": {"success": True}}

        reward, done, info = self.tasks[self.current_index].step(env, action)

        # Sparse early subtasks if requested
        if self.sparse_early_sub_goals and (self.current_index < len(self.tasks) - 1):
            reward = 0.0
        if done:
            self.current_index += 1
            if info["done"]["success"]:
                return self.bonus_completed_subtask, (self.current_index >= len(self.tasks)), info
            else:
                return -self.bonus_completed_subtask, (self.current_index >= len(self.tasks)), info

        return float(reward), False, info


class TaskEnv:
    """
    A wrapper environment for managing a multi-stage robotic task composed of multiple subtasks.
    """

    def __init__(
            self,
            config: dict[str, ...],
            motor_type: str = "position",
            instance_id: int | None = None,
            max_steps: int | None = None,
            use_domain_randomization: bool = False,
    ) -> None:
        """
        Initialize the TaskEnv environment and load all required components.
        Args:
            config: Configuration dictionary containing a key `'config'` with environment parameters.
            motor_type: Robot control mode, e.g., `"position"` or `"velocity"`. Default is `"position"`.
            instance_id: Scene instanceID of a specific task instance to load.
            max_steps: Maximum number of simulation steps before termination.
            use_domain_randomization: Whether to apply domain randomization. Default is False.
        """
        self.obs = None
        self.cfg = config.get("config")
        assert self.cfg is not None, "You must pass the main config object under the 'config' key in config."
        self.task_name = self.cfg.task.name
        self.motor_type = motor_type
        self.max_steps = max_steps
        self.instance_id = instance_id
        self.use_domain_randomization = use_domain_randomization
        self._robot = None
        self.robot_type = "R1Pro"

        # Set up headless mode and video path from config
        gm.HEADLESS = self.cfg.headless
        if self.cfg.write_video:
            video_path = Path(self.cfg.log_path).expanduser() / "videos"
            video_path.mkdir(parents=True, exist_ok=True)
            date_str= datetime.now().strftime("%Y%m%d")
            video_name = str(video_path) + f"/{self.task_name}_{date_str}.mp4"
            self._video_writer = create_video_writer(fpath=video_name,resolution=(448, 672))

        self.subtasks = []
        self._task_stages = None
        self.task_combo: TaskCombination | None = None
        self.active_subtask = 0

        self._subtask = None
        self._stage_idx = 0
        self._completed = False

        self._env = self.load_env()
        self.load_robot()
        self.load_task_instance()
        self.set_subtasks()
        self.reset()

    def _prepare_config(self) -> dict[str, ...]:
        """
        Prepare the simulator configuration for the given task.
        Returns:
            A finalized configuration dictionary ready for initializing the simulation environment.
        """
        human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]

        task_idx = TASK_NAMES_TO_INDICES[self.task_name]

        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for k in human_stats.keys():
                    human_stats[k].append(episode[k])

        # take a mean
        for k in human_stats.keys():
            human_stats[k] = sum(human_stats[k]) / len(human_stats[k])

        # Load the seed instance by default
        available_tasks = load_available_tasks()
        task_cfg = available_tasks[self.task_name][0]
        cfg = generate_basic_environment_config(task_name=self.task_name, task_cfg=task_cfg)

        relevant_rooms = get_task_relevant_room_types(activity_name=self.task_name)
        relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], self.task_name)
        cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            generate_robot_config(
                task_name=self.task_name,
                task_cfg=task_cfg,
            )
        ]
        # Update observation modalities
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]  # TODO include more or take it as arg
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        cfg["task"]["termination_config"]["max_steps"] = int(human_stats["length"] * 2)

        # Override env-level frequencies if requested
        if self.max_steps is not None:
            cfg["task"]["termination_config"]["max_steps"] = self.max_steps
        else:
            self.max_steps = cfg["task"]["termination_config"]["max_steps"]
        if self.motor_type == "position":
            base = {
                "name": "HolonomicBaseJointController",
                "motor_type": "position",
                "pos_kp": 50,
                "command_input_limits": None,
                "command_output_limits": None,
                "use_impedances": False,
            }
            cfg["robots"][0]["controller_config"]["base"] = base
        return cfg

    def load_env(self) -> EnvironmentWrapper:
        """
        Load and initialize the simulation environment.
        Returns:
            A wrapped simulation environment ready for interaction.
        """
        cfg = self._prepare_config()
        _env = og.Environment(configs=cfg)
        _env = EnvironmentWrapper(env=_env)
        return _env

    def load_robot(self) -> BaseRobot:
        """
        Retrieve the robot instance from the simulation environment.
        Returns:
            None
        """
        self._robot = self.env.scene.object_registry("name", "robot_r1")

    def load_task_instance(self) -> None:
        """
        Loads the configuration for a specific task instance.
        Returns:
            None
        """
        scene_model = self._env.task.scene_name
        tro_filename = self._env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self._env.task.activity_name,
            activity_definition_id=self._env.task.activity_definition_id,
            activity_instance_id=self.instance_id,
        )
        tro_file_path = os.path.join(
            get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{self._env.task.activity_name}_instances/{tro_filename}-tro_state.json",
        )

        if self.use_domain_randomization:
            scene_data = self.randomize_scene_instances(Path(tro_file_path).parent)
        else:
            with open(tro_file_path, "r") as f:
                scene_data = json.load(f)

        tro_state = recursively_convert_to_torch(scene_data)
        for tro_key, state_data in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = state_data
                robot_pos = presampled_robot_poses[self._robot.model_name][0]["position"]
                robot_quat = presampled_robot_poses[self._robot.model_name][0]["orientation"]
                self._robot.set_position_orientation(robot_pos, robot_quat)
                # Write robot poses to scene metadata
                self._env.scene.write_task_metadata(key=tro_key, data=state_data)
            else:
                self._env.task.object_scope[tro_key].load_state(state_data, serialized=False)

        # Try to ensure that all task-relevant objects are stable
        # They should already be stable from the sampled instance, but there is some issue where loading the state
        # causes some jitter (maybe for small mass / thin objects?)
        for _ in range(25):
            og.sim.step_physics()
            for entity in self._env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self._env.scene.update_initial_file()
        self._env.scene.reset()

    @property
    def env(self) -> EnvironmentWrapper:
        """Return the underlying wrapped simulation environment."""
        return self._env

    @property
    def stage_index(self) -> int:
        """Return the index of the currently active subtask."""
        return self._stage_idx

    @property
    def num_stages(self) -> int:
        """Return the total number of subtasks in the current task."""
        return len(self.subtasks)

    @property
    def done(self) -> bool:
        """Return whether all subtasks in the task have been completed."""
        return self._completed

    def set_subtasks(self) -> None:
        """
        Initialize and configure all subtasks for the current task.
        Returns:
            None
        """
        self._task_stages = get_sub_tasks(task_name=self.task_name)
        self.subtasks = []
        for sub_task_map in self._task_stages or []:
            sub_task = sub_task_map.get("factory")
            task_obj = sub_task(termination_config={"max_steps": self.max_steps})
            task_obj.reset(self._env)
            self.subtasks.append(task_obj)
        self.task_combo = TaskCombination(
            self.subtasks or [], bonus_completed_subtask=10.0, sparse_early_sub_goals=False
        )
        self._reset_subtask_progress()

    def reset(self) -> dict[str, ...]:
        """
        Reset the full environment and all subtasks.
        Returns:
             The initial observation from the environment after reset.
        """
        obs, info = self._env.reset()

        self.load_task_instance()

        if self.task_combo is not None:
            self.task_combo.reset(self._env)
        self._reset_subtask_progress()
        return obs

    def step(self, action: th.Tensor) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one environment step and update subtask progress.
        Args:
            action: The control action to apply to the environment.

        Returns:
            (obs, reward_env, terminated_env, truncated_env, info_out)
            where:
                - obs : Observation after the action.
                - reward_env : Environment-level reward signal.
                - terminated_env : Whether the episode is terminated.
                - truncated_env : Whether the episode was truncated.
                - info_out : Detailed info including subtask progress and completion flags.

        """
        obs, reward_env, terminated_env, truncated_env, info_env = self._env.step(action)
        self.obs = self._preprocess_obs(obs)
        self._write_video()

        # Create sub task info
        subtask_info = {
            "name": None,
            "index": self._stage_idx,
            "reward": 0.0,
            "done": False,
            "success": False,
            "timeout": False,
            "falling": False,
            "max_collision": False,
        }

        if self.task_combo is not None and self.subtasks:
            rew_s, combo_done, info_s = self.task_combo.step(env=self._env, action=action)
            info_s = info_s or {}
            success = bool(info_s.get("done", {}).get("success", False))
            self._stage_idx = min(self.task_combo.current_index, len(self.subtasks))
            name = None
            if self._stage_idx < len(self.subtasks):
                name = self._task_stages[self._stage_idx]["name"]

            falling = False
            max_collision = False
            timeout = False
            try:
                if "falling" in info_s["done"]["termination_conditions"]:
                    falling = info_s["done"]["termination_conditions"]["falling"]["done"]
                if "max_collision" in info_s["done"]["termination_conditions"]:
                    max_collision = info_s["done"]["termination_conditions"]["max_collision"]["done"]
                if "timeout" in info_s["done"]["termination_conditions"]:
                    timeout = info_s["done"]["termination_conditions"]["timeout"]["done"]
            except Exception as e:
                print(f" stage : {name}, {e}")

            subtask_info.update(
                name=name,
                reward=float(rew_s),
                done=success or combo_done,
                success=success,
                timeout=False if success else timeout,
                falling=False if success else falling,
                max_collision=False if success else max_collision,
            )

            if combo_done:
                self._completed = True
        else:
            self._completed = True

        # Compose info
        info_out = info_env
        info_out["subtask"] = subtask_info
        info_out["all_subtasks_complete"] = self._completed

        return obs, reward_env, terminated_env, truncated_env, info_out

    def close(self) -> None:
        """
        Close the simulation environment and perform cleanup.
        Returns:
            None
        """
        self._video_writer = None
        self._env.close()
        og.shutdown()

    def _reset_subtask_progress(self) -> None:
        """
        Reset internal tracking of subtask progress.
        Returns:
            None
        """
        self._subtask = None
        self._stage_idx = 0
        self._completed = False

    def _preprocess_obs(self, obs: dict) -> dict:
        """
        Preprocess the observation dictionary before passing it to the policy.
        Args:
            obs (dict): The observation dictionary to preprocess.

        Returns:
            dict: The preprocessed observation dictionary.
        """
        obs = flatten_obs_dict(obs)
        base_pose = self._robot.get_position_orientation()
        cam_rel_poses = []
        # The first time we query for camera parameters, it will return all zeros
        # For this case, we use camera.get_position_orientation() instead.
        # The reason we are not using camera.get_position_orientation() by defualt is because it will always return the most recent camera poses
        # However, since og render is somewhat "async", it takes >= 3 render calls per step to actually get the up-to-date camera renderings
        # Since we are using n_render_iterations=1 for speed concern, we need the correct corresponding camera poses instead of the most update-to-date one.
        # Thus, we use camera parameters which are guaranteed to be in sync with the visual observations.
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = self._robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        return obs

    def _write_video(self) -> None:
        """
        Write the current robot observations to video.
        """
        # concatenate obs
        left_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        head_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy(),
            (448, 448),
        )
        write_video(
            np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0),
            video_writer=self._video_writer,
            batch_size=1,
            mode="rgb",
        )

    @staticmethod
    def collect_tokens_and_entries(data: dict) -> tuple[dict[str, dict], dict]:
        """
        Extracts object entries and robot pose information from a scene JSON dictionary
        Args:
            data: A dictionary loaded from a scene JSON file.

        Returns:
            A tuple containing:
              - entries: Mapping of object names to their state definitions.
              - robot_poses: Mapping of robot types to their corresponding pose data.

        """
        entries = {}
        robot_poses = data.get("robot_poses")
        for k, v in data.items():
            if k == "robot_poses":
                continue
            if isinstance(v, dict) and ("root_link" in v or "args" in v):
                entries[k] = v
        return entries, robot_poses

    def randomize_scene_instances(self, instances_dir) -> dict[str, dict]:
        """
        Randomly combines object and robot configurations from multiple scene instance files.
        Args:
            instances_dir: Path to the directory containing scene instance JSON files.

        Returns:
            A dictionary representing the randomized scene configuration.

        """
        instances_dir = Path(instances_dir)
        cache_path = Path("random_scenes") / "per_file_entries.json"

        if cache_path.exists():
            per_file_entries = json.loads(cache_path.read_text())
        else:
            scene_files = sorted(instances_dir.glob("*.json"))
            if not scene_files:
                raise FileNotFoundError(f"No JSON instance files found in {instances_dir}")

            per_file_entries = {}

            for scene_file in scene_files:
                data = json.loads(scene_file.read_text())
                entries, robot_poses = TaskEnv.collect_tokens_and_entries(data)

                for obj, state in {**entries, **robot_poses}.items():
                    per_file_entries.setdefault(obj, [])
                    if state not in per_file_entries[obj]:
                        per_file_entries[obj].append(state)

            # Save to file
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(per_file_entries, indent=2))

        # Use the first scene as a template
        first_scene = sorted(instances_dir.glob("*.json"))[0]
        scene_data = json.loads(first_scene.read_text())

        # Randomly pick one variant for each object
        for key in scene_data.keys():
            if key == "robot_poses":
                robot_states = per_file_entries.get(self.robot_type, [])
                if robot_states:
                    scene_data[key][self.robot_type] = random.choice(robot_states)
            else:
                if key in per_file_entries and per_file_entries[key]:
                    scene_data[key] = random.choice(per_file_entries[key])

        return scene_data


# ----- Utilities to drive the example code-----

def build_transform(theta, pos_xy, z=0.0) -> np.ndarray:
    """
    Build a 4×4 homogeneous transformation matrix representing a rotation around the Z-axis
    and a translation in the XY plane.
    Args:
        theta: Rotation angle in radians about the Z-axis.
        pos_xy: XY translation components [x, y].
        z: Z translation component (height).

    Returns:
        A 4×4 homogeneous transformation matrix

    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, pos_xy[0]],
            [np.sin(theta), np.cos(theta), 0, pos_xy[1]],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )


def get_transformed_action(row, base_pos, yaw2d) -> np.ndarray:
    """
     Compute a robot action transformed into a base-relative coordinate frame.
    Args:
        row: A dictionary containing observation and action.
        base_pos: Original action vector to be updated.
        yaw2d: Base yaw angle (rotation about Z-axis) in radians.

    Returns:
        The transformed action vector
    """
    pos = base_pos.detach().cpu().numpy().tolist()
    z = 0  # Z coordinate
    tm_1 = build_transform(yaw2d.detach().cpu().numpy().tolist(), pos, z)

    theta2 = row["observation.state"][149]  # rotation
    pos_xy2 = row["observation.state"][140:142]  # XY
    tm_2 = build_transform(theta2, pos_xy2, z)

    tm_new = np.linalg.inv(tm_1) @ tm_2  # chaining transformations
    action_translation = tm_new[:2, 3]
    action_rotation = math.atan2(tm_new[1, 0], tm_new[0, 0])

    action = row["action"].copy()
    action[0:2] = action_translation
    action[2] = action_rotation
    return action


def make_table(stage_states):
    """
    Create a Rich-formatted table displaying progress and rewards for multiple stages.
    Args:
        stage_states: List of stage state dictionaries.

    Returns:
        A Rich Table object ready for console rendering.
    """
    table = Table(title="Stage Progress", expand=True)
    table.add_column("Idx", justify="right")
    table.add_column("Stage", justify="left")
    table.add_column("Status", justify="center")
    table.add_column("Reward", justify="right")

    for idx, st in enumerate(stage_states):
        state = stage_states[idx]
        if state["status"] == "completed":
            status = "[green]done[/green]"
        elif state["status"] == "active":
            status = "[yellow]active[/yellow]"
        elif state["status"] == "Failed":
            status = "[red]active[/red]"
        else:
            status = "[grey]pending[/grey]"

        reward_str = f"{state['reward']:.3f}"
        table.add_row(str(idx), st["name"], status, reward_str)

    return table


if __name__ == "__main__":
    """
    Usage:
    # Run a single episode
    python BEHAVIOR-1K/OmniGibson/omnigibson/envs/task_env.py policy=local task.name=cook_bacon \
        log_path=./ +parquet="/home/jiacheng/b1k-baselines/data/data/task-0046/episode_00460040.parquet" headless=false

    # Run all episodes in a directory
    python BEHAVIOR-1K/OmniGibson/omnigibson/envs/task_env.py policy=local task.name=cook_bacon \
        log_path=./ +parquet_dir="/home/jiacheng/b1k-baselines/data/data/task-0046/" headless=true \
        +run_all=true +write_rewards=true        
    """
    import sys
    import torch as th
    import pandas as pd
    from inspect import getsourcefile
    from tqdm import tqdm
    from rich.console import Console
    from rich.live import Live
    import hydra
    from task_env import TaskEnv

    with hydra.initialize_config_dir(
            f"{Path(getsourcefile(lambda: 0)).parents[0].parent}/learning/configs", version_base="1.1"
    ):
        config = hydra.compose("base_config.yaml", overrides=sys.argv[1:])

    console = Console()

    # --- Determine mode ---
    run_all = getattr(config, "run_all", False)
    write_rewards = getattr(config, "write_rewards", False)

    if run_all:
        parquet_dir = Path(config.get("parquet_dir", config.get("parquet", "")))
        parquet_files = sorted(parquet_dir.glob("*.parquet"))
        console.rule(f"[bold cyan]Running ALL episodes from {parquet_dir}")
    else:
        parquet = Path(config.get("parquet", ""))
        parquet_files = [parquet]
        console.rule(f"[bold cyan]Running SINGLE episode: {parquet.name}")

    # --- Output directory ---
    out_dir = Path(config.get("log_path", "./logs")) / config.task.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize environment ---
    env = TaskEnv(config={"config": config}, instance_id=1, use_domain_randomization=False)
    stages = get_sub_tasks(task_name=config.task.name)

    # --- Loop over episodes ---
    for ep_idx, parquet_path in enumerate(tqdm(parquet_files, desc="Episodes", unit="episode"), start=1):
        instance_id = int((int(parquet_path.stem.split("_")[-1]) // 10) % 1e3)

        # Re-use environment
        env.instance_id = instance_id
        env.load_robot()
        env.load_task_instance()
        env.set_subtasks()
        obs = env.reset()

        step_rewards = {}
        step_success = {}

        df = pd.read_parquet(parquet_path)
        stage_states = [{"name": s["name"], "reward": 0.0, "status": "pending"} for s in stages]
        stage_states[0]["status"] = "active"
        step_rewards.update({stage_states[0]['name']:[]})
        step_success.update({stage_states[0]['name']:[]})


        with Live(make_table(stage_states), console=console, refresh_per_second=4) as live:
            for _, row in df.iterrows():
                base_pos = obs["robot_r1"]["proprio"][140:142]
                yaw2d = obs["robot_r1"]["proprio"][149]
                action = th.from_numpy(get_transformed_action(row, base_pos, yaw2d))

                obs, reward_env, terminated_env, truncated_env, info = env.step(action)
                sub_task_info = info["subtask"]
                idx = sub_task_info["index"]
                next_idx = idx + 1
                has_next_stage = next_idx < len(stage_states)

                # Update stage info
                if sub_task_info["done"]:
                    stage_states[idx]["status"] = "completed"
                elif any(sub_task_info.get(k, False) for k in ("falling", "max_collision", "timeout")):
                    console.print("[red]Sub task terminated due to collision/falling/timeout")
                    stage_states[idx]["status"] = "failed"

                stage_states[idx]["reward"] = sub_task_info["reward"]
                step_rewards[stage_states[idx]['name']].append(float(sub_task_info["reward"]))
                step_success[stage_states[idx]['name']].append(int(sub_task_info["done"]))

                # Move to the next stage
                if sub_task_info["done"] and has_next_stage:
                    stage_states[next_idx]["status"] = "active"
                    step_rewards.update({stage_states[next_idx]['name']: []})
                    step_success.update({stage_states[next_idx]['name']: []})

                live.update(make_table(stage_states))

                # Stop if all done
                if info.get("all_subtasks_complete", False):
                    console.print("[green]All subtasks completed!")
                    break

        # --- Save metrics ---
        if write_rewards:
            metrics = {
                "file_name": parquet_path.name,
                "reward": step_rewards,
                "is_successful": step_success,
            }
            date_str = datetime.now().strftime("%Y%m%d")
            metrics_file = out_dir / f"rewards_{date_str}.jsonl"
            with open(metrics_file, "a", encoding="utf-8") as fjsonl:
                fjsonl.write(json.dumps(metrics) + "\n")

            console.print(f"[blue]Metrics saved to {metrics_file}")

        console.rule(f"[green]Episode {ep_idx} finished")

    env.close()

    console.print("[bold cyan] All requested episodes completed successfully.")
