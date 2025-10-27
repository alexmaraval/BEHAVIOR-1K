import os
import sys
from pathlib import Path

import math
import numpy as np
import torch as th

sys.path.insert(0, "BEHAVIOR-1K/OmniGibson")

import omnigibson as og
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
from omnigibson.macros import gm
from omnigibson.envs.env_wrapper import EnvironmentWrapper

from omnigibson.learning.utils.eval_utils import (TASK_NAMES_TO_INDICES, generate_basic_environment_config)
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
            return self.bonus_completed_subtask, (self.current_index >= len(self.tasks)), info

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
        self.cfg = config.get("config")
        assert self.cfg is not None, "You must pass the main config object under the 'config' key in config."
        self.task_name = self.cfg.task.name
        self.motor_type = motor_type
        self.max_steps = max_steps
        self.instance_id = instance_id
        self.use_domain_randomization = use_domain_randomization
        self._robot = None

        # Set up headless mode and video path from config
        gm.HEADLESS = self.cfg.headless
        if self.cfg.write_video:
            self.video_path = Path(self.cfg.log_path).expanduser() / "videos"
            self.video_path.mkdir(parents=True, exist_ok=True)
            self._video_writer = None

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
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"),
                  "r") as f:
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
        robot_type = "R1Pro"  # cfg.robot.type TODO
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
            base = {"name": "HolonomicBaseJointController", "motor_type": "position", "pos_kp": 50,
                    "command_input_limits": None,
                    "command_output_limits": None,
                    "use_impedances": False}
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
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
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
        for sub_task_map in (self._task_stages or []):
            sub_task = sub_task_map.get("factory")
            task_obj = sub_task(termination_config={"max_steps": self.max_steps})
            task_obj.reset(self._env)
            self.subtasks.append(task_obj)
        self.task_combo = TaskCombination(self.subtasks or [], bonus_completed_subtask=10.0,
                                          sparse_early_sub_goals=False)
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

        # Create sub task info
        subtask_info = {
            "name": None,
            "index": self._stage_idx,
            "reward": 0.0,
            "done": False,
            "success": False,
            "timeout": False,
            "falling": False,
            "max_collision": False
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
                max_collision=False if success else max_collision
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


# ----- Utilities to drive the example code-----

def build_transform(theta, pos_xy, z=0.0):
    """
    Create a 4x4 homogeneous transform for rotation around Z and translation in XY plane
    z: robot height
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, pos_xy[0]],
        [np.sin(theta), np.cos(theta), 0, pos_xy[1]],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])


def get_transformed_action(row, base_pos, yaw2d):
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
    """Return a Rich Table object representing current stage states."""
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
    import json
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
    env = TaskEnv(config={"config": config}, instance_id=1)
    stages = get_sub_tasks(task_name=config.task.name)

    # --- Loop over episodes ---
    for ep_idx, parquet_path in enumerate(
            tqdm(parquet_files, desc="Episodes", unit="episode"), start=1
    ):
        instance_id = int((int(parquet_path.stem.split("_")[-1]) // 10) % 1e3)

        # Re-use environment
        env.instance_id = instance_id
        env.load_robot()
        env.load_task_instance()
        env.set_subtasks()
        obs = env.reset()

        df = pd.read_parquet(parquet_path)
        stage_states = [{"name": s["name"], "reward": 0.0, "status": "pending"} for s in stages]
        stage_states[0]["status"] = "active"

        step_rewards = []
        step_success = False

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
                step_rewards.append(float(sub_task_info["reward"]))

                # Move to the next stage
                if sub_task_info["done"] and has_next_stage:
                    stage_states[next_idx]["status"] = "active"

                live.update(make_table(stage_states))

                # Stop if all done
                if info.get("all_subtasks_complete", False):
                    console.print("[green]All subtasks completed!")
                    step_success = True
                    break

        # --- Save metrics ---
        if write_rewards:
            metrics = {
                "file_name": parquet_path.name,
                "reward": step_rewards,
                "is_successful": step_success,
            }
            metrics_file = out_dir / "metrics.jsonl"
            with open(metrics_file, "a", encoding="utf-8") as fjsonl:
                fjsonl.write(json.dumps(metrics) + "\n")

            console.print(f"[blue]Metrics saved to {metrics_file}")

        console.rule(f"[green]Episode {ep_idx} finished")

    env.close()

    console.print("[bold cyan] All requested episodes completed successfully.")
