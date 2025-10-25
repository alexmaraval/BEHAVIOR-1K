import json
import os
import sys
from typing import Dict, List, Any

import pandas as pd
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
from task_factory import get_sub_tasks

gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True


class TaskCombination:

    def __init__(self, tasks: list, bonus_completed_subtask: float = 10.0, sparse_early_subgoals: bool = False) -> None:
        self.tasks = tasks
        self.current_index = 0
        self.bonus_completed_subtask = bonus_completed_subtask
        self.sparse_early_subgoals = sparse_early_subgoals

    def reset(self, env):
        self.current_index = 0
        for task in self.tasks:
            task.reset(env)

    def step(self, env, action):
        if self.current_index >= len(self.tasks):
            return 0.0, True, {"done": {"success": True}}

        reward, done, info = self.tasks[self.current_index].step(env, action)

        # Sparse early subtasks if requested
        if self.sparse_early_subgoals and (self.current_index < len(self.tasks) - 1):
            reward = 0.0

        if done:
            self.current_index += 1
            return self.bonus_completed_subtask, (self.current_index >= len(self.tasks)), info

        return float(reward), False, info


class TaskEnv:

    def __init__(
            self,
            task_name: str,
            motor_type: str = "position",
            max_steps: int | None = None,
            instance_id: int = 1,
            use_domain_randomization: bool = False,
            subtask_max_steps=10000
    ) -> None:
        self.task_name = task_name
        self.motor_type = motor_type
        self.max_steps = max_steps
        self.instance_id = instance_id
        self.subtask_max_steps = subtask_max_steps
        self.use_domain_randomization = use_domain_randomization

        self.subtasks = []
        self._task_stages = None
        self.task_combo: TaskCombination | None = None
        self.active_subtask = 0

        self._subtask = None
        self._stage_idx = 0
        self._completed = False

        self._env = self.load_env()
        self._robot = self.load_robot()
        self.load_task_instance()
        self.set_subtasks()
        self.reset()

    def _prepare_config(self):
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

        # Override env-level frequencies if requested (kept for completeness)
        if self.max_steps is not None:
            cfg["task"]["termination_config"]["max_steps"] = self.max_steps
        if self.motor_type == "position":
            base = {"name": "HolonomicBaseJointController", "motor_type": "position", "pos_kp": 50,
                    "command_input_limits": None,
                    "command_output_limits": None,
                    "use_impedances": False}
            cfg["robots"][0]["controller_config"]["base"] = base

        return cfg

    def load_env(self):
        cfg = self._prepare_config()
        _env = og.Environment(configs=cfg)
        _env = EnvironmentWrapper(env=_env)
        return _env

    def load_robot(self):
        """
        Loads and returns the robot instance from the environment.
        Returns:
            BaseRobot: The robot instance loaded from the environment.
        """
        robot = self.env.scene.object_registry("name", "robot_r1")
        return robot

    def load_task_instance(self) -> None:
        """
        Loads the configuration for a specific task instance.

        Args:
            instance_id (int): The ID of the task instance to load.
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
        return self._env

    @property
    def stage_index(self) -> int:
        return self._stage_idx

    @property
    def num_stages(self) -> int:
        return len(self.subtasks)

    @property
    def done(self) -> bool:
        return self._completed

    def set_subtasks(self) -> None:
        """
        Set sub-task factories.
        Each item should be a dict: {"name": str, "factory": callable}
        The factory must accept (max_steps: int, env: EnvironmentWrapper) and return a Task instance.
        """
        self._task_stages = get_sub_tasks(task_name=task)
        self.subtasks = []
        for sub_task_map in (self._task_stages or []):
            sub_task = sub_task_map.get("factory")
            task_obj = sub_task(termination_config={"max_steps": 10000})
            task_obj.reset(self._env)
            self.subtasks.append(task_obj)
        self.task_combo = TaskCombination(self.subtasks or [], bonus_completed_subtask=10.0, sparse_early_subgoals=False)
        self._reset_subtask_progress()

    def reset(self) -> Any:
        """
        Reset the underlying OG environment and optionally load a specific instance.
        Returns the observation.
        """
        obs, info = self._env.reset()

        self.load_task_instance()

        if self.task_combo is not None:
            self.task_combo.reset(self._env)
        self._reset_subtask_progress()
        return obs

    def step(self, action: th.Tensor):
        """
        Step with a low-level action into both the OG env and current sub-task.

        Returns: obs, reward_env, terminated_env, truncated_env, info
        info contains sub-task fields:
          info["subtask"] = {
            "name": str,
            "index": int,
            "reward": float,
            "done": bool,
            "success": bool,
          }
        """
        obs, reward_env, terminated_env, truncated_env, info_env = self._env.step(action)

        # Create sub task info
        subtask_info = {
            "name": None,
            "index": self._stage_idx,
            "reward": 0.0,
            "done": False,
            "success": False,
            "timeout":False,
            "falling":False,
            "max_collision":False
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
        self._env.close()
        og.shutdown()

    def _reset_subtask_progress(self) -> None:
        self._subtask = None
        self._stage_idx = 0
        self._completed = False


if __name__ == "__main__":
    from env_utils import get_transformed_action
    from rich.console import Console
    from rich.live import Live
    from env_utils import get_transformed_action, make_table


    task = "cook_bacon"
    parquet = "/home/jiacheng/b1k-baselines/data/data/task-0046/episode_00460010.parquet"
    env = TaskEnv(
        task_name=task,
        instance_id=1,
        subtask_max_steps=5000,
    )

    stages = get_sub_tasks(task_name=task)

    # Reset environment and subtasks
    obs = env.reset()

    # Drive with replay
    df = pd.read_parquet(parquet)

    console = Console()
    stage_states = [{"name": s["name"], "reward": 0.0, "status": "pending"} for s in stages]
    stage_states[0]["status"] = "active"

    with Live(make_table(stage_states), console=console, refresh_per_second=4) as live:
        for _, row in df.iterrows():
            base_pos = obs["robot_r1"]["proprio"][140:142]
            yaw2d = obs["robot_r1"]["proprio"][149]
            action = th.from_numpy(get_transformed_action(row, base_pos, yaw2d))
            obs, reward_env, terminated_env, truncated_env, info = env.step(action)
            sub_task_info = info["subtask"]
            if sub_task_info["done"]:
                stage_states[sub_task_info["index"]]["status"] = "completed"
                if sub_task_info["index"]+1 < len(stage_states):
                    stage_states[sub_task_info["index"]+1]["status"] = "active"
            elif sub_task_info["falling"] or sub_task_info["max_collision"]:
                print("Sub task terminated due to collision/falling")
                stage_states[sub_task_info["index"]]["status"] = "Failed"
                if sub_task_info["index"] + 1 < len(stage_states):
                    stage_states[sub_task_info["index"]+1]["status"] = "active"
            else:
                pass
            stage_states[sub_task_info["index"]]["reward"] = sub_task_info["reward"]

            live.update(make_table(stage_states))
            if info["all_subtasks_complete"] :
                print("All tasks completed")
                break

    env.close()
