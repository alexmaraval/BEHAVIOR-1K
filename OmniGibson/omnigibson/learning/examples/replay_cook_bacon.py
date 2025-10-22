import argparse
import json as pyjson
from pathlib import Path
import sys

import pandas as pd
import torch as th
from rich.console import Console
from rich.live import Live

sys.path.insert(0, "BEHAVIOR-1K/OmniGibson")

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import Pose
from omnigibson.tasks.point_reaching_task import PointReachingTask

from env_utils import build_env, load_task_instance, get_transformed_action, make_table, setup_task, task_setup

from omnigibson.tasks.custom_tasks import (
    MoveBaseToObjectTask,
    OnTask,
    NextToTask,
    MoveEEToObjectTask,
    OnTopTask,
    RobustGraspTask,
    SufficientlyOpenTask,
    SufficientlyClosedTask,
    InsideTask,
    OnTopStableTask,
    _get_named,
    _front_target,
)

gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

name_fridge = "fridge_dszchb_0"
name_countertop = "countertop_kelker_0"
name_pan = "frying_pan_207"
name_tray = "tray_208"
name_burner = "burner_mjvqii_0"
name_bacon_1 = "bacon_209"
name_bacon_2 = "bacon_210"
name_bacon_3 = "bacon_211"
name_bacon_4 = "bacon_212"
name_bacon_5 = "bacon_213"
name_bacon_6 = "bacon_214"


@task_setup(goal_type="nav", target=name_fridge, front_offset=0.3)
def make_move_to_fridge(prev_reward, env):
    reward_config = {
        "r_offset": prev_reward or 0.0,
        "use_normalized_potential": True,
        "r_potential": 1.0,
    }
    return MoveBaseToObjectTask(
        target_object_name=name_fridge,
        goal_tolerance=1,
        # reward_config=reward_config,
        termination_config={"max_steps": 10000},
        include_obs=False,
    )


@task_setup(goal_type=None, target=name_fridge)
def make_open_fridge(prev_reward, env):
    reward_config = {"r_offset": prev_reward or 0.0}
    task = SufficientlyOpenTask(
        target_object_name=name_fridge,
        allowed_deg=90,
        reward_config=reward_config,
        termination_config={"max_steps": 10000},
        include_obs=False,
    )
    return task


@task_setup(goal_type=None, target=name_fridge)
def make_close_fridge(prev_reward, env):
    reward_config = {"r_offset": prev_reward or 0.0}
    task = SufficientlyClosedTask(
        target_object_name=name_fridge,
        allowed_deg=0,
        reward_config=reward_config,
        termination_config={"max_steps": 10000},
        include_obs=False,
    )
    return task


@task_setup(goal_type="nav", target=name_countertop, front_offset=0.3)
def make_move_to_counter_top(prev_reward, env):
    reward_config = {
        "r_collision": 0.00000001,
    }
    return MoveBaseToObjectTask(
        target_object_name=name_countertop,
        goal_tolerance=0.8,
        reward_config=reward_config,
        termination_config={"max_steps": 10000},
        include_obs=False,
    )

@task_setup(goal_type=None, target=name_tray, source=name_burner)
def make_place_next_to_burner(prev_reward, env):
    reward_config = {"r_offset": prev_reward or 0.0}
    task = NextToTask(
        target_object_name=name_tray,
        source_object_name=name_burner,
        desired_value=True,
        # reward_config=reward_config,
        termination_config={"max_steps": 10000},
        include_obs=False,
    )
    return task


@task_setup(goal_type="eef", target=name_pan)
def make_move_to_frying_pan(prev_reward, env):
    # reward_config = {"r_offset": prev_reward or 0.0, "r_potential": 1.0}
    reward_config = {"r_collision": 0.00000001}
    task = MoveEEToObjectTask(
        target_object_name=name_pan,
        goal_tolerance=0.05,
        reward_config=reward_config,
        termination_config={"max_steps": 10000},
        include_obs=False,
    )
    return task


@task_setup(goal_type="eef", target=name_tray)
def make_move_to_tray(prev_reward, env):
    # reward_config = {"r_offset": prev_reward or 0.0, "r_potential": 1.0}
    reward_config = {"r_collision": 0.00000001}
    task = MoveEEToObjectTask(
        target_object_name=name_tray,
        goal_tolerance=0.05,
        reward_config=reward_config,
        termination_config={"max_steps": 10000},
        include_obs=False,
    )
    return task


@task_setup(goal_type=None, target=name_pan, source=name_burner)
def make_place_frying_pan(prev_reward, env):
    # reward_config = {"r_offset": prev_reward or 0.0}
    task = OnTopStableTask(
        target_object_name=name_pan,
        source_object_name=name_burner,
        # desired_value=True,
        # reward_config=reward_config,
        termination_config={"max_steps": 10000},
        include_obs=False,
    )
    return task


@task_setup(goal_type=None, target=name_burner)
def make_burner_on(prev_reward, env):
    reward_config = {"r_offset": prev_reward or 0.0}
    task = OnTask(
        target_object_name=name_burner,
        # reward_config=reward_config,
        termination_config={"max_steps": 10000},
        include_obs=False,
    )
    return task


@task_setup(goal_type=None, target=name_bacon_5, source=name_pan)
def make_pour_tray(prev_reward, env):
    # reward_config = {"r_offset": prev_reward or 0.0}
    task = OnTopTask(
        target_object_name=name_bacon_3,
        source_object_name=name_pan,
        desired_value=True,
        # reward_config=reward_config,
        termination_config={"max_steps": 10000},
        include_obs=False,
    )
    return task


@task_setup(goal_type=None, target=name_tray)
def make_grasp_tray(prev_reward, env):
    # reward_config = {"r_offset": prev_reward or 0.0}
    reward_config = {"collision_penalty": 0.00000001}
    task = RobustGraspTask(
        obj_name=name_tray,
        termination_config={"max_steps": 10000},
        reward_config=reward_config,
        include_obs=False,
        objects_config=[],
    )
    return task


@task_setup(goal_type=None, target=name_pan, source=None)
def make_grasp_pan(prev_reward, env):
    # reward_config = {"r_offset": prev_reward or 0.0}
    reward_config = {"collision_penalty": 0.00000001}
    task = RobustGraspTask(
        obj_name=name_pan,
        termination_config={"max_steps": 10000},
        reward_config=reward_config,
        include_obs=False,
        objects_config=[],
    )
    return task


console = Console()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, required=True)
    args = ap.parse_args()

    env = build_env(activity_definition_id=0, instance_id=0,
                    activity_name="cook_bacon")  # Predefined IDs for episode_00460010
    obs, _ = env.reset()
    load_task_instance(env, 1)

    df = pd.read_parquet(args.parquet)

    # Stage list with either subtask objects or predicate callables
    stages = [
        {"name": "move_to_fridge", "kind": "task", "obj": make_move_to_fridge},
        {"name": "open_fridge", "kind": "task", "obj": make_open_fridge},
        {"name": "pick_tray", "kind": "task", "obj": make_grasp_tray},
        {"name": "close_fridge", "kind": "task", "obj": make_close_fridge},
        {"name": "move_to_counter_top", "kind": "task", "obj": make_move_to_counter_top},
        {"name": "place_on_next_to_burner1", "kind": "task", "obj": make_place_next_to_burner},
        {"name": "Move_to_frying_pan", "kind": "task", "obj": make_move_to_frying_pan},
        {"name": "pick_up_frying_pan", "kind": "task", "obj": make_grasp_pan},
        {"name": "place_frying_pan", "kind": "task", "obj": make_place_frying_pan},
        {"name": "move_to_tray", "kind": "task", "obj": make_move_to_tray},
        {"name": "pick_up_tray", "kind": "task", "obj": make_grasp_tray},
        {"name": "pour_tray", "kind": "task", "obj": make_pour_tray},
        {"name": "place_on_next_to_burner2", "kind": "task", "obj": make_place_next_to_burner},
        {"name": "burner_on_switch", "kind": "task", "obj": make_burner_on},
    ]

    stage_states = [{"name": s["name"], "reward": 0.0, "status": "pending"} for s in stages]

    episode_id = Path(args.parquet).stem
    out_dir = Path("reward_logs/cook_bacon")
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_logs = {s["name"]: {"reward": [], "success": []} for s in stages}

    stage_i = 0
    stage_states[stage_i]["status"] = "active"
    prev_reward = 0.0
    sub_task = None

    with Live(make_table(stage_states), console=console, refresh_per_second=4) as live:
        for _, row in df.iterrows():
            if stage_i >= len(stages):
                break

            base_pos = obs["robot_r1"]["proprio"][140:142]
            yaw2d = obs["robot_r1"]["proprio"][149]
            action = th.from_numpy(get_transformed_action(row, base_pos, yaw2d))

            obs, reward, terminated, truncated, info = env.step(action)

            stage = stages[stage_i]
            if sub_task is None:
                sub_task = stage["obj"](prev_reward, env)

            rew_s, done_s, info_s = sub_task.step(env=env, action=action)

            # Determine completion
            completed = bool(info_s.get("done", {}).get("success", False)) if isinstance(info_s, dict) else False

            current_reward = float(rew_s)
            stage_states[stage_i]["reward"] = current_reward

            # Log reward & success
            stage_logs[stage["name"]]["reward"].append(current_reward)
            stage_logs[stage["name"]]["success"].append(int(completed))

            if completed:
                stage_states[stage_i]["status"] = "completed"

                # prev_reward = current_reward + 0.1

                # Move to next stage
                stage_i += 1
                sub_task = None

                if stage_i < len(stages):
                    stage_states[stage_i]["status"] = "active"
                    # stage_states[stage_i]["reward"] = prev_reward

            # Update live table
            live.update(make_table(stage_states))

            # Early termination
            if terminated or truncated:
                break

    record = {"episode": episode_id, "stages": stage_logs}
    with open(out_dir / f"{episode_id}.json", "w", encoding="utf-8") as f:
        pyjson.dump(record, f)

    env.close()
    og.shutdown()

if __name__ == "__main__":
    main()
