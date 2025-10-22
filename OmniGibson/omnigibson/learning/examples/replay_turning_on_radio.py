import argparse
import json
import json as pyjson
import os
from pathlib import Path
import sys
from copy import deepcopy
from pathlib import Path

import math
import numpy as np
import pandas as pd
import torch as th
from rich.console import Console
from rich.live import Live
from tqdm import tqdm

sys.path.insert(0, "BEHAVIOR-1K/OmniGibson")

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import Pose
from omnigibson.tasks.grasp_task import GraspTask
from omnigibson.tasks.point_reaching_task import PointReachingTask

from env_utils import build_env, load_task_instance, get_transformed_action, make_table, setup_task, task_setup

import omnigibson as og

from omnigibson.tasks.custom_tasks import (
    MoveBaseToObjectTask,
    OnTask,
    MoveEEToObjectTask,
    OnTopTask,
    RobustGraspTask,
    _get_named,
    _front_target,
)

gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

name_radio = "radio_89"
name_coffe_table = "coffee_table_koagbh_0"

@task_setup(goal_type="nav", target=name_radio, front_offset=0)
def make_move_to_radio(prev_reward, env):
    return MoveBaseToObjectTask(
        target_object_name=name_radio,
        goal_tolerance=1.3,
        termination_config={"max_steps": 10000},
        include_obs=False,
    )

@task_setup(goal_type=None, target=name_radio)
def make_grasp_radio(prev_reward, env):
    reward_config = {"collision_penalty": 0.000000001}
    task = RobustGraspTask(
        obj_name=name_radio,
        termination_config={"max_steps": 10000},
        reward_config=reward_config,
        include_obs=False,
        objects_config=[],
    )
    return task


@task_setup(goal_type=None, target=name_radio)
def make_radio_on(prev_reward, env):
    reward_config = {"r_offset": prev_reward or 0.0}
    task = OnTask(
        target_object_name=name_radio,
        # reward_config=reward_config,
        termination_config={"max_steps": 10000},
        include_obs=False,
    )
    return task

def get_sub_stages_factory():
    stages = [
        {"name": "move_to_radio", "kind": "task", "factory": make_move_to_radio},
        {"name": "pick_radio", "kind": "task", "factory": make_grasp_radio},
        {"name": "radio_on", "kind": "task", "factory": make_radio_on},
        # {"name": "place_radio", "kind": "task", "factory": place_radio},
    ]
    return stages

console = Console()


def run_episode(parquet, env, instance_id, task_name):

    obs, _ = env.reset()
    load_task_instance(env, instance_id)

    df = pd.read_parquet(parquet)


    # Stage list with either subtask objects or predicate callables
    stages = get_sub_stages_factory()

    stage_states = [{"name": s["name"], "reward": 0.0, "status": "pending"} for s in stages]

    episode_id = Path(parquet).stem
    out_dir = Path(f"reward_logs/{task_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_logs = {s["name"]: {"reward": [], "success": []} for s in stages}

    stage_i = 0
    stage_states[stage_i]["status"] = "active"
    prev_reward = 0.0
    sub_task = None

    step_rewards = []
    step_success = []
    sub_task_status = [False] * len(stages)

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
                sub_task = stage["factory"](prev_reward, env)

            rew_s, done_s, info_s = sub_task.step(env=env, action=action)

            completed = bool(info_s.get("done", {}).get("success", False)) if isinstance(info_s, dict) else False

            current_reward = float(rew_s)
            stage_states[stage_i]["reward"] = current_reward

            # Log reward & success
            stage_logs[stage["name"]]["reward"].append(current_reward)
            stage_logs[stage["name"]]["success"].append(int(completed))

            step_rewards.append(current_reward)
            step_success.append(1 if completed else 0)

            if completed:
                stage_states[stage_i]["status"] = "completed"
                sub_task_status[stage_i] = True
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

    # record = {"episode": episode_id, "stages": stage_logs}
    # with open(out_dir / f"{episode_id}.json", "w", encoding="utf-8") as f:
    #     pyjson.dump(record, f)

    if not all(sub_task_status):
        return None, None
    return step_rewards, step_success




def main():
    task_name = "turning_on_radio"
    dataset_path = "/home/jiacheng/b1k-baselines/data/data/task-0000/"
    i = 0
    env = build_env(activity_definition_id=0, instance_id=0, activity_name=task_name,
                    scene="house_double_floor_lower")

    out_dir = Path(task_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rewards = []
    pgb_episode = tqdm(sorted(os.listdir(dataset_path)))
    for f in pgb_episode:
        pgb_episode.set_description(f"{f}")
        # print(f)
        # if f != "episode_00000050.parquet":
        #     continue
        episode_index = Path(f)
        instance_id = int(episode_index.stem.split("_")[-1])
        instance_id = int((instance_id // 10) % 1e3)

        step_rewards, step_success = run_episode(os.path.join(dataset_path, f), env, instance_id, task_name)
        if step_rewards is None:
            continue
        metrics = {
            "file_name": f,
            "reward": step_rewards,
            "is_successful": step_success,
        }
        all_rewards.append(metrics)
        with open(out_dir / "metrics_21_10.jsonl", "a", encoding="utf-8") as fjsonl:
                fjsonl.write(json.dumps(metrics) + "\n")
        # break
        # if i>3:
        #     break
        # else:
        #     i += 1




    # with open(out_dir / "metrics.jsonl", "a", encoding="utf-8") as fjsonl:
    #     for reward in all_rewards:
    #         fjsonl.write(json.dumps(reward) + "\n")

    # with open(out_dir / "metrics.jsonl", "a", encoding="utf-8") as fjsonl:
    #     fjsonl.write(json.dumps(all_rewards) + "\n")


    env.close()
    og.shutdown()



if __name__ == "__main__":
    main()
