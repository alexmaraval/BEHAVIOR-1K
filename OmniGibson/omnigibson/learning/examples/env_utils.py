import json
import os
import sys

import math
import numpy as np
import torch as th

sys.path.insert(0, "BEHAVIOR-1K/OmniGibson")

import omnigibson as og
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
from omnigibson.macros import gm
from omnigibson.envs.env_wrapper import EnvironmentWrapper

from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.tasks.custom_tasks import _get_named, _front_target
from omnigibson.object_states import Pose
from omnigibson.tasks.point_reaching_task import PointReachingTask
from omnigibson.learning.utils.eval_utils import (TASK_NAMES_TO_INDICES, generate_basic_environment_config)
from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    load_available_tasks,
    generate_robot_config,
    get_task_relevant_room_types,
)

from rich.table import Table

gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True


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

def get_max_steps(task_name):
    human_stats = {
        "length": [],
        "distance_traveled": [],
        "left_eef_displacement": [],
        "right_eef_displacement": [],
    }
    with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
        episodes = [json.loads(line) for line in f]

    task_idx = TASK_NAMES_TO_INDICES[task_name]

    for episode in episodes:
        if episode["episode_index"] // 1e4 == task_idx:
            for k in human_stats.keys():
                human_stats[k].append(episode[k])
    # take a mean
    for k in human_stats.keys():
        human_stats[k] = sum(human_stats[k]) / len(human_stats[k])

    # Load the seed instance by default
    available_tasks = load_available_tasks()
    task_cfg = available_tasks[task_name][0]
    robot_type = "R1Pro" #cfg.robot.type TODO
    cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
    # breakpoint()
    # if cfg["partial_scene_load"]:
    relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
    relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
    cfg["scene"]["load_room_types"] = relevant_rooms

    cfg["robots"] = [
        generate_robot_config(
            task_name=task_name,
            task_cfg=task_cfg,
        )
    ]
    # Update observation modalities
    cfg["robots"][0]["obs_modalities"] = ["proprio"]
    cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
    cfg["task"]["termination_config"]["max_steps"] = int(human_stats["length"] * 2)

    return cfg


def build_env(activity_definition_id: int, instance_id: int, activity_name: str, scene: str = "house_single_floor"):
    cfg = {
        "env": {"action_frequency": 30, "rendering_frequency": 30, "physics_frequency": 120},
        "scene": {"type": "InteractiveTraversableScene", "scene_model": scene},
        "task": {
            "type": "BehaviorTask",
            "activity_name": activity_name,
            "activity_definition_id": int(activity_definition_id),
            "activity_instance_id": int(instance_id),
            "online_object_sampling": False,
            "termination_config": {"max_steps": 100000},
            "include_obs": False,
        },
        "robots": [
            {
                "type": "R1Pro",
                "name": "robot_r1",
                "action_normalize": False,
                "self_collisions": True,
                "grasping_mode": "assisted",
                "proprio_obs": list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
                "obs_modalities": ["proprio"],
                "controller_config": {
                    "arm_left": {"name": "JointController", "motor_type": "position", "pos_kp": 150,
                                 "command_input_limits": None, "command_output_limits": None, "use_impedances": False,
                                 "use_delta_commands": False},
                    "arm_right": {"name": "JointController", "motor_type": "position", "pos_kp": 150,
                                  "command_input_limits": None, "command_output_limits": None, "use_impedances": False,
                                  "use_delta_commands": False},
                    "gripper_left": {"name": "MultiFingerGripperController", "mode": "smooth",
                                     "command_input_limits": "default", "command_output_limits": "default"},
                    "gripper_right": {"name": "MultiFingerGripperController", "mode": "smooth",
                                      "command_input_limits": "default", "command_output_limits": "default"},
                    "base": {"name": "HolonomicBaseJointController", "motor_type": "position", "pos_kp": 50,
                             "command_input_limits": None,
                             "command_output_limits": None,
                             "use_impedances": False},
                    # "base": {"name": "HolonomicBaseJointController", "motor_type": "velocity", "vel_kp": 150,
                    #          "command_input_limits": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                    #          "command_output_limits": [[-0.75, -0.75, -1.0], [0.75, 0.75, 1.0]],
                    #          "use_impedances": False},
                    "trunk": {"name": "JointController", "motor_type": "position", "pos_kp": 150,
                              "command_input_limits": None, "command_output_limits": None, "use_impedances": False,
                              "use_delta_commands": False},
                    "camera": {"name": "NullJointController"},
                },
                "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_height": 1080, "image_width": 1080}}},
            }
        ],
    }
    env = og.Environment(configs=cfg)
    env = EnvironmentWrapper(env=env)

    print("scene_model:", env.scene.scene_model)
    print("scene_instance:", getattr(env.scene, "scene_instance", None))

    # env.robots[0].base_footprint_link.mass = 250.0

    return env


def load_task_instance(env, instance_id):
    robot = env.robots[0]
    scene_model = env.task.scene_name
    tro_filename = env.task.get_cached_activity_scene_filename(
        scene_model=scene_model,
        activity_name=env.task.activity_name,
        activity_definition_id=env.task.activity_definition_id,
        activity_instance_id=instance_id,
    )
    tro_file_path = os.path.join(
        get_task_instance_path(scene_model),
        f"json/{scene_model}_task_{env.task.activity_name}_instances/{tro_filename}-tro_state.json",
    )
    # TODO temp  patch
    # tro_file_path =  "/home/jiacheng/Documents/combined_scene.json"
    with open(tro_file_path, "r") as f:
        tro_state = recursively_convert_to_torch(json.load(f))
    for tro_key, tro_state in tro_state.items():
        if tro_key == "robot_poses":
            presampled_robot_poses = tro_state
            robot_pos = presampled_robot_poses[robot.model_name][0]["position"]
            robot_quat = presampled_robot_poses[robot.model_name][0]["orientation"]
            robot.set_position_orientation(robot_pos, robot_quat)
            # Write robot poses to scene metadata
            env.scene.write_task_metadata(key=tro_key, data=tro_state)
        else:
            env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

    for _ in range(25):
        og.sim.step_physics()
        for entity in env.task.object_scope.values():
            if not entity.is_system and entity.exists:
                entity.keep_still()

    env.scene.update_initial_file()


def setup_task(task, env, goal_type=None, target_name=None, **kwargs):
    # --- Set goal ---
    if goal_type == "nav" and target_name:
        front_offset = kwargs.get("front_offset", 0.6)
        obj = _get_named(env, target_name)
        base_z = env.robots[getattr(task, "_robot_idn", 0)].states[Pose].get_value()[0][2]
        goal_xy = _front_target(obj, offset=front_offset)[:2]
        task._goal_pos = th.tensor([goal_xy[0], goal_xy[1], base_z], dtype=th.float32)
        task._randomize_goal_pos = False

    elif goal_type == "eef" and target_name:
        offset = kwargs.get("offset", 0.12)
        eef_z = env.robots[getattr(task, "_robot_idn", 0)].get_eef_position()[2]
        target = _get_named(env, target_name)
        goal_xy = _front_target(target, offset=offset)[:2]
        task._goal_pos = th.tensor([goal_xy[0], goal_xy[1], eef_z], dtype=th.float32)
        task._randomize_goal_pos = False

    # --- Prime runtime ---
    setattr(task, "_loaded", True)
    try:
        if getattr(task, "_initial_pos", None) is None:
            if isinstance(task, PointReachingTask):
                task._initial_pos = env.robots[getattr(task, "_robot_idn", 0)].get_eef_position()
            else:
                task._initial_pos = env.robots[getattr(task, "_robot_idn", 0)].states[Pose].get_value()[0]
        task._current_robot_pos = task._initial_pos
        if getattr(task, "_path_length", None) is None:
            task._path_length = 0.0

    except Exception as e:
        print(e)

    for term in getattr(task, "_termination_conditions", {}).values():
        term.reset(task, env)

    for rew in getattr(task, "_reward_functions", {}).values():
        rew.reset(task, env)

    return task


def task_setup(goal_type=None, target=None, **setup_kwargs):
    """
    Decorator that wraps a task factory to automatically set up goals and runtime.
    """

    def decorator(factory):
        def wrapper(max_steps, env):
            task = factory(max_steps, env)
            return setup_task(task, env, goal_type=goal_type, target_name=target, **setup_kwargs)

        return wrapper

    return decorator


def set_missing_objects(env):
    # Set tray position
    pos = th.tensor([7.7494, -1.8819, 0.9802], dtype=th.float32)
    quat = th.tensor([-7.9652e-05, -3.5517e-04, 6.3861e-02, 9.9796e-01], dtype=th.float32)
    # quat = quat / th.norm(quat) # TODO why need it ?
    env.scene.object_registry("name", "tray_208").set_position_orientation(position=pos, orientation=quat)

    # Set bacon position
    pos = th.tensor([7.7099, -1.7753, 0.9779], dtype=th.float32)
    quat = th.tensor([0.0129, -0.0029, 0.8781, 0.4783], dtype=th.float32)
    env.scene.object_registry("name", "bacon_209").set_position_orientation(position=pos, orientation=quat)

    pos = th.tensor([7.7940, -1.8130, 0.9782], dtype=th.float32)
    quat = th.tensor([0.0130, -0.0016, 0.9123, 0.4093], dtype=th.float32)
    env.scene.object_registry("name", "bacon_210").set_position_orientation(position=pos, orientation=quat)

    pos = th.tensor([7.8018, -1.8740, 0.9782], dtype=th.float32)
    quat = th.tensor([0.0129, -0.0042, 0.8400, 0.5424], dtype=th.float32)
    env.scene.object_registry("name", "bacon_211").set_position_orientation(position=pos, orientation=quat)

    pos = th.tensor([7.7890, -1.7218, 0.9783], dtype=th.float32)
    quat = th.tensor([-0.0035, -0.0154, -0.6832, 0.7301], dtype=th.float32)
    env.scene.object_registry("name", "bacon_212").set_position_orientation(position=pos, orientation=quat)

    pos = th.tensor([7.6616, -1.8900, 0.9777], dtype=th.float32)
    quat = th.tensor([0.0012, -0.0202, -0.1698, 0.9853], dtype=th.float32)
    env.scene.object_registry("name", "bacon_213").set_position_orientation(position=pos, orientation=quat)

    pos = th.tensor([7.7099, -1.7753, 0.9779], dtype=th.float32)
    quat = th.tensor([0.0129, -0.0029, 0.8781, 0.4783], dtype=th.float32)
    env.scene.object_registry("name", "bacon_214").set_position_orientation(position=pos, orientation=quat)

    pos = th.tensor([7.7167, -1.8593, 0.9779], dtype=th.float32)
    quat = th.tensor([0.0072, -0.0153, 0.1091, 0.9939], dtype=th.float32)
    env.scene.object_registry("name", "bacon_209").set_position_orientation(position=pos, orientation=quat)

    # Set frying pan position
    # pos = th.tensor([4.2329, -0.5367, 0.9322], dtype=th.float32)
    pos = th.tensor([4.35, -0.5367, 0.9322], dtype=th.float32)
    quat = th.tensor([-7.7625e-04, 1.1444e-04, 7.6698e-01, 6.4167e-01], dtype=th.float32)

    def quat_mul(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return th.tensor([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ], dtype=q1.dtype)

    # rotate 30° around Z-axis (handle turns 30° horizontally)
    angle_deg = -65.0  # adjust + or - depending on direction
    angle_rad = math.radians(angle_deg)

    delta_quat = th.tensor([
        0.0,
        0.0,
        math.sin(angle_rad / 2.0),
        math.cos(angle_rad / 2.0)
    ], dtype=th.float32)

    # apply rotation
    new_quat = quat_mul(delta_quat, quat)

    env.scene.object_registry("name", "frying_pan_207").set_position_orientation(position=pos, orientation=new_quat)

    env.scene.object_registry("name", "bacon_209").get_position_orientation()
    # bacon_209 - (tensor([ 7.7099, -1.7753,  0.4779]), tensor([ 0.0129, -0.0029,  0.8781,  0.4783]))
    # bacon_210 - (tensor([ 7.7940, -1.8130,  0.4782]), tensor([ 0.0130, -0.0016,  0.9123,  0.4093]))
    # bacon_211 - (tensor([ 7.8018, -1.8740,  0.4782]), tensor([ 0.0129, -0.0042,  0.8400,  0.5424]))
    # bacon_212 - (tensor([ 7.7890, -1.7218,  0.4783]), tensor([-0.0035, -0.0154, -0.6832,  0.7301]))
    # bacon_213 - (tensor([ 7.6616, -1.8900,  0.4777]), tensor([ 0.0012, -0.0202, -0.1698,  0.9853]))
    # bacon_214 - (tensor([ 7.7167, -1.8593,  0.4779]), tensor([ 0.0072, -0.0153,  0.1091,  0.9939]))


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
