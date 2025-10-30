import math
import torch as th
from omnigibson.object_states import AttachedTo
from omnigibson.object_states.robot_related_states import IsGrasping
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.tasks.custom_task_base import BaseTask
from omnigibson.termination_conditions.falling import Falling, ObjectFalling
from omnigibson.termination_conditions.termination_condition_base import SuccessCondition
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
from omnigibson.utils.python_utils import classproperty


class _GraspSuccess(SuccessCondition):
    """Success when the specified object is currently grasped by any arm."""

    def __init__(self, obj_name: str, robot_idn: int = 0):
        self._obj_name = obj_name
        self._robot_idn = int(robot_idn)
        super().__init__()

    def _is_grasping_target(self, env) -> bool:
        robot = env.robots[self._robot_idn]
        obj = env.scene.object_registry("name", self._obj_name)
        if obj is None:
            return False

        if IsGrasping in robot.states and robot.states[IsGrasping].get_value(obj):
            return True

        if AttachedTo in obj.states and obj.states[AttachedTo].get_value(robot):
            return True

        return False

    def _step(self, task, env, action):
        return self._is_grasping_target(env)


class _SimpleGraspReward(BaseRewardFunction):
    """
    Minimal grasp reward:
    - +r_grasp when the target object is currently grasped
    - +exp(-dist) * dist_coeff as approach shaping before grasp
    """

    def __init__(
            self, obj_name: str,
            dist_coeff: float = 0.001,
            r_grasp: float = 1.0,
            collision_penalty: float = 1.0,
            transform_matrix=None
    ):
        self._obj_name = obj_name
        self._dist_coeff = dist_coeff
        self._r_grasp = r_grasp
        self._collision_penalty = collision_penalty
        self.transform_matrix = transform_matrix
        self._initial_dist = None
        super().__init__()

    def _eef_pos(self, env):
        robot = env.robots[0]
        return th.as_tensor(robot.get_eef_position(robot.default_arm), dtype=th.float32)

    def _is_grasping(self, robot, obj) -> bool:
        if IsGrasping in robot.states and robot.states[IsGrasping].get_value(obj):
            return True

        if AttachedTo in obj.states and obj.states[AttachedTo].get_value(robot):
            return True

        return False

    def reset(self, task, env):
        self._initial_dist = None

    def _step(self, task, env, action):
        obj = env.scene.object_registry("name", self._obj_name)
        robot = env.robots[0]
        max_steps = getattr(env, "max_episode_steps", 100)
        max_shaping_per_step = self._dist_coeff

        if obj is None:
            # Still penalize collisions even if object is missing
            coll = detect_robot_collision_in_sim(robot)
            pen = (-self._collision_penalty) if coll else 0.0
            return pen, {
                "grasp_success": False,
                "missing_object": True,
                "collision": bool(coll),
                "collision_penalty": pen,
            }

        eef = self._eef_pos(env)
        goal_pos = th.as_tensor(obj.get_position_orientation()[0], dtype=th.float32)

        if self._initial_dist is None:
            self._initial_dist = th.norm(self._eef_pos(env) - goal_pos)

        if self.transform_matrix is not None:
            goal_pos = th.from_numpy(self.transform_matrix).float() @ goal_pos

        grasping = self._is_grasping(robot, obj)
        if grasping:
            robot = env.robots[0]
            coll = detect_robot_collision_in_sim(robot, filter_objs=[obj])
            pen = (-self._collision_penalty) if coll else 0.0
            return self._r_grasp + pen, {"grasp_success": True, "collision": bool(coll), "collision_penalty": pen}

        dist = th.norm(eef - goal_pos)
        shaped = math.exp(-float(dist)) * self._dist_coeff
        shaped = shaped * (self._r_grasp / (max_steps * max_shaping_per_step))
        robot = env.robots[0]
        coll = detect_robot_collision_in_sim(robot, filter_objs=[obj])
        pen = (-self._collision_penalty) if coll else 0.0
        return shaped + pen, {
            "grasp_success": False,
            "dist": float(dist),
            "shaping": shaped,
            "collision": bool(coll),
            "collision_penalty": pen,
        }


class RobustGraspTask(BaseTask):
    """
    Minimal GraspTask: succeeds when target object is grasped; provides simple approach shaping and grasp bonus.
    - No scene randomization; no joint sampling; only releases grasps on reset.
    - Termination: timeout and grasp success.
    - Rewards: simple shaping + success bonus.
    """

    def __init__(
            self,
            obj_name: str,
            robot_idn: int = 0,
            termination_config=None,
            reward_config=None,
            transform_matrix=None,
    ):
        self._obj_name = obj_name
        self._robot_idn = int(robot_idn)
        self.transform_matrix = transform_matrix
        super().__init__(termination_config=termination_config, reward_config=reward_config)

    def _create_termination_conditions(self):
        return {
            "timeout": Timeout(max_steps=self._termination_config["max_steps"]),
            "graspgoal": _GraspSuccess(obj_name=self._obj_name),
            "falling": Falling(robot_idn=self._robot_idn, fall_height=self._termination_config["fall_height"]),
            "object_falling": ObjectFalling(obj_name=self._obj_name, fall_height=self._termination_config["fall_height"]),
        }

    def _create_reward_functions(self):
        cfg = self._reward_config
        return {
            "grasp": _SimpleGraspReward(
                obj_name=self._obj_name,
                dist_coeff=cfg["dist_coeff"],
                r_grasp=cfg["r_grasp"],
                collision_penalty=cfg["collision_penalty"],
                transform_matrix=self.transform_matrix,
            )
        }

    def reset(self, env):
        # Release any existing grasps and reset reward/terminations
        robot = env.robots[self._robot_idn]
        for arm in getattr(robot, "arm_names", []):
            robot.release_grasp_immediately(arm=arm)

        super().reset(env)

    @classproperty
    def default_termination_config(cls):
        return {
            "max_collisions": 500,
            "max_steps": 500,
            "fall_height": 0.03,
        }

    @classproperty
    def default_reward_config(cls):
        return {"dist_coeff": 0.001, "r_grasp": 1.0, "collision_penalty": 1.0}
