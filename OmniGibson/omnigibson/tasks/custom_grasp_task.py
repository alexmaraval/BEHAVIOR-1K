import omnigibson.utils.transform_utils as T
import torch as th
from omnigibson.object_states import AttachedTo
from omnigibson.object_states.robot_related_states import IsGrasping
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.reward_functions.stand_upright_reward import StandUprightReward
from omnigibson.tasks.custom_task_base import BaseTask
from omnigibson.tasks.task_utils import _MaxCollisionFiltered, SuccessBonusReward
from omnigibson.termination_conditions.falling import Falling, ObjectFalling
from omnigibson.termination_conditions.termination_condition_base import SuccessCondition
from omnigibson.termination_conditions.timeout import Timeout
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
            ori_coeff: float = 0.001,
            transform_matrix=None
    ):
        self._obj_name = obj_name
        self._dist_coeff = dist_coeff
        self._ori_coeff = ori_coeff
        self.transform_matrix = transform_matrix
        self._potential = None
        super().__init__()

    def _eef_position_orientation(self, env):
        robot = env.robots[0]
        pos = th.as_tensor(robot.get_eef_position(robot.default_arm), dtype=th.float32)
        orientation = th.as_tensor(robot.get_eef_orientation(robot.default_arm), dtype=th.float32)

        return pos, orientation

    def _is_grasping(self, robot, obj) -> bool:
        if IsGrasping in robot.states and robot.states[IsGrasping].get_value(obj):
            return True

        if AttachedTo in obj.states and obj.states[AttachedTo].get_value(robot):
            return True

        return False

    def reset(self, task, env):
        self._potential = self._potential_fcn(env)

    def _potential_fcn(self, env):
        eef_pos, eef_orientation = self._eef_position_orientation(env)

        obj = env.scene.object_registry("name", self._obj_name)
        goal_pos, goal_orientation = obj.get_position_orientation()

        # Find EEF transformation
        rotation_matrix_eef = T.quat2mat(eef_orientation)

        # Find Object transformation
        transform_object = th.eye(4)
        transform_object[:3, :3] = T.quat2mat(eef_orientation)
        transform_object[:3, 3] = goal_pos

        transformation_target = transform_object @ self.transform_matrix
        pos_dist = T.l2_distance(transformation_target[:3, 3], eef_pos)
        transformation_target = th.as_tensor(transformation_target, dtype=th.float32)
        rotation_matrix_eef = th.as_tensor(rotation_matrix_eef, dtype=th.float32)

        ori_dist = th.acos((th.trace(transformation_target[:3, :3].T @ rotation_matrix_eef) - 1) / 2)

        return float(pos_dist) * self._dist_coeff + float(ori_dist) * self._ori_coeff

    def _step(self, task, env, action):
        # Reward is proportional to the potential difference between the current and previous timestep
        new_potential = self._potential_fcn(env)
        reward = self._potential - new_potential

        # Update internal potential
        self._potential = new_potential
        return reward, {}


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
            skip_collision_with_objs=None,
    ):
        self._obj_name = obj_name
        self._robot_idn = int(robot_idn)
        self.transform_matrix = transform_matrix
        super().__init__(termination_config=termination_config, reward_config=reward_config)

        self._skip_collision_with_objs_names = skip_collision_with_objs

    def _create_termination_conditions(self):
        return {
            "timeout": Timeout(max_steps=self._termination_config["max_steps"]),
            "graspgoal": _GraspSuccess(obj_name=self._obj_name),
            "falling": Falling(robot_idn=self._robot_idn, fall_height=self._termination_config["fall_height"]),
            "object_falling": ObjectFalling(obj_name=self._obj_name,
                                            fall_height=self._termination_config["fall_height"]),
            "max_collision": _MaxCollisionFiltered(task_ref=self,
                                                   max_collisions=self._termination_config["max_collisions"])
        }

    def _create_reward_functions(self):
        cfg = self._reward_config
        rewards = dict()
        rewards["potential"] = _SimpleGraspReward(
                obj_name=self._obj_name,
                dist_coeff=cfg["dist_coeff"],
                ori_coeff=cfg["ori_coeff"],
                transform_matrix=self.transform_matrix,
            )
        rewards["stand_upright"] = StandUprightReward(
            robot_idn=self._robot_idn, coeff=self._reward_config["r_stand_upright"]
        )
        rewards["graspgoal"] = SuccessBonusReward(
            success_condition=self._termination_conditions["graspgoal"], r_success=cfg["r_grasp"]
        )
        return rewards

    def reset(self, env):
        # Release any existing grasps and reset reward/terminations
        robot = env.robots[self._robot_idn]
        for arm in getattr(robot, "arm_names", []):
            robot.release_grasp_immediately(arm=arm)

        super().reset(env)

    @classproperty
    def default_termination_config(cls):
        return {
            "max_collisions": 1,
            "max_steps": 500,
            "fall_height": 0.03,
        }

    @classproperty
    def default_reward_config(cls):
        return {"dist_coeff": 10.0, "r_grasp": 10.0, "ori_coeff": 5.0, "r_stand_upright": 1e-3}
