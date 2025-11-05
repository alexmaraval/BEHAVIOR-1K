import omnigibson.utils.transform_utils as T
import torch as th
from omnigibson.termination_conditions.max_collision import MaxCollision
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.reward_functions.collision_reward import CollisionReward
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.object_states import Pose

def _get_named(env, name):
    return env.scene.object_registry("name", name)


def _front_target(obj, offset=0.6):
    pos, quat = obj.get_position_orientation()
    return pos + T.quat_apply(quat, th.tensor([offset, 0, 0], dtype=th.float32))


def _center_xy(obj):
    lo, hi = obj.aabb
    return ((lo + hi) / 2.0)[:2]

def get_orientation_error(robot, goal_pos):
    pos, quat = robot.states[Pose].get_value()
    yaw = T.quat2euler(quat)[2]
    d = goal_pos[:2] - pos[:2]
    desired_yaw = th.atan2(d[1], d[0])
    pi = th.tensor(3.141592653589793, dtype=th.float32, device=desired_yaw.device)
    two_pi = 2.0 * pi
    diff = desired_yaw - yaw
    diff_wrapped = (diff + pi) % two_pi - pi
    heading_error = th.abs(diff_wrapped)

    return heading_error

class _MaxCollisionFiltered(MaxCollision):
    def __init__(self, task_ref, **kwargs):
        self._task_ref = task_ref
        super().__init__(**kwargs)

    def _step(self, task, env, action):
        robot = env.robots[self._robot_idn]
        floors = list(env.scene.object_registry("category", "floors", []))
        extra_ignores = self._task_ref.skip_collision_objs
        ignore_objs = floors if self._ignore_self_collisions is None else floors + [robot]
        ignore_objs = tuple(list(ignore_objs) + extra_ignores)
        in_contact = len(robot.states[ContactBodies].get_value(ignore_objs=ignore_objs)) > 0
        self._n_collisions += int(in_contact)
        return self._n_collisions >= self._max_collisions


class _CollisionRewardFiltered(CollisionReward):
    def __init__(self, task_ref, **kwargs):
        self._task_ref = task_ref
        super().__init__(**kwargs)

    def _step(self, task, env, action):
        robot = env.robots[self._robot_idn]
        floors = list(env.scene.object_registry("category", "floors", []))
        extra_ignores = list(self._task_ref.skip_collision_objs)
        ignore_objs = floors if self._ignore_self_collisions is None else floors + [robot]
        ignore_objs = tuple(list(ignore_objs) + extra_ignores)
        in_contact = len(robot.states[ContactBodies].get_value(ignore_objs=ignore_objs)) > 0
        reward = float(in_contact) * -self._r_collision
        return reward, {}


class OrientationAlignReward(BaseRewardFunction):
    """Penalize absolute yaw error between robot heading and goal direction: reward = -coef * |delta_yaw|."""

    def __init__(self, robot_idn: int, coef: float = 0.5):
        super().__init__()
        self._robot_idn = robot_idn
        self._coef = coef

    def _step(self, task, env, action):
        # Robot pose
        pos, quat = env.robots[self._robot_idn].states[Pose].get_value()
        # Current yaw from quaternion
        rpy = T.quat2euler(quat)
        yaw = rpy[2]

        # Direction to goal
        goal = task.get_goal_pos()
        d = goal[:2] - pos[:2]
        desired_yaw = th.atan2(d[1], d[0])

        # Smallest-angle difference wrap to [-pi, pi]
        pi = th.tensor(3.141592653589793, dtype=th.float32, device=desired_yaw.device)
        two_pi = 2.0 * pi
        diff = desired_yaw - yaw
        diff_wrapped = (diff + pi) % two_pi - pi

        penalty = th.abs(diff_wrapped)
        rew = -self._coef * penalty
        return float(rew.item()), {"heading_error": float(penalty.item())}