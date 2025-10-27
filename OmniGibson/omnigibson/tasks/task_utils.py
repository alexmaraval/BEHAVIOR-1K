import omnigibson.utils.transform_utils as T
import torch as th
from omnigibson.termination_conditions.max_collision import MaxCollision
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.reward_functions.collision_reward import CollisionReward


def _get_named(env, name):
    return env.scene.object_registry("name", name)


def _front_target(obj, offset=0.6):
    pos, quat = obj.get_position_orientation()
    return pos + T.quat_apply(quat, th.tensor([offset, 0, 0], dtype=th.float32))


def _center_xy(obj):
    lo, hi = obj.aabb
    return ((lo + hi) / 2.0)[:2]


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
        return self._n_collisions > self._max_collisions


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
