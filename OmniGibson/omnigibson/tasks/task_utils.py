import omnigibson.utils.transform_utils as T
import torch as th
from math import radians
from omnigibson.termination_conditions.max_collision import MaxCollision
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.reward_functions.collision_reward import CollisionReward
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.object_states import Pose
from omnigibson.controllers import IsGraspingState
from omnigibson.utils.constants import JointType
from omnigibson.object_states.open_state import _compute_joint_threshold
# from omnigibson.tasks.custom_open_close_task import _iter_openable_joints_and_dirs

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
    heading_error_nor = heading_error / pi

    return heading_error_nor

def get_collided_objects(env, in_contact_objects):
    collided = set()
    for rp in in_contact_objects:
        # Parent prim path is the object prim path
        obj_prim_path = "/".join(rp.prim_path.split("/")[:-1])
        obj = env.scene.object_registry("prim_path", obj_prim_path)
        if obj is not None:
            collided.add(obj.name)
    return collided

def get_free_robot_arms(robot, object, consider_free_arms_only ):
    free_arms = []
    for arm in getattr(robot, "arm_names", [robot.default_arm]):
        state = robot.is_grasping(arm=arm, candidate_obj=object)
        if consider_free_arms_only and state != IsGraspingState.TRUE:
            free_arms.append(arm)

    return free_arms

def _iter_openable_joints_and_dirs(obj):
    """
    Return the list of openable joints and their corresponding open directions.
    If the object's metadata defines `openable_joint_ids`, only those joints are considered.
    Otherwise, all joints are assumed openable with a default positive direction (+1).
    Args:
        obj: The simulated object with `joints` and optional `metadata`.

    Returns:
        A tuple `(joints, dirs)` where:
            - `joints` is a list of joint instances.
            - `dirs` is a list of direction multipliers (+1 or -1) for each joint.

    """
    md = getattr(obj, "metadata", None)
    joints, dirs = [], []
    if md and "openable_joint_ids" in md and len(md["openable_joint_ids"]) > 0:
        for tup in list(md["openable_joint_ids"].items()):
            name = tup[1]
            d = tup[2] if len(tup) > 2 else 1
            if name in obj.joints:
                joints.append(obj.joints[name])
                dirs.append(1 if d >= 0 else -1)

    # Fallback if none resolved
    if not joints:
        joints = list(obj.joints.values())
        dirs = [1] * len(joints)
    return joints, dirs

def set_door_angle_deg(env, obj_name: str, deg: float = 80.0) -> None:
    obj = env.scene.object_registry("name", obj_name)
    assert obj is not None, f"Object {obj_name} not found"

    joints, dirs = _iter_openable_joints_and_dirs(obj)
    for joint, direction in zip(joints, dirs):
        if joint.joint_type == JointType.JOINT_REVOLUTE:
            # breakpoint()
            _, open_end, closed_end = _compute_joint_threshold(joint, direction)
            target = closed_end + direction * radians(deg)
            lo, hi = (min(open_end, closed_end), max(open_end, closed_end))
            target = max(lo, min(hi, target))  # clamp to limits
            joint.set_pos(target)
            break


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
        in_contact_objects = robot.states[ContactBodies].get_value(ignore_objs=ignore_objs)
        in_contact = len(in_contact_objects) > 0
        self._n_collisions += int(in_contact)
        collided_names = get_collided_objects(env, in_contact_objects)
        if collided_names:
            print(f"Robot collided with {collided_names}")
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

class SuccessBonusReward(BaseRewardFunction):
    def __init__(self, success_condition, r_success=10.0):
        self._success_condition = success_condition
        self._r = float(r_success)
        super().__init__()

    def reset(self, task, env):
        pass

    def _step(self, task, env, action):
        return (self._r if self._success_condition.success else 0.0), {}
