import math
import omnigibson.utils.transform_utils as T
import torch as th
from omnigibson.object_states import AttachedTo
from omnigibson.object_states import Pose
from omnigibson.object_states.inside import Inside
from omnigibson.object_states.next_to import NextTo
from omnigibson.object_states.on_top import OnTop
from omnigibson.object_states.open_state import Open
from omnigibson.object_states.open_state import _compute_joint_threshold
from omnigibson.object_states.robot_related_states import IsGrasping
from omnigibson.object_states.toggle import ToggledOn
from omnigibson.reward_functions.grasp_reward import GraspReward
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.tasks.grasp_task import GraspTask
from omnigibson.tasks.point_navigation_task import PointNavigationTask
from omnigibson.tasks.point_reaching_task import PointReachingTask
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.termination_condition_base import SuccessCondition
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.constants import JointType
from omnigibson.utils.python_utils import classproperty


def _get_named(env, name):
    return env.scene.object_registry("name", name)


def _front_target(obj, offset=0.6):
    pos, quat = obj.get_position_orientation()
    return pos + T.quat_apply(quat, th.tensor([offset, 0, 0], dtype=th.float32))


class OffsetOnceReward(BaseRewardFunction):
    def __init__(self, r_offset=0.0):
        self._r_offset = float(r_offset)
        self.potential_reward = 0.0
        super().__init__()

    def reset(self, task, env):
        self._paid = False

    def _step(self, task, env, action):
        return self._r_offset, {}


class NormalizedPotentialReward(BaseRewardFunction):
    def __init__(self, potential_fcn, r_potential=1.0):
        self._potential_fcn = potential_fcn
        self._r_potential = float(r_potential)
        self._phi0 = 1.0
        self._prev = 0.0
        super().__init__()

    def reset(self, task, env):
        phi = float(self._potential_fcn(env))
        self._phi0 = max(phi, 1e-6)
        self._prev = 1.0 - min(1.0, phi / self._phi0)

    def _step(self, task, env, action):
        phi = float(self._potential_fcn(env))
        prog = 1.0 - min(1.0, phi / self._phi0)
        r = (prog - self._prev) * self._r_potential
        self._prev = prog
        return r , {}


class GraspWithOffsetReward(BaseRewardFunction):
    def __init__(self, r_offset=0.0, transform="tanh",k=1.0, **grasp_kwargs):
        self._inner = GraspReward(**grasp_kwargs)
        self._r_offset = r_offset
        self._transform = transform
        self._k = k
        super().__init__()

    def reset(self, task, env):
        self._inner.reset(task, env)

    def _apply_transform(self, r: float) -> float:
        if self._transform == "sigmoid":
            return 1.0 / (1.0 + math.exp(-self._k * r))
        elif self._transform == "tanh":
            return 0.5 * (math.tanh(self._k * r) + 1.0)

    def _step(self, task, env, action):
        r, info = self._inner.step(task, env, action)
        r = self._apply_transform(float(r))
        return r + self._r_offset, info


class SuccessBonusReward(BaseRewardFunction):
    def __init__(self, success_condition, r_success=10.0):
        self._succ = success_condition
        self._r = float(r_success)
        super().__init__()
    def reset(self, task, env):
        pass
    def _step(self, task, env, action):
        return (self._r if self._succ.success else 0.0), {}

class MoveBaseToObjectTask(PointNavigationTask):
    def __init__(
            self,
            target_object_name: str,
            front_offset: float = 0.0,
            robot_idn: int = 0,
            goal_tolerance: float = 1.5,
            reward_type: str = "l2",
            max_steps: int | None = None,
            visualize_goal: bool = False,
            termination_config=None,
            reward_config=None,
            include_obs: bool = True,
            **kwargs,
    ):
        self._target_object_name = target_object_name
        self._front_offset = front_offset

        term_cfg = dict(termination_config or {})
        if max_steps is not None:
            term_cfg["max_steps"] = max_steps

        # Initialize the navigation task
        super().__init__(
            robot_idn=robot_idn,
            goal_tolerance=goal_tolerance,
            reward_type=reward_type,
            termination_config=term_cfg,
            reward_config=reward_config,
            include_obs=include_obs,
            visualize_goal=visualize_goal,
            **kwargs,
        )

    def reset(self, env):
        """
        Resolve the object and set the navigation goal before the parent reset so built-in rewards/terminations see it.
        """

        base_z = env.robots[self._robot_idn].states[Pose].get_value()[0][2]
        goal_xy = _front_target(_get_named(env, self._target_object_name))[:2]
        self._goal_pos = th.tensor([goal_xy[0], goal_xy[1], base_z], dtype=th.float32)
        self._randomize_goal_pos = False
        super().reset(env)

    def _create_reward_functions(self):
        rewards = super()._create_reward_functions()

        # Swap default potential for normalized progress shaping
        rp = float(self._reward_config.get("r_potential", 1.0))
        rewards["potential"] = NormalizedPotentialReward(
            potential_fcn=self.get_potential,
            r_potential=rp,
        )

        return rewards


class MoveEEToObjectTask(PointReachingTask):
    def __init__(
            self,
            target_object_name: str,
            front_offset: float = 0.0,
            robot_idn: int = 0,
            goal_tolerance: float = 0.04,
            max_steps: int | None = None,
            visualize_goal: bool = False,
            termination_config=None,
            reward_config=None,
            include_obs: bool = True,
            **kwargs,
    ):
        self._target_object_name = target_object_name
        self._front_offset = front_offset
        self.prev_reward = 0.0

        term_cfg = dict(termination_config or {})
        if max_steps is not None:
            term_cfg["max_steps"] = max_steps


        # Initialize the navigation task
        super().__init__(
            robot_idn=robot_idn,
            goal_tolerance=goal_tolerance,
            termination_config=term_cfg,
            reward_config=reward_config,
            include_obs=include_obs,
            visualize_goal=visualize_goal,
            **kwargs,
        )

    def reset(self, env):
        eef = th.as_tensor(env.robots[self._robot_idn].get_eef_position(), dtype=th.float32)

        # Set goal to the closest point on the object's AABB
        obj = _get_named(env, self._target_object_name)
        goal_pos = None
        if obj is not None:
            try:
                lo, hi = obj.aabb  # each th.tensor(3)
                lo = th.as_tensor(lo, dtype=th.float32)
                hi = th.as_tensor(hi, dtype=th.float32)
                closest = th.maximum(th.minimum(eef, hi), lo)  # clamp EEF onto the box
                goal_pos = closest
            except Exception:
                pass

        # Fallback: place goal “in front” of the object at the EEF’s current Z
        if goal_pos is None:
            goal_xy = _front_target(obj, offset=self._front_offset)[:2]
            goal_pos = th.tensor([goal_xy[0], goal_xy[1], eef[2]], dtype=th.float32)

        self._goal_pos = goal_pos
        self._randomize_goal_pos = False

        super().reset(env)

    def _create_termination_conditions(self):
        # Start from the parent setup
        terms = super()._create_termination_conditions()

        class _EEFsToObjectAABBXYLE(SuccessCondition):
            def __init__(self, robot_idn, target_name, tol):
                self._robot_idn = robot_idn
                self._target = target_name
                self._tol = float(tol)
                super().__init__()

            def _closest_xy_dist(self, p_xy, obj):
                lo, hi = obj.aabb
                lo_xy, hi_xy = lo[:2], hi[:2]
                # Closest point on AABB in XY
                cl_xy = th.maximum(th.minimum(p_xy, hi_xy), lo_xy)
                d = th.norm(p_xy - cl_xy)
                return float(d)

            def _step(self, task, env, action):
                r = env.robots[self._robot_idn]
                poses = []
                try:
                    poses.append(r.get_eef_position(arm="left"))
                except Exception:
                    pass
                try:
                    poses.append(r.get_eef_position(arm="right"))
                except Exception:
                    pass
                if not poses:
                    poses.append(r.get_eef_position())

                dmin = None
                obj = _get_named(env, self._target)
                if obj is not None:
                    for p in poses:
                        val = self._closest_xy_dist(p[:2], obj)
                        dmin = val if dmin is None else min(dmin, val)

                if dmin is None:
                    # Fallback to fixed goal
                    goal_xy = task.get_goal_pos()[:2]
                    for p in poses:
                        val = float(th.norm(p[:2] - goal_xy))
                        dmin = val if dmin is None else min(dmin, val)

                return (dmin is not None) and (dmin <= self._tol)

        terms["pointgoal"] = _EEFsToObjectAABBXYLE(
            self._robot_idn,
            self._target_object_name,
            self._goal_tolerance,
        )
        return terms

    def _get_l2_potential(self, env):
        """
        Potential for shaping: XY distance from the EEF to the current goal point.
        Using the goal point avoids the zero-at-reset issue seen with AABB clamped distance.
        """
        try:
            eef = env.robots[self._robot_idn].get_eef_position()
        except Exception:
            eef = env.robots[self._robot_idn].states[Pose].get_value()[0]
        goal = self.get_goal_pos()
        return T.l2_distance(eef[:2], goal[:2])


class _PredicateToggleTask(BaseTask):
    def __init__(
            self,
            target_object_name: str,
            desired_predicate: str,
            desired_value: bool,
            termination_config=None,
            reward_config=None,
            include_obs: bool = False,
    ):
        self._target_object_name = target_object_name
        self._pred = desired_predicate.lower()
        self._val = bool(desired_value)

        term_cfg = dict(termination_config or {})
        term_cfg.setdefault("max_steps", 4000)

        super().__init__(termination_config=term_cfg, reward_config=reward_config, include_obs=include_obs)

    def _create_termination_conditions(self):
        return {"timeout": Timeout(max_steps=self._termination_config["max_steps"])}

    def _create_reward_functions(self):
        return {}

    def _load(self, env):
        return

    def _get_obs(self, env):
        return {}, {}

    def _load_non_low_dim_observation_space(self):
        return {}

    @classproperty
    def default_termination_config(cls):
        return {"max_steps": 3000}

    @classproperty
    def default_reward_config(cls):
        return {}

    def reset(self, env):
        super().reset(env)

    def _get_state(self, obj):
        if self._pred == "on" and ToggledOn in obj.states:
            return obj.states[ToggledOn].get_value()
        if self._pred == "open" and Open in obj.states:
            return obj.states[Open].get_value()
        return None

    def step(self, env, action):
        info = {"done": {"success": False, "termination_conditions": dict()}}
        obj = _get_named(env, self._target_object_name)
        if obj is None:
            info["done"]["termination_conditions"] = {"object_not_found": {"done": True}}
            return 0.0, True, info

        cur = self._get_state(obj)

        if cur == self._val:
            info["done"]["success"] = True
            info["done"]["termination_conditions"] = {"predicate": {"done": True}}
            return 1.0, True, info

        # Not yet successful: allow global timeout
        base_done, base_info = super()._step_termination(
            env=env,
            action=action,
            info={"done": {"success": False, "termination_conditions": {}}},
        )
        tc = dict(base_info.get("done", {}).get("termination_conditions", {}))
        for k, v in list(tc.items()):
            if not isinstance(v, dict):
                tc[k] = {"done": bool(v)}
        if base_done and not any(d.get("done", False) for d in tc.values()):
            tc.setdefault("timeout", {"done": True})
        info["done"]["termination_conditions"] = tc
        done_out = any(d.get("done", False) for d in tc.values())

        reward = self._reward_config.get("r_offset", 0.0)
        return reward, done_out, info

    @classproperty
    def valid_scene_types(cls):
        return {TraversableScene}

    @classproperty
    def default_termination_config(cls):
        return {"max_steps": 2000}

    @classproperty
    def default_reward_config(cls):
        return {}

    @classproperty
    def default_reward_config(cls):
        base = {}
        base.update({
            "r_offset": 0.0,
            "use_normalized_potential": False,
        })
        return base


class OnTask(_PredicateToggleTask):
    def __init__(self, target_object_name: str, **kwargs):
        super().__init__(target_object_name=target_object_name, desired_predicate="on", desired_value=True, **kwargs)


class OpenTask(_PredicateToggleTask):
    def __init__(self, target_object_name: str, **kwargs):
        super().__init__(target_object_name=target_object_name, desired_predicate="open", desired_value=True, **kwargs)


class CloseTask(_PredicateToggleTask):
    def __init__(self, target_object_name: str, **kwargs):
        super().__init__(target_object_name=target_object_name, desired_predicate="open", desired_value=False, **kwargs)

class GraspGoal(SuccessCondition):
    """
    Success when the robot is grasping / attached to the target object.
    Checks both assisted grasp (IsGrasping) and physical attachment (AttachedTo).
    """

    def __init__(self, obj_name: str):
        self._obj_name = obj_name
        super().__init__()

    def _step(self, task, env, action):
        robot = env.robots[0]
        obj = env.scene.object_registry("name", self._obj_name)
        if obj is None:
            return False
        try:
            if IsGrasping in robot.states and robot.states[IsGrasping].get_value(obj):
                return True
        except Exception:
            pass
        try:
            if AttachedTo in obj.states and obj.states[AttachedTo].get_value(robot):
                return True
        except Exception:
            pass
        return False

class DeltaOfInnerReward(BaseRewardFunction):
    """
    Wraps an inner reward function and returns its per-step delta:
    r_delta_t = (r_raw_t - r_raw_{t-1}) * scale
    First step returns 0. Optionally clips the delta.
    """
    def __init__(self, inner, scale=1.0, clip_abs=0.0):
        self._inner = inner
        self._scale = float(scale)
        self._clip = float(clip_abs)
        self._prev = None
        super().__init__()

    def reset(self, task, env):
        self._prev = None
        self._inner.reset(task, env)

    def _step(self, task, env, action):
        r_raw, info = self._inner.step(task, env, action)
        r_raw = float(r_raw)
        if self._prev is None:
            r_delta = 0.0
        else:
            r_delta = (r_raw - self._prev) * self._scale
        if self._clip > 0.0:
            r_delta = max(-self._clip, min(self._clip, r_delta))
        self._prev = r_raw
        # Expose both raw and delta for debugging
        info = dict(info or {})
        info["grasp_raw"] = r_raw
        info["grasp_delta"] = r_delta
        return r_delta, info

class RobustGraspTask(GraspTask):

    def __init__(
            self,
            obj_name: str,
            termination_config=None,
            reward_config=None,
            include_obs: bool = False,
            precached_reset_pose_path=None,
            objects_config=None,
    ):
        self._obj_name = obj_name

        super().__init__(
            obj_name=obj_name,
            termination_config=termination_config,
            reward_config=reward_config,
            include_obs=include_obs,
            precached_reset_pose_path=precached_reset_pose_path,
            objects_config=objects_config,
        )

    def _create_termination_conditions(self):
        terminations = dict()
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        terminations["graspgoal"] = GraspGoal(obj_name=self._obj_name)
        return terminations

    def _create_reward_functions(self):
        allowed = (
            "dist_coeff",
            "grasp_reward",
            "collision_penalty",
            "eef_position_penalty_coef",
            "eef_orientation_penalty_coef",
            "regularization_coef",
        )

        cfg = {k: self._reward_config[k] for k in allowed if k in self._reward_config}


        inner = GraspReward(obj_name=self._obj_name, **cfg)

        # Wrap to return per-step delta, with optional scale / clip
        r_scale = float(self._reward_config.get("r_delta_scale", 1.0))
        r_clip = float(self._reward_config.get("r_delta_clip_abs", 0.0))
        grasp_delta = DeltaOfInnerReward(inner=inner, scale=r_scale, clip_abs=r_clip)

        rewards = {
            "grasp_delta": grasp_delta,
            "success_bonus": SuccessBonusReward(
                success_condition=self._termination_conditions["graspgoal"],
                r_success=float(self._reward_config.get("r_success", 10.0)),
            ),
        }
        return rewards

    @classproperty
    def default_reward_config(cls):
        base = dict(super(RobustGraspTask, cls).default_reward_config)
        base.update({
            "r_offset": 0.0,
            "use_normalized_potential": False,
        })
        return base


class _RelativeStatusTask(BaseTask):

    def __init__(
            self,
            target_object_name: str,
            source_object_name: str,
            desired_predicate: str,
            desired_value: bool,
            termination_config=None,
            reward_config=None,
            include_obs: bool = False,
    ):
        self._target_object_name = target_object_name
        self._source_object_name = source_object_name
        self._pred = desired_predicate.lower()
        self._val = bool(desired_value)

        term_cfg = dict(termination_config or {})
        term_cfg.setdefault("max_steps", 4000)
        super().__init__(termination_config=term_cfg, reward_config=reward_config or {}, include_obs=include_obs)

    def _create_termination_conditions(self):
        return {"timeout": Timeout(max_steps=self._termination_config["max_steps"])}

    def _create_reward_functions(self):
        return {}

    def _load(self, env):
        return

    def _get_obs(self, env):
        return {}, {}

    def _load_non_low_dim_observation_space(self):
        return {}

    @classproperty
    def default_termination_config(cls):
        return {"max_steps": 3000}

    @classproperty
    def default_reward_config(cls):
        return {}

    def reset(self, env):
        super().reset(env)

    def _get_state(self, a, b):
        if self._pred == "next_to" and NextTo in a.states:
            return a.states[NextTo].get_value(b)
        if self._pred == "inside" and Inside in a.states:
            return a.states[Inside].get_value(b)
        if self._pred == "on_top" and OnTop in a.states:
            return a.states[OnTop].get_value(b)
        return None

    def step(self, env, action):
        info = {"done": {"success": False, "termination_conditions": {}}}

        a = _get_named(env, self._target_object_name)
        b = _get_named(env, self._source_object_name)
        reward_offset = self._reward_config.get("r_offset", 0.0)
        if a is None or b is None:
            missing = []
            if a is None:
                missing.append(self._target_object_name)
            if b is None:
                missing.append(self._source_object_name)
            info["done"]["termination_conditions"] = dict(object_not_found={"done": True, "which": missing})
            return 0.0, True, info

        cur = self._get_state(a, b)
        if cur == self._val:
            info["done"]["success"] = True
            info["done"]["termination_conditions"] = dict(predicate={"done": True})
            return reward_offset + 1.0, True, info

        # Not yet successful, allow global timeout
        base_done, base_info = super()._step_termination(
            env=env,
            action=action,
            info={"done": {"success": False, "termination_conditions": {}}},
        )
        tc = dict(base_info.get("done", {}).get("termination_conditions", {}))
        for k, v in list(tc.items()):
            if not isinstance(v, dict):
                tc[k] = {"done": bool(v)}
        if base_done and not any(d.get("done", False) for d in tc.values()):
            tc.setdefault("timeout", {"done": True})
        info["done"]["termination_conditions"] = tc
        done_out = any(d.get("done", False) for d in tc.values())
        return reward_offset, done_out, info

    @classproperty
    def valid_scene_types(cls):
        return {TraversableScene}

    # @classproperty
    # def default_reward_config(cls):
    #     base = {}
    #     base.update({
    #         "r_offset": 1.0,
    #         "use_normalized_potential": False,
    #     })
    #     return base


class NextToTask(_RelativeStatusTask):
    def __init__(self, target_object_name: str, source_object_name: str, desired_value: bool = True, **kwargs):
        super().__init__(
            target_object_name=target_object_name,
            source_object_name=source_object_name,
            desired_predicate="next_to",
            desired_value=desired_value,
            **kwargs,
        )


class InsideTask(_RelativeStatusTask):
    def __init__(self, target_object_name: str, source_object_name: str, desired_value: bool = True, **kwargs):
        super().__init__(
            target_object_name=target_object_name,
            source_object_name=source_object_name,
            desired_predicate="inside",
            desired_value=desired_value,
            **kwargs,
        )


class OnTopTask(_RelativeStatusTask):
    def __init__(self, target_object_name: str, source_object_name: str, desired_value: bool = True, **kwargs):
        super().__init__(
            target_object_name=target_object_name,
            source_object_name=source_object_name,
            desired_predicate="on_top",
            desired_value=desired_value,
            **kwargs,
        )


def _open_fraction(obj):
    """Return max open fraction across all openable joints on this object (0..1)."""
    md = getattr(obj, "metadata", None)
    joints, dirs = [], []
    if md and "openable_joint_ids" in md and len(md["openable_joint_ids"]) > 0:
        for tup in list(md["openable_joint_ids"].items()):
            name = tup[1]
            d = tup[2] if len(tup) > 2 else 1
            if name in obj.joints:
                joints.append(obj.joints[name])
                dirs.append(1 if d >= 0 else -1)
    else:
        joints = list(obj.joints.values())
        dirs = [1] * len(joints)
    fracs = []
    for j, d in zip(joints, dirs):
        pos = float(j.get_state()[0])
        _, open_end, closed_end = _compute_joint_threshold(j, d)
        total = abs(open_end - closed_end) or 1.0
        fracs.append(max(0.0, min(1.0, abs(pos - closed_end) / total)))
    return max(fracs) if fracs else 0.0


def _is_open_ge_angle(obj, min_deg: float = 90.0, min_frac: float = 0.8) -> bool:
    """Return True if any openable joint is opened at least min_deg (revolute) or min_frac (generic)."""
    angle_req = math.radians(min_deg)
    md = getattr(obj, "metadata", None)
    joints, dirs = [], []
    if md and "openable_joint_ids" in md and len(md["openable_joint_ids"]) > 0:
        for tup in list(md["openable_joint_ids"].items()):
            name = tup[1]
            d = tup[2] if len(tup) > 2 else 1
            if name in obj.joints:
                joints.append(obj.joints[name])
                dirs.append(1 if d >= 0 else -1)
    else:
        joints = list(obj.joints.values())
        dirs = [1] * len(joints)

    for j, d in zip(joints, dirs):
        pos = float(j.get_state()[0])
        _, open_end, closed_end = _compute_joint_threshold(j, d)
        opened = abs(pos - closed_end)
        total = abs(open_end - closed_end)
        if j.joint_type == JointType.JOINT_REVOLUTE:
            if total >= angle_req and opened >= angle_req:
                return True
            if total > 1e-6 and (opened / total) >= min_frac:
                return True
        else:
            if total > 1e-6 and (opened / total) >= min_frac:
                return True
    return False


def _iter_openable_joints_and_dirs(obj):
    """Return (joints, directions) for openable joints on obj.

    If metadata lists specific openable joints, include those present in obj.joints.
    If none are found or metadata absent, fall back to all joints with default direction +1.
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


def _joint_open_metrics(joint, direction):
    # Uses your existing threshold helper
    pos = float(joint.get_state()[0])
    _, open_end, closed_end = _compute_joint_threshold(joint, direction)
    opened = abs(pos - closed_end)  # how far from fully closed
    total = abs(open_end - closed_end)  # full travel
    frac = (opened / total) if total > 1e-6 else 0.0
    return opened, total, frac


def _is_closed_le_angle(
        obj,
        max_deg: float = 5.0,
        max_frac: float = 0.1,
        require_all: bool = True,
) -> bool:
    """
    Return True if the door is sufficiently closed:
    - Revolute: opened <= max_deg (or opened/total <= max_frac)
    - Prismatic / other: opened/total <= max_frac
    require_all=True means all openable joints must satisfy the closed condition.
    """
    angle_tol = math.radians(max_deg)
    joints, dirs = _iter_openable_joints_and_dirs(obj)
    results = []
    for j, d in zip(joints, dirs):
        opened, total, frac = _joint_open_metrics(j, d)
        if j.joint_type == JointType.JOINT_REVOLUTE:
            closed_ok = (opened <= angle_tol) or (total > 1e-6 and frac <= max_frac)
        else:
            closed_ok = (total > 1e-6 and frac <= max_frac)
        results.append(closed_ok)

    return all(results) if require_all else any(results)


def _open_angles(obj):
    """
    Compute detailed opening info for all openable joints on @obj.

    Returns:
        dict with keys:
            - max_angle_deg (float): maximum opened angle in degrees among revolute joints (0.0 if none)
            - per_joint (list[dict]): one entry per considered joint with fields
                {name, type, opened, total, fraction, angle_deg}
    """
    md = getattr(obj, "metadata", None)
    joints, dirs, names = [], [], []
    if md and "openable_joint_ids" in md and len(md["openable_joint_ids"]) > 0:
        for tup in list(md["openable_joint_ids"].items()):
            name = tup[1]
            d = tup[2] if len(tup) > 2 else 1
            if name in obj.joints:
                joints.append(obj.joints[name])
                dirs.append(1 if d >= 0 else -1)
                names.append(name)
    else:
        for k, j in obj.joints.items():
            joints.append(j)
            dirs.append(1)
            names.append(k)

    per_joint = []
    max_angle_deg = 0.0
    for j, d, nm in zip(joints, dirs, names):
        pos = float(j.get_state()[0])
        _, open_end, closed_end = _compute_joint_threshold(j, d)
        opened = abs(pos - closed_end)
        total = abs(open_end - closed_end)
        frac = (opened / total) if total > 1e-6 else 0.0
        angle_deg = math.degrees(opened) if j.joint_type == JointType.JOINT_REVOLUTE else 0.0
        if j.joint_type == JointType.JOINT_REVOLUTE:
            max_angle_deg = max(max_angle_deg, angle_deg)
        per_joint.append({
            "name": nm,
            "type": j.joint_type,
            "opened": opened,
            "total": total,
            "fraction": frac,
            "angle_deg": angle_deg,
        })

    return {"max_angle_deg": max_angle_deg, "per_joint": per_joint}


class SufficientlyOpenTask(BaseTask):
    """
    Succeeds when the target object's door / openable joint satisfies an openness condition.
    For status="open": any openable joint meeting the angle / fraction threshold suffices.
    For status="closed": all openable joints must be within the closed tolerance.

    Args:
        target_object_name (str): name of the object to evaluate (e.g., fridge)
        allowed_deg (float): threshold in degrees. For status="open": minimum opened angle (revolute).
                             For status="closed": maximum opened angle tolerance (revolute).
        allowed_frac (float): fraction threshold in [0,1]. For status="open": minimum opened fraction.
                              For status="closed": maximum opened fraction.
    """

    def __init__(
            self,
            target_object_name: str,
            allowed_deg: float = 90.0,
            allowed_frac: float = 0.8,
            status: str = "open",
            termination_config=None,
            reward_config=None,
            include_obs: bool = False,
    ):
        self._target = target_object_name
        self._allowed_deg = float(allowed_deg)
        self._allowed_frac = float(allowed_frac)
        self._status = status
        self._prev_progress = None
        term_cfg = dict(termination_config or {})
        term_cfg.setdefault("max_steps", 4000)
        super().__init__(termination_config=term_cfg, reward_config=reward_config or {}, include_obs=include_obs)

    def _create_termination_conditions(self):
        return {"timeout": Timeout(max_steps=self._termination_config["max_steps"])}

    def _create_reward_functions(self):
        return {}

    # Minimal BaseTask API
    def _load(self, env):
        return

    def _get_obs(self, env):
        return {}, {}

    def _load_non_low_dim_observation_space(self):
        return {}

    @classproperty
    def default_termination_config(cls):
        return {"max_steps": 3000}

    @classproperty
    def default_reward_config(cls):
        return {}

    def reset(self, env):
        super().reset(env)
        self._prev_progress = None

    @staticmethod
    def _open_progress(obj, allowed_deg: float, allowed_frac: float) -> float:
        ang = _open_angles(obj)["max_angle_deg"]
        p_ang = 0.0 if allowed_deg <= 1e-6 else max(0.0, min(1.0, ang / allowed_deg))
        p_frac = max(0.0, min(1.0, _open_fraction(obj)))
        return max(p_ang, p_frac)  # optimistic progress toward open

    @staticmethod
    def _closed_progress(obj, allowed_deg: float, allowed_frac: float) -> float:
        p_open = SufficientlyOpenTask._open_progress(obj, allowed_deg, allowed_frac)
        return 1.0 - p_open

    def _progress(self, obj) -> float:
        st = (self._status or "open").lower()
        if st == "open":
            return SufficientlyOpenTask._open_progress(obj, self._allowed_deg, self._allowed_frac)
        elif st == "closed":
            return SufficientlyOpenTask._closed_progress(obj, self._allowed_deg, self._allowed_frac)
        return 0.0

    def step(self, env, action):
        info = {"done": {"success": False, "termination_conditions": {}}}
        obj = env.scene.object_registry("name", self._target)
        if obj is None:
            info["done"]["termination_conditions"] = dict(object_not_found={"done": True})
            return 0.0, True, info

        status = (self._status or "open").lower()
        if status == "open":
            ok = _is_open_ge_angle(obj, min_deg=self._allowed_deg, min_frac=self._allowed_frac)
        elif status == "closed":
            ok = _is_closed_le_angle(obj, max_deg=self._allowed_deg, max_frac=self._allowed_frac, require_all=True)
        else:
            ok = False

        # Signed delta dense reward on bounded progress
        prog = float(self._progress(obj))
        if self._prev_progress is None:
            dense = 0.0
        else:
            dense = (prog - self._prev_progress) * float(self._reward_config.get("r_scale", 1.0))
            clip = float(self._reward_config.get("r_clip_abs", 0.0))
            if clip > 0.0:
                dense = max(-clip, min(clip, dense))
        self._prev_progress = prog

        ang_info = _open_angles(obj)
        info["progress"] = prog
        info["open_fraction"] = _open_fraction(obj)
        info["open_angle_deg"] = ang_info["max_angle_deg"]
        info["open_debug"] = ang_info["per_joint"]
        if ok:
            info["done"]["success"] = True
            info["done"]["termination_conditions"] = dict(predicate={"done": True})
            return float(self._reward_config.get("r_success", 1.0)), True, info

        # Not yet successful: allow global timeout
        base_done, base_info = super()._step_termination(
            env=env,
            action=action,
            info={"done": {"success": False, "termination_conditions": {}}},
        )
        tc = dict(base_info.get("done", {}).get("termination_conditions", {}))
        for k, v in list(tc.items()):
            if not isinstance(v, dict):
                tc[k] = {"done": bool(v)}
        if base_done and not any(d.get("done", False) for d in tc.values()):
            tc.setdefault("timeout", {"done": True})
        info["done"]["termination_conditions"] = tc
        done_out = any(d.get("done", False) for d in tc.values())
        return float(dense), done_out, info

    @classproperty
    def valid_scene_types(cls):
        return {TraversableScene}

    @classproperty
    def default_reward_config(cls):
        return {
            "r_scale": 1.0,
            "r_success": 10.0,
            "r_clip_abs": 0.0,
            "r_offset": 0.0
        }


class SufficientlyClosedTask(SufficientlyOpenTask):
    def __init__(
        self,
        target_object_name: str,
        allowed_deg: float = 5.0,
        allowed_frac: float = 0.1,
        termination_config=None,
        reward_config=None,
        include_obs: bool = False,
        ):
        super().__init__(
        target_object_name=target_object_name,
        allowed_deg=allowed_deg,
        allowed_frac=allowed_frac,
        status="closed",
        termination_config=termination_config,
        reward_config=reward_config,
        include_obs=include_obs,
        )
        self._prev_progress = None # track previous closed progress

    def reset(self, env):
        super().reset(env)
        self._prev_progress = None

    def step(self, env, action):
        info = {"done": {"success": False, "termination_conditions": {}}}
        obj = env.scene.object_registry("name", self._target)
        if obj is None:
            info["done"]["termination_conditions"] = {"object_not_found": {"done": True}}
            return 0.0, True, info

        # Success check (closed enough for all joints)
        ok = _is_closed_le_angle(
            obj,
            max_deg=self._allowed_deg,
            max_frac=self._allowed_frac,
            require_all=True,
        )

        # Closed progress in [0,1] and signed delta
        open_frac = max(0.0, min(1.0, float(_open_fraction(obj))))
        closed_prog = 1.0 - open_frac
        if self._prev_progress is None:
            dense = 0.0
        else:
            dense = (closed_prog - self._prev_progress) * float(self._reward_config.get("r_scale", 1.0))
            clip = float(self._reward_config.get("r_clip_abs", 0.0))
            if clip > 0.0:
                dense = max(-clip, min(clip, dense))
        self._prev_progress = closed_prog

        # Debug
        ang_info = _open_angles(obj)
        info["closed_progress"] = closed_prog
        info["open_fraction"] = open_frac
        info["open_angle_deg"] = ang_info["max_angle_deg"]
        info["open_debug"] = ang_info["per_joint"]

        if ok:
            info["done"]["success"] = True
            info["done"]["termination_conditions"] = {"predicate": {"done": True}}
            return float(self._reward_config.get("r_success", 1.0)), True, info

        # Timeout
        base_done, base_info = super()._step_termination(
            env=env,
            action=action,
            info={"done": {"success": False, "termination_conditions": {}}},
        )
        tc = dict(base_info.get("done", {}).get("termination_conditions", {}))
        for k, v in list(tc.items()):
            if not isinstance(v, dict):
                tc[k] = {"done": bool(v)}
        if base_done and not any(d.get("done", False) for d in tc.values()):
            tc.setdefault("timeout", {"done": True})
        info["done"]["termination_conditions"] = tc
        done_out = any(d.get("done", False) for d in tc.values())

        return dense, done_out, info


class SufficientlyClosedTask_(SufficientlyOpenTask):
    """Convenience wrapper: require doors / joints to be sufficiently closed.

    Equivalent to SufficientlyOpenTask with status="closed" and the same thresholds interpreted as maxima.
    """

    def __init__(
            self,
            target_object_name: str,
            allowed_deg: float = 5.0,
            allowed_frac: float = 0.1,
            termination_config=None,
            reward_config=None,
            include_obs: bool = False,
    ):
        self._prev_closed_progress = 0
        super().__init__(
            target_object_name=target_object_name,
            allowed_deg=allowed_deg,
            allowed_frac=allowed_frac,
            status="closed",
            termination_config=termination_config,
            reward_config=reward_config,
            include_obs=include_obs,
        )

    def reset(self, env):
        super().reset(env)
        self._prev_closed_progress = None

    def step(self, env, action):
        info = {"done": {"success": False, "termination_conditions": {}}}

        obj = env.scene.object_registry("name", self._target)
        if obj is None:
            info["done"]["termination_conditions"] = dict(object_not_found={"done": True})
            return 0.0, True, info

        # Success check: closed enough (all joints)
        ok = _is_closed_le_angle(
            obj,
            max_deg=self._allowed_deg,
            max_frac=self._allowed_frac,
            require_all=True,
        )

        # Shaping metrics
        open_frac = _open_fraction(obj)  # 0..1, 1 = fully open
        closed_prog = float(1.0 - max(0.0, min(1.0, open_frac)))  # 0..1, 1 = fully closed

        if self._prev_closed_progress is None:
            dense = 0.0
        else:
            dense = max(0.0, closed_prog - self._prev_closed_progress)

        self._prev_closed_progress = closed_prog

        ang_info = _open_angles(obj)
        info["closed_progress"] = closed_prog
        info["open_fraction"] = open_frac
        info["open_angle_deg"] = ang_info["max_angle_deg"]
        info["open_debug"] = ang_info["per_joint"]

        if ok:
            info["done"]["success"] = True
            info["done"]["termination_conditions"] = dict(predicate={"done": True})
            return 10.0, True, info  # final success reward

        # Allow global timeout
        base_done, base_info = super()._step_termination(
            env=env,
            action=action,
            info={"done": {"success": False, "termination_conditions": {}}},
        )
        tc = dict(base_info.get("done", {}).get("termination_conditions", {}))
        for k, v in list(tc.items()):
            if not isinstance(v, dict):
                tc[k] = {"done": bool(v)}
        if base_done and not any(d.get("done", False) for d in tc.values()):
            tc.setdefault("timeout", {"done": True})
        info["done"]["termination_conditions"] = tc
        done_out = any(d.get("done", False) for d in tc.values())

        r_offset = float(self._reward_config.get("r_offset", 0.0))
        dense += r_offset

        return dense, done_out, info


def _center_xy(obj):
    lo, hi = obj.aabb
    return ((lo + hi) / 2.0)[:2]


class OnTopStableTask(BaseTask):
    def __init__(
            self,
            target_object_name: str,
            source_object_name: str,
            xy_tol: float = 0.23,  # center alignment tolerance
            require_release: bool = True,  # must not be grasped by robot
            termination_config=None,
            reward_config=None,
            include_obs: bool = False,
    ):
        self._tgt = target_object_name
        self._src = source_object_name
        self._xy_tol = float(xy_tol)
        self._require_release = bool(require_release)
        term_cfg = dict(termination_config or {})
        term_cfg.setdefault("max_steps", 4000)
        super().__init__(termination_config=term_cfg, reward_config=reward_config or {}, include_obs=include_obs)

    def _create_termination_conditions(self):
        return {"timeout": Timeout(max_steps=self._termination_config["max_steps"])}

    def _create_reward_functions(self):
        return {}

    def _load(self, env):
        return

    def _get_obs(self, env):
        return {}, {}

    def _load_non_low_dim_observation_space(self):
        return {}

    @classproperty
    def default_termination_config(cls):
        return {"max_steps": 3000}

    @classproperty
    def default_reward_config(cls):
        base = {}
        base.update({
            "r_offset": 1.0,
            "use_normalized_potential": False,
        })
        return base

    def reset(self, env):
        super().reset(env)

    def step(self, env, action):
        info = {"done": {"success": False, "termination_conditions": {}}}
        tgt = _get_named(env, self._tgt)
        src = _get_named(env, self._src)
        if tgt is None or src is None:
            info["done"]["termination_conditions"] = dict(object_not_found={"done": True})
            return 0.0, True, info

        on_top_now = bool(tgt.states.get(OnTop, None) and tgt.states[OnTop].get_value(src))

        # Additional placement checks
        try:
            xy_err = float(th.norm(_center_xy(tgt) - _center_xy(src)))
        except Exception:
            xy_err = 1e9
        aligned = (xy_err <= self._xy_tol)

        released = True
        if self._require_release and IsGrasping in tgt.states:
            released = not any(tgt.states[IsGrasping].get_value(arm_name=a) for a in env.robots[0].arm_names)
            # released = not tgt.states[IsGrasping].get_value()

        # Dense shaping use centered closeness (higher is better)
        reward_offset = self._reward_config.get("r_offset", 0.0)
        dense = max(0.0, 1.0 - xy_err / max(self._xy_tol, 1e-6)) + reward_offset

        if on_top_now and aligned and released:
            info["done"]["success"] = True
            info["done"]["termination_conditions"] = dict(predicate={"done": True})
            return 1.0, True, info

        # Allow timeout
        base_done, base_info = super()._step_termination(env=env, action=action, info={
            "done": {"success": False, "termination_conditions": {}}})
        tc = dict(base_info.get("done", {}).get("termination_conditions", {}))
        for k, v in list(tc.items()):
            if not isinstance(v, dict):
                tc[k] = {"done": bool(v)}
        if base_done and not any(d.get("done", False) for d in tc.values()):
            tc.setdefault("timeout", {"done": True})
        info["done"]["termination_conditions"] = tc
        done_out = any(d.get("done", False) for d in tc.values())
        return dense, done_out, info
