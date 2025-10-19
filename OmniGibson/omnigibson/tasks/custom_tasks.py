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
        # Store target and offset
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


class MoveEEToObjectTask_(PointReachingTask):
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
        # Store target and offset
        self._target_object_name = target_object_name
        self._front_offset = front_offset

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
        eef_z = env.robots[self._robot_idn].get_eef_position()[2]
        goal_xy = _front_target(_get_named(env, self._target_object_name))[:2]
        self._goal_pos = th.tensor([goal_xy[0], goal_xy[1], eef_z], dtype=th.float32)
        self._randomize_goal_pos = False

        super().reset(env)


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
        # No shaping; success itself is the reward
        return {}

    # Satisfy abstract BaseTask API with minimal implementations
    def _load(self, env):
        # No scene construction required for predicate check
        return

    def _get_obs(self, env):
        # No low-dim observations needed for this task
        return {}, {}

    def _load_non_low_dim_observation_space(self):
        # No non-low-dim observations
        return {}

    # Provide defaults so BaseTask can validate configs
    @classproperty
    def default_termination_config(cls):
        return {"max_steps": 3000}

    @classproperty
    def default_reward_config(cls):
        return {}

    def reset(self, env):
        # Nothing special beyond BaseTask
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

        # Read current predicate
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
        return 0.0, done_out, info

    def _load_non_low_dim_observation_space(self):
        return {}

    @classproperty
    def valid_scene_types(cls):
        return {TraversableScene}

    @classproperty
    def default_termination_config(cls):
        return {"max_steps": 2000}

    @classproperty
    def default_reward_config(cls):
        return {}


class OnTask(_PredicateToggleTask):
    def __init__(self, target_object_name: str, **kwargs):
        super().__init__(target_object_name=target_object_name, desired_predicate="on", desired_value=True, **kwargs)


class OpenTask(_PredicateToggleTask):
    def __init__(self, target_object_name: str, **kwargs):
        super().__init__(target_object_name=target_object_name, desired_predicate="open", desired_value=True, **kwargs)


class CloseTask(_PredicateToggleTask):
    def __init__(self, target_object_name: str, **kwargs):
        super().__init__(target_object_name=target_object_name, desired_predicate="open", desired_value=False, **kwargs)


class GraspAndRetreatTask(BaseTask):
    """
    Two-phase low-level wrapper:
      1) Move EEF in front of target object (uses MoveEEToObjectTask)
      2) Verify grasp predicate (simple proximity-based check) and then retreat EEF to a home pose

    Success = predicate satisfied and retreat goal reached.
    """

    def __init__(
        self,
        target_object_name: str="tray_208",
        approach_offset: float = 0.12,
        ee_tol: float = 0.08,
        robot_idn: int = 0,
        retreat_height: float = 0.15,
        verify_horizon: int = 50,
        max_steps: int = 4000,
        visualize_goal: bool = False,
        termination_config=None,
        reward_config=None,
        include_obs: bool = False,
    ):
        self._target = target_object_name
        self._offset = float(approach_offset)
        self._ee_tol = float(ee_tol)
        self._retreat_h = float(retreat_height)
        self._verify_horizon = int(verify_horizon)

        term_cfg = dict(termination_config or {})
        term_cfg.setdefault("max_steps", int(max_steps))
        super().__init__(termination_config=term_cfg, reward_config={}, include_obs=include_obs)

        # Subtasks
        # self._approach = PointReachingTask(
        #     target_object_name=self._target,
        #     front_offset=self._offset,
        #     goal_tolerance=self._ee_tol,
        #     visualize_goal=visualize_goal,
        #     termination_config={"max_steps": int(max_steps)},
        #     include_obs=False,
        # )
        self._approach = PointReachingTask(
            robot_idn=robot_idn,
            goal_tolerance=ee_tol,
            termination_config=term_cfg,
            reward_config=reward_config,
            include_obs=include_obs,
            visualize_goal=visualize_goal,
        )


        # self._retreat = PointReachingTask(
        #     goal_pos=[0.0, 0.0, 0.0],  # set at reset
        #     goal_tolerance=self._ee_tol,
        #     visualize_goal=visualize_goal,
        #     termination_config={"max_steps": int(max_steps)},
        #     include_obs=False,
        # )

        self._phase = 0  # 0: approach, 1: verify+retreat
        self._home_eef = None
        self._verify_steps = 0

    def _create_termination_conditions(self):
        return {"timeout": Timeout(max_steps=self._termination_config["max_steps"])}

    def _create_reward_functions(self):
        return {}

    # Minimal API
    def _load(self, env):
        return

    def _get_obs(self, env):
        return {}, {}

    def _load_non_low_dim_observation_space(self):
        return {}

    def reset(self, env):
        # Record home EEF and set retreat goal above it
        self._home_eef = env.robots[0].get_eef_position().clone()
        retreat_goal = self._home_eef + th.tensor([0.0, 0.0, self._retreat_h], dtype=th.float32)
        self._retreat._goal_pos = retreat_goal
        self._retreat._randomize_goal_pos = False

        # Approach goal in front of tray at current eef height
        tray = _get_named(env, name_tray)
        eef_z = env.robots[0].get_eef_position()[2]
        goal_xy = _front_target(tray, offset=self._offset)[:2]
        self._approach._goal_pos = th.tensor([goal_xy[0], goal_xy[1], eef_z], dtype=th.float32)
        self._approach._randomize_goal_pos = False

        # Reset approach subtask (sets its goal based on target object)
        self._approach.reset(env)
        self._phase = 0
        self._verify_steps = 0

    def _grasp_predicate(self, env) -> bool:
        """Simple proximity-based predicate as a stand-in for a true grasp state."""
        obj = _get_named(env, self._target)
        if obj is None or not obj.exists:
            return False
        eef = env.robots[0].get_eef_position()
        pos, _ = obj.get_position_orientation()
        return float(th.norm(eef - pos)) < max(0.5 * self._ee_tol, 0.04)

    def step(self, env, action):
        reward = 0.0
        done = False
        info = {"done": {"success": False, "termination_conditions": {}}}

        if self._phase == 0:
            r0, done0, info0 = self._approach.step(env=env, action=action)
            breakpoint()
            if isinstance(r0, (int, float)):
                reward += float(r0)
                print(f"reward : {reward}" )
            # Only use subtask success to advance; do not leak terminations
            success0 = bool(info0.get("done", {}).get("success", False)) if isinstance(info0, dict) else False
            if success0:
                print(f"Pick up tray first stage, reach to tray : {success0}")
                self._phase = 1
            else:
                info["done"]["termination_conditions"].clear()

        if self._phase == 1 and not done:
            self._verify_steps += 1
            grasp_ok = self._grasp_predicate(env)
            if grasp_ok:
                print(f"Pick up tray second stage, grasp to tray : {done}")
                r2, d2, i2 = self._retreat.step(env=env, action=action)
                if isinstance(r2, (int, float)):
                    reward += float(r2)
                success2 = bool(i2.get("done", {}).get("success", False)) if isinstance(i2, dict) else False
                if success2:
                    done = True
                    info["done"]["success"] = True
                    info["done"]["termination_conditions"]["predicate_retreat"] = {"done": True}
                    print(f"Pick up tray third stage, retriet : {done}")
            if not done:
                info["done"]["termination_conditions"].clear()

        if not done:
            d, dinfo = super()._step_termination(env=env, action=action, info={"done": {"success": False, "termination_conditions": {}}})
            if d:
                tc = dinfo.get("done", {}).get("termination_conditions", {}) if isinstance(dinfo, dict) else {}
                fixed_tc = {}
                if isinstance(tc, dict):
                    for k, v in tc.items():
                        fixed_tc[k] = v if isinstance(v, dict) else {"done": bool(v)}
                if not fixed_tc:
                    fixed_tc["timeout"] = {"done": True}
                done = True
                info["done"]["termination_conditions"].update(fixed_tc)

        return reward, done, info

    @classproperty
    def default_termination_config(cls):
        return {"max_steps": 3000}

    @classproperty
    def default_reward_config(cls):
        return {}

    @classproperty
    def valid_scene_types(cls):
        return {TraversableScene}


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

    # Satisfy abstract BaseTask API
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
        if a is None or b is None:
            missing = []
            if a is None:
                missing.append(self._target_object_name)
            if b is None:
                missing.append(self._source_object_name)
            info["done"]["termination_conditions"] = {"object_not_found": {"done": True, "which": missing}}
            return 0.0, True, info

        cur = self._get_state(a, b)

        if cur == self._val:
            info["done"]["success"] = True
            info["done"]["termination_conditions"] = {"predicate": {"done": True}}
            return 1.0, True, info

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
        return 0.0, done_out, info

    @classproperty
    def valid_scene_types(cls):
        return {TraversableScene}


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


class SufficientlyOpenTask(BaseTask):
    """
    Succeeds when the target object's door / openable joint is sufficiently open.
    Uses angle threshold for revolute joints or fractional opening for generic joints.

    Args:
        target_object_name (str): name of the object to evaluate (e.g., fridge)
        min_deg (float): minimum degrees for revolute joints to consider sufficiently open
        min_frac (float): minimum normalized fraction [0..1] considered open if angle check is not applicable
    """

    def __init__(
            self,
            target_object_name: str,
            min_deg: float = 90.0,
            min_frac: float = 0.8,
            termination_config=None,
            reward_config=None,
            include_obs: bool = False,
    ):
        self._target = target_object_name
        self._min_deg = float(min_deg)
        self._min_frac = float(min_frac)
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

    def step(self, env, action):
        info = {"done": {"success": False, "termination_conditions": {}}}
        obj = env.scene.object_registry("name", self._target)
        if obj is None:
            info["done"]["termination_conditions"] = {"object_not_found": {"done": True}}
            return 0.0, True, info

        ok = _is_open_ge_angle(obj, min_deg=self._min_deg, min_frac=self._min_frac)
        # If not OK, also compute fraction for debugging (not used in termination)
        frac = _open_fraction(obj)
        info["open_fraction"] = frac
        if ok:
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
        return 0.0, done_out, info

    @classproperty
    def valid_scene_types(cls):
        return {TraversableScene}


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
        # Store target and offset
        self._target_object_name = target_object_name
        self._front_offset = float(front_offset)
        term_cfg = dict(termination_config or {})
        if max_steps is not None:
            term_cfg["max_steps"] = int(max_steps)

        super().__init__(
            robot_idn=robot_idn,
            goal_tolerance=goal_tolerance,
            termination_config=term_cfg,
            reward_config=reward_config,
            include_obs=include_obs,
            visualize_goal=visualize_goal,
            **kwargs,
        )

    def _eef_positions(self, env):
        r = env.robots[self._robot_idn]
        poses = {}
        try:
            poses["left"] = r.get_eef_position(arm="left")
        except Exception:
            pass
        try:
            poses["right"] = r.get_eef_position(arm="right")
        except Exception:
            pass
        if not poses:
            # Fallback if robot API doesn’t expose arm selector
            poses["default"] = r.get_eef_position()
        return poses

    def _select_arm_pos(self, env):
        # Pick the arm whose EEF is closer to the goal
        poses = self._eef_positions(env)
        goal = self.get_goal_pos()  # PointReachingTask accessor
        best_key, best_pos, best_d = None, None, None
        for k, p in poses.items():
            d = float(th.norm(p - goal))
            if best_d is None or d < best_d:
                best_key, best_pos, best_d = k, p, d
        return best_pos, best_key

    def reset(self, env):
        obj = _get_named(env, self._target_object_name)
        goal_xy = _front_target(obj, offset=self._front_offset)[:2]

        eef_pos, _ = self._select_arm_pos(env)
        eef_z = eef_pos[2] if isinstance(eef_pos, th.Tensor) else eef_pos[2]
        self._goal_pos = th.tensor([goal_xy[0], goal_xy[1], eef_z], dtype=th.float32)
        self._randomize_goal_pos = False

        super().reset(env)

    def get_current_pos(self, env):
        pos, _ = self._select_arm_pos(env)
        return pos

    def _get_l2_potential(self, env):
        # Use the same EEF we use for termination so shaping matches success detection
        eef = self.get_current_pos(env)
        return T.l2_distance(eef, self._goal_pos)
