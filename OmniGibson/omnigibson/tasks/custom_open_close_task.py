import math
from omnigibson.object_states.open_state import _compute_joint_threshold
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.tasks.custom_task_base import BaseTask
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.constants import JointType
from omnigibson.utils.python_utils import classproperty


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
            closed_ok = total > 1e-6 and frac <= max_frac
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
        per_joint.append(
            {
                "name": nm,
                "type": j.joint_type,
                "opened": opened,
                "total": total,
                "fraction": frac,
                "angle_deg": angle_deg,
            }
        )

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

    def _get_obs(self, env):
        return {}, {}

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
        info = dict(done={"success": False, "termination_conditions": {}})
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

        # Delta dense reward on bounded progress
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
    def default_reward_config(cls):
        return {"r_scale": 1.0, "r_success": 10.0, "r_clip_abs": 0.0, "r_offset": 0.0}


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
        self._prev_progress = None  # track previous closed progress

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
