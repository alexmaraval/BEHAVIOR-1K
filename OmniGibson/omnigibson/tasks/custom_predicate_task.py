from omnigibson.object_states.inside import Inside
from omnigibson.object_states.next_to import NextTo
from omnigibson.object_states.on_top import OnTop
from omnigibson.object_states.open_state import Open
from omnigibson.object_states.robot_related_states import IsGrasping
from omnigibson.object_states.toggle import ToggledOn
from omnigibson.tasks.custom_task_base import BaseTask
from omnigibson.tasks.task_utils import _get_named
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.python_utils import classproperty


class _PredicateToggleTask(BaseTask):
    def __init__(
        self,
        target_object_name: str,
        desired_predicate: str,
        desired_value: bool,
        termination_config=None,
        reward_config=None,
    ):
        self._target_object_name = target_object_name
        self._pred = desired_predicate.lower()
        self._val = bool(desired_value)

        term_cfg = dict(termination_config or {})
        term_cfg.setdefault("max_steps", 4000)

        super().__init__(termination_config=term_cfg, reward_config=reward_config)

    def _create_termination_conditions(self):
        return {"timeout": Timeout(max_steps=self._termination_config["max_steps"])}

    def _create_reward_functions(self):
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


class OnTask(_PredicateToggleTask):
    def __init__(self, target_object_name: str, **kwargs):
        super().__init__(target_object_name=target_object_name, desired_predicate="on", desired_value=True, **kwargs)


class OpenTask(_PredicateToggleTask):
    def __init__(self, target_object_name: str, **kwargs):
        super().__init__(target_object_name=target_object_name, desired_predicate="open", desired_value=True, **kwargs)


class CloseTask(_PredicateToggleTask):
    def __init__(self, target_object_name: str, **kwargs):
        super().__init__(target_object_name=target_object_name, desired_predicate="open", desired_value=False, **kwargs)


class _RelativeStatusTask(BaseTask):
    def __init__(
        self,
        target_object_name: str,
        source_object_name: str,
        desired_predicate: str,
        desired_value: bool,
        termination_config=None,
        reward_config=None,
    ):
        self._target_object_name = target_object_name
        self._source_object_name = source_object_name
        self._pred = desired_predicate.lower()
        self._val = bool(desired_value)

        term_cfg = dict(termination_config or {})
        term_cfg.setdefault("max_steps", 4000)
        super().__init__(termination_config=term_cfg, reward_config=reward_config or {})

    def _create_termination_conditions(self):
        return {"timeout": Timeout(max_steps=self._termination_config["max_steps"])}

    def _create_reward_functions(self):
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
            return b.states[Inside].get_value(a)
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


class OnTopStableTask(BaseTask):
    def __init__(
        self,
        target_object_name: str,
        source_object_name: str,
        xy_tol: float = 0.23,  # center alignment tolerance
        require_release: bool = True,  # must not be grasped by robot
        termination_config=None,
        reward_config=None,
    ):
        self._tgt = target_object_name
        self._src = source_object_name
        self._xy_tol = xy_tol
        self._require_release = require_release
        self._grasping_arm = None
        term_cfg = termination_config or {}
        term_cfg.setdefault("max_steps", 4000)
        super().__init__(termination_config=term_cfg, reward_config=reward_config or {})

    def _create_termination_conditions(self):
        return {"timeout": Timeout(max_steps=self._termination_config["max_steps"])}

    def _create_reward_functions(self):
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
        tgt = _get_named(env, self._tgt)
        src = _get_named(env, self._src)
        if tgt is None or src is None:
            info["done"]["termination_conditions"] = dict(object_not_found={"done": True})
            return 0.0, True, info

        on_top_now = src.states[OnTop].get_value(tgt)

        released = True
        if self._require_release:
            robot = env.robots[0]
            released = not robot.states[IsGrasping].get_value(src)

        if on_top_now and released:
            info["done"]["success"] = True
            info["done"]["termination_conditions"] = dict(predicate={"done": True})
            return 1.0, True, info

        # Allow timeout
        base_done, base_info = super()._step_termination(
            env=env, action=action, info={"done": {"success": False, "termination_conditions": {}}}
        )
        tc = dict(base_info.get("done", {}).get("termination_conditions", {}))
        for k, v in list(tc.items()):
            if not isinstance(v, dict):
                tc[k] = {"done": bool(v)}
        if base_done and not any(d.get("done", False) for d in tc.values()):
            tc.setdefault("timeout", {"done": True})
        info["done"]["termination_conditions"] = tc
        done_out = any(d.get("done", False) for d in tc.values())
        return 0, done_out, info
