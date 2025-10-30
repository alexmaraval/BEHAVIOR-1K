import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.object_states.on_top import OnTop
from omnigibson.object_states import AttachedTo
from omnigibson.object_states.robot_related_states import IsGrasping
from omnigibson.termination_conditions.termination_condition_base import FailureCondition


class Falling(FailureCondition):
    """
    Falling (failure condition) used for any navigation-type tasks
    Episode terminates if the robot falls out of the world (i.e.: falls below the floor height by at least
    @fall_height

    Args:
        robot_idn (int): robot identifier to evaluate condition with. Default is 0, corresponding to the first
            robot added to the scene
        fall_height (float): distance (m) > 0 below the scene's floor height under which the the robot is considered
            to be falling out of the world
        topple (bool): whether to also consider the robot to be falling if it is toppling over (i.e.: if it is
            no longer upright
    """

    def __init__(self, robot_idn=0, fall_height=0.03, topple=True, tilt_tolerance=0.75):
        # Store internal vars
        self._robot_idn = robot_idn
        self._fall_height = fall_height
        self._topple = topple
        self._tilt_tolerance = tilt_tolerance

        # Run super init
        super().__init__()

    def _step(self, task, env, action):
        # Terminate if the specified robot is falling out of the scene
        robot_z = env.scene.robots[self._robot_idn].get_position_orientation()[0][2]
        if robot_z < (env.scene.get_floor_height() - self._fall_height):
            return True

        # Terminate if the robot has toppled over
        if self._topple:
            robot_up = T.quat_apply(
                env.scene.robots[self._robot_idn].get_position_orientation()[1], th.tensor([0, 0, 1], dtype=th.float32)
            )
            if robot_up[2] < self._tilt_tolerance:
                return True

        return False


class ObjectFalling(FailureCondition):
    """
    Object falling (failure condition) for manipulation-type tasks.
    Episode terminates if the specified object falls below the floor height
    by at least @fall_height.

    Args:
        obj_name (str): Name of the target object in the scene registry.
        fall_height (float): Distance (m) > 0 below the scene's floor height
            under which the object is considered to have fallen out of the world.
    """

    def __init__(
        self,
        obj_name: str,
        fall_height: float = 0.03,
        topple: bool = True,
        tilt_tolerance: float = 0.75,
        sustain_steps: int = 5,
        only_when_supported: bool = True,
        ignore_when_grasped: bool = True,
    ):
        self._obj_name = obj_name
        self._fall_height = fall_height
        self._topple = topple
        self._tilt_tolerance = tilt_tolerance
        self._sustain_steps = max(1, int(sustain_steps))
        self._only_when_supported = only_when_supported
        self._ignore_when_grasped = ignore_when_grasped
        self._violation_steps = 0
        super().__init__()

    def _step(self, task, env, action):
        obj = env.scene.object_registry("name", self._obj_name)
        if obj is None:
            # If object is not found, do not trigger termination here
            return False

        # Terminate if the specified object is falling out of the scene
        obj_z = obj.get_position_orientation()[0][2]
        if obj_z < (env.scene.get_floor_height() - self._fall_height):
            return True

        # Terminate if the object has toppled (not upright)
        if self._topple:
            # Skip if object is still grasped / attached to robot
            if self._ignore_when_grasped:
                for robot in env.scene.robots:
                    if (IsGrasping in robot.states and robot.states[IsGrasping].get_value(obj)) or (
                        AttachedTo in obj.states and obj.states[AttachedTo].get_value(robot)
                    ):
                        # Reset counter while grasped
                        self._violation_steps = 0
                        return False


            # Check support if requested (object is resting on a surface)
            is_supported = True
            if self._only_when_supported:
                is_supported = False
                for other in env.scene.objects:
                    if other in env.scene.robots:
                        continue
                    if OnTop in obj.states and obj.states[OnTop].get_value(other):
                        is_supported = True
                        break


            # Compute object "up" vector in world; compare with world +Z
            obj_up = T.quat_apply(obj.get_position_orientation()[1], th.tensor([0, 0, 1], dtype=th.float32))
            toppled = obj_up[2] < self._tilt_tolerance

            if toppled and is_supported:
                self._violation_steps += 1
                if self._violation_steps >= self._sustain_steps:
                    return True
                return False
            else:
                self._violation_steps = 0

        return False
