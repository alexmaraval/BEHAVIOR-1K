import omnigibson.utils.transform_utils as T
import torch as th
from omnigibson.tasks.custom_base_navigation_task import BaseNavigationTask
from omnigibson.tasks.task_utils import _get_named, _front_target
from omnigibson.termination_conditions.point_goal import PointGoal
from omnigibson.termination_conditions.termination_condition_base import SuccessCondition

# Valid point navigation reward types
POINT_NAVIGATION_REWARD_TYPES = {"l2", "geodesic"}


class EEFsReachingTask(BaseNavigationTask):
    """
    Point Reaching Task
    The goal is to reach a random goal position with the robot's end effector

    Args:
        robot_idn (int): Which robot that this task corresponds to
        floor (int): Which floor to navigate on
        initial_pos (None or 3-array): If specified, should be (x,y,z) global initial position to place the robot
            at the start of each task episode. If None, a collision-free value will be randomly sampled
        initial_quat (None or 3-array): If specified, should be (r,p,y) global euler orientation to place the robot
            at the start of each task episode. If None, a value will be randomly sampled about the z-axis
        goal_pos (None or 3-array): If specified, should be (x,y,z) global goal position to reach for the given task
            episode. If None, a collision-free value will be randomly sampled
        goal_tolerance (float): Distance between goal position and current position below which is considered a task
            success
        goal_in_polar (bool): Whether to represent the goal in polar coordinates or not when capturing task observations
        path_range (None or 2-array): If specified, should be (min, max) values representing the range of valid
            total path lengths that are valid when sampling initial / goal positions
        height_range (None or 2-array): If specified, should be (min, max) values representing the range of valid
            total heights that are valid when sampling goal positions
        visualize_goal (bool): Whether to visualize the initial / goal locations
        visualize_path (bool): Whether to visualize the path from initial to goal location, as represented by
            discrete waypoints
        goal_height (float): If visualizing, specifies the height of the visual goals (m)
        waypoint_height (float): If visualizing, specifies the height of the visual waypoints (m)
        waypoint_width (float): If visualizing, specifies the width of the visual waypoints (m)
        n_vis_waypoints (int): If visualizing, specifies the number of waypoints to generate
        termination_config (None or dict): Keyword-mapped configuration to use to generate termination conditions. This
            should be specific to the task class. Default is None, which corresponds to a default config being usd.
            Note that any keyword required by a specific task class but not specified in the config will automatically
            be filled in with the default config. See cls.default_termination_config for default values used
        reward_config (None or dict): Keyword-mapped configuration to use to generate reward functions. This should be
            specific to the task class. Default is None, which corresponds to a default config being usd. Note that
            any keyword required by a specific task class but not specified in the config will automatically be filled
            in with the default config. See cls.default_reward_config for default values used
        include_obs (bool): Whether to include observations or not for this task
    """

    def __init__(
        self,
        target_object_name: str,
        robot_idn=0,
        floor=0,
        initial_pos=None,
        initial_quat=None,
        goal_pos=None,
        goal_tolerance=0.1,
        goal_in_polar=False,
        path_range=None,
        goal_height=0.06,
        waypoint_height=0.05,
        waypoint_width=0.1,
        n_vis_waypoints=10,
        reward_config=None,
        termination_config=None,
        include_obs=True,
        skip_collision_with_objs=None,
    ):
        # Run super
        super().__init__(
            target_object_name=target_object_name,
            robot_idn=robot_idn,
            floor=floor,
            initial_pos=initial_pos,
            initial_quat=initial_quat,
            goal_pos=goal_pos,
            goal_tolerance=goal_tolerance,
            goal_in_polar=goal_in_polar,
            path_range=path_range,
            goal_height=goal_height,
            waypoint_height=waypoint_height,
            waypoint_width=waypoint_width,
            n_vis_waypoints=n_vis_waypoints,
            reward_type="l2",  # Must use l2 for reaching task
            reward_config=reward_config,
            termination_config=termination_config,
            include_obs=include_obs,
            skip_collision_with_objs=skip_collision_with_objs,
        )

    def _create_termination_conditions(self):
        # Run super first
        terminations = super()._create_termination_conditions()

        # We replace the pointgoal condition with a new one, specifying xyz instead of only xy as the axes to measure
        # distance to the goal
        terminations["pointgoal"] = PointGoal(
            robot_idn=self._robot_idn,
            distance_tol=self._goal_tolerance,
            distance_axes="xyz",
        )

        return terminations

    def _get_l2_potential(self, env):
        # Distance calculated from robot EEF, not base!
        return T.l2_distance(env.robots[self._robot_idn].get_eef_position(), self._goal_pos)

    def _get_obs(self, env):
        # Get obs from super
        low_dim_obs, obs = super()._get_obs(env=env)

        # Remove xy-pos and replace with full xyz relative distance between current and goal pos
        low_dim_obs.pop("xy_pos_to_goal")
        low_dim_obs["eef_to_goal"] = self._global_pos_to_robot_frame(env=env, pos=self._goal_pos)

        # Add local eef position as well
        low_dim_obs["eef_local_pos"] = self._global_pos_to_robot_frame(
            env=env, pos=env.robots[self._robot_idn].get_eef_position()
        )

        return low_dim_obs, obs

    def get_current_pos(self, env):
        # Current position is the robot's EEF, not base!
        return env.robots[self._robot_idn].get_eef_position()


class MoveEEToObjectTask(EEFsReachingTask):
    def __init__(
        self,
        target_object_name: str,
        front_offset: float = 0.0,
        robot_idn: int = 0,
        goal_tolerance: float = 0.04,
        max_steps: int | None = None,
        termination_config=None,
        reward_config=None,
        include_obs: bool = True,
        **kwargs,
    ):
        self._front_offset = front_offset
        self.prev_reward = 0.0

        term_cfg = dict(termination_config or {})
        if max_steps is not None:
            term_cfg["max_steps"] = max_steps

        # Initialize the navigation task
        super().__init__(
            target_object_name=target_object_name,
            robot_idn=robot_idn,
            goal_tolerance=goal_tolerance,
            termination_config=term_cfg,
            reward_config=reward_config,
            include_obs=include_obs,
            **kwargs,
        )

    def reset(self, env):
        eef = th.as_tensor(env.robots[self._robot_idn].get_eef_position(), dtype=th.float32)

        # Set goal to the closest point on the object's AABB
        obj = _get_named(env, self._target_object_name)
        goal_pos = None
        if obj is not None:
            lo, hi = obj.aabb  # each th.tensor(3)
            lo = th.as_tensor(lo, dtype=th.float32)
            hi = th.as_tensor(hi, dtype=th.float32)
            closest = th.maximum(th.minimum(eef, hi), lo)  # clamp EEF onto the box
            goal_pos = closest

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
                except TypeError:
                    pass
                try:
                    poses.append(r.get_eef_position(arm="right"))
                except TypeError:
                    pass

                if not poses:
                    poses.append(r.get_eef_position())

                min_dist = None
                obj = _get_named(env, self._target)
                if obj is not None:
                    for p in poses:
                        val = self._closest_xy_dist(p[:2], obj)
                        min_dist = val if min_dist is None else min(min_dist, val)

                if min_dist is None:
                    # Fallback to fixed goal
                    goal_xy = task.get_goal_pos()[:2]
                    for p in poses:
                        val = float(th.norm(p[:2] - goal_xy))
                        min_dist = val if min_dist is None else min(min_dist, val)

                return (min_dist is not None) and (min_dist <= self._tol)

        terms["pointgoal"] = _EEFsToObjectAABBXYLE(
            self._robot_idn,
            self._target_object_name,
            self._goal_tolerance,
        )
        return terms
