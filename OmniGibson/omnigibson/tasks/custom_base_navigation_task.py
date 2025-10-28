import omnigibson.utils.transform_utils as T
import torch as th
from omnigibson.object_states import Pose
from omnigibson.reward_functions.point_goal_reward import PointGoalReward
from omnigibson.reward_functions.potential_reward import PotentialReward
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.tasks.custom_task_base import BaseTask
from omnigibson.tasks.task_utils import _MaxCollisionFiltered, _CollisionRewardFiltered, _get_named, _front_target
from omnigibson.termination_conditions.falling import Falling
from omnigibson.termination_conditions.point_goal import PointGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.python_utils import assert_valid_key, classproperty

# Valid point navigation reward types
POINT_NAVIGATION_REWARD_TYPES = {"l2", "geodesic"}


class BaseNavigationTask(BaseTask):
    """
    Point Navigation Task
    The task is to navigate to a goal position

    Args:
        target_object_name: Target object name.
        front_offset: Front offset to calculate the goal position.
        robot_idn (int): Which robot that this task corresponds to
        floor (int): Which floor to navigate on
        goal_pos (None or 3-array): If specified, should be (x,y,z) global goal position to reach for the given task
            episode. If None, a collision-free value will be randomly sampled
        goal_tolerance (float): Distance between goal position and current position below which is considered a task
            success
        path_range (None or 2-array): If specified, should be (min, max) values representing the range of valid
            total path lengths that are valid when sampling initial / goal positions
        reward_type (str): Type of reward to use. Valid options are: {"l2", "geodesic"}
        termination_config (None or dict): Keyword-mapped configuration to use to generate termination conditions. This
            should be specific to the task class. Default is None, which corresponds to a default config being usd.
            Note that any keyword required by a specific task class but not specified in the config will automatically
            be filled in with the default config. See cls.default_termination_config for default values used
        reward_config (None or dict): Keyword-mapped configuration to use to generate reward functions. This should be
            specific to the task class. Default is None, which corresponds to a default config being usd. Note that
            any keyword required by a specific task class but not specified in the config will automatically be filled
            in with the default config. See cls.default_reward_config for default values used
    """

    def __init__(
        self,
        target_object_name: str,
        front_offset: float = 0.0,
        robot_idn=0,
        floor=0,
        goal_pos=None,
        goal_tolerance=0.5,
        path_range=None,
        reward_type="l2",
        termination_config=None,
        reward_config=None,
        skip_collision_with_objs=None,
    ):
        # Store inputs
        self._target_object_name = target_object_name
        self._front_offset = front_offset
        self._robot_idn = robot_idn
        self._floor = floor
        self._goal_pos = goal_pos if goal_pos is None else th.tensor(goal_pos)
        self._goal_tolerance = goal_tolerance
        self._path_range = path_range
        assert_valid_key(key=reward_type, valid_keys=POINT_NAVIGATION_REWARD_TYPES, name="reward type")
        self._reward_type = reward_type

        # Collision-skip support: store names and resolved objects
        self._skip_collision_with_objs_names = skip_collision_with_objs
        self.skip_collision_objs = []

        # Create other attributes that will be filled in at runtime
        self._path_length = None
        self._current_robot_pos = None
        self._geodesic_dist = None

        # Run super
        super().__init__(termination_config=termination_config, reward_config=reward_config)

    def _create_termination_conditions(self):
        # Initialize termination conditions dict and fill in with MaxCollision, Timeout, Falling, and PointGoal
        terminations = dict()

        terminations["max_collision"] = _MaxCollisionFiltered(
            self, max_collisions=self._termination_config["max_collisions"]
        )
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        terminations["falling"] = Falling(
            robot_idn=self._robot_idn, fall_height=self._termination_config["fall_height"]
        )
        terminations["pointgoal"] = PointGoal(
            robot_idn=self._robot_idn,
            distance_tol=self._goal_tolerance,
            distance_axes="xy",
        )

        return terminations

    def _create_reward_functions(self):
        # Initialize reward functions dict and fill in with Potential, Collision, and PointGoal rewards
        rewards = dict()

        rewards["potential"] = PotentialReward(
            potential_fcn=self.get_potential,
            r_potential=self._reward_config["r_potential"],
        )

        rewards["collision"] = _CollisionRewardFiltered(
            self, robot_idn=self._robot_idn, r_collision=self._reward_config["r_collision"]
        )
        rewards["pointgoal"] = PointGoalReward(
            pointgoal=self._termination_conditions["pointgoal"],
            r_pointgoal=self._reward_config["r_pointgoal"],
        )

        return rewards

    def _get_geodesic_potential(self, env):
        """
        Get potential based on geodesic distance

        Args:
            env: environment instance

        Returns:
            float: geodesic distance to the target position
        """
        _, geodesic_dist = self.get_shortest_path_to_goal(env=env)
        return geodesic_dist

    def _get_l2_potential(self, env):
        """
        Get potential based on L2 distance

        Args:
            env: environment instance

        Returns:
            float: L2 distance to the target position
        """
        return T.l2_distance(env.robots[self._robot_idn].states[Pose].get_value()[0][:2], self._goal_pos[:2])

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal

        Args:
            env (Environment): Environment instance

        Returns:
            float: Computed potential
        """
        if self._reward_type == "l2":
            potential = self._get_l2_potential(env)
        elif self._reward_type == "geodesic":
            potential = self._get_geodesic_potential(env)
            # If no path is found, fall back to L2 potential
            if potential is None:
                potential = self._get_l2_potential(env)
        else:
            raise ValueError(f"Invalid reward type! {self._reward_type}")

        return potential

    def _step_termination(self, env, action, info=None):
        # Run super first
        done, info = super()._step_termination(env=env, action=action, info=info)

        # Add additional info
        info["path_length"] = self._path_length

        return done, info

    def reset(self, env):
        """
        Resolve the object and set the navigation goal before the parent reset so built-in rewards/terminations see it.
        """

        self._current_robot_pos = env.robots[self._robot_idn].states[Pose].get_value()[0]
        base_z = self._current_robot_pos[2]
        goal_xy = _front_target(_get_named(env, self._target_object_name), offset=self._front_offset)[:2]
        self._goal_pos = th.tensor([goal_xy[0], goal_xy[1], base_z], dtype=th.float32)

        self._path_length = 0.0
        # Store only the position tensor (x,y,z), not the full (pos, quat) tuple
        self._geodesic_dist = self._get_geodesic_potential(env)

        # Resolve skip-collision objects by name
        self.skip_collision_objs = []
        if self._skip_collision_with_objs_names:
            for name in self._skip_collision_with_objs_names:
                obj = _get_named(env, name)
                if obj is not None:
                    self.skip_collision_objs.append(obj)

        super().reset(env)

    def _global_pos_to_robot_frame(self, env, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        Args:
            env (TraversableEnv): Environment instance
            pos (th.Tensor): global (x,y,z) position

        Returns:
            th.Tensor: (x,y,z) position in self._robot_idn agent's local frame
        """
        delta_pos_global = pos - env.robots[self._robot_idn].states[Pose].get_value()[0]
        return T.quat2mat(env.robots[self._robot_idn].states[Pose].get_value()[1]).T @ delta_pos_global


    def get_goal_pos(self):
        """
        Returns:
            3-array: (x,y,z) global current goal position
        """
        return self._goal_pos

    def get_current_pos(self, env):
        """
        Returns:
            3-array: (x,y,z) global current position representing the robot
        """
        return env.robots[self._robot_idn].states[Pose].get_value()[0]

    def get_shortest_path_to_goal(self, env, start_xy_pos=None, entire_path=False):
        """
        Get the shortest path and geodesic distance from @start_pos to the target position

        Args:
            env (TraversableEnv): Environment instance
            start_xy_pos (None or 2-array): If specified, should be the global (x,y) start position from which
                to calculate the shortest path to the goal position. If None (default), the robot's current xy position
                will be used
            entire_path (bool): Whether to return the entire shortest path

        Returns:
            2-tuple:
                - list of 2-array: List of (x,y) waypoints representing the path # TODO: is this true?
                - float: geodesic distance of the path to the goal position
        """
        start_xy_pos = (
            env.robots[self._robot_idn].states[Pose].get_value()[0][:2] if start_xy_pos is None else start_xy_pos
        )
        return env.scene.get_shortest_path(
            self._floor, start_xy_pos, self._goal_pos[:2], entire_path=entire_path, robot=env.robots[self._robot_idn]
        )

    def step(self, env, action):
        # Run super method first
        reward, done, info = super().step(env=env, action=action)

        # Update other internal variables
        new_robot_pos = env.robots[self._robot_idn].states[Pose].get_value()[0]
        self._path_length += T.l2_distance(self._current_robot_pos[:2], new_robot_pos[:2])
        self._current_robot_pos = new_robot_pos

        return reward, done, info

    @classproperty
    def valid_scene_types(cls):
        # Must be a traversable scene
        return {TraversableScene}

    @classproperty
    def default_termination_config(cls):
        return {
            "max_collisions": 500,
            "max_steps": 500,
            "fall_height": 0.03,
        }

    @classproperty
    def default_reward_config(cls):
        return {
            "r_potential": 1.0,
            "r_collision": 0.1,
            "r_pointgoal": 10.0,
        }
