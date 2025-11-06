import omnigibson as og
from omnigibson.macros import gm
import omnigibson.object_states as object_states
import pathlib
import numpy as np
import csv
import tqdm
from omegaconf import DictConfig, OmegaConf
from omnigibson.robots import BaseRobot
from omnigibson.object_states import Pose
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.motion_planning_utils import astar
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    load_available_tasks,
    generate_robot_config,
    get_task_relevant_room_types,
)
from omnigibson.learning.utils.eval_utils import (
    generate_basic_environment_config,
)
import json
import os
import sys
import hydra
from signal import signal, SIGINT
import traceback
import torch as th
import math
import cv2

PIPELINE_ROOT = pathlib.Path('/home/j84403411/BEHAVIOR-1K/asset_pipeline')
RESOLUTION = 0.1
Z_START = 2.  # Just above the typical robot height
Z_END = -0.1  # Just below the floor
HALF_Z = (Z_START + Z_END) / 2.
HALF_HEIGHT = (Z_START - Z_END) / 2.

WALL_CATEGORIES = ["walls", "rail_fence"]
FLOOR_CATEGORIES = ["floors", "driveway", "lawn"]
DOOR_CATEGORIES = ["door", "sliding_door", "garage_door", "gate"]
IGNORE_CATEGORIES = ["carpet"]
NEEDED_STRUCTURE_CATEGORIES = FLOOR_CATEGORIES + WALL_CATEGORIES

# Segmentation maps will be generated with the data from the below map's overlap query
GENERATE_SEG_MAPS_DURING_FNAME = "floor_trav_no_obj_0.png"

# Segmentation maps will be saved with the below filenames even though they don't have their
# own passes.
SEMSEG_MAP_FNAME = "floor_semseg_0.png"
INSSEG_MAP_FNAME = "floor_insseg_0.png"


MAP_GENERATION_PASSES = [
    [
        ("floor_trav_0.png", None, IGNORE_CATEGORIES),
    ]
]

class TestEnv:
    """
    Evaluator class for running and evaluating policies for behavior task.
    This class manages the setup, execution, and evaluation of policy rollouts in OmniGibson environment,
    tracking metrics such as the number of trials, successes, and total time. It supports loading environments,
    robots, policies, and metrics, and provides methods for stepping through the environment, resetting state,
    and handling video outputs and loggings.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()

        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        self.robot = self.load_robot()

        self.reset()
    
    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        """
        Read the environment config file and create the environment.
        The config file is located in the configs/envs directory.
        """
        # Disable a subset of transition rules for data collection
        task_name = self.cfg.task.name
        available_tasks = load_available_tasks()
        # Load the seed instance by default
        task_cfg = available_tasks[task_name][0]
        robot_type = self.cfg.robot.type
        assert robot_type == "R1Pro", f"Got invalid robot type: {robot_type}, only R1Pro is supported."
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            generate_robot_config(
                task_name=task_name,
                task_cfg=task_cfg,
            )
        ]
        print("load env...")
        env = og.Environment(configs=cfg)

        return env

    def load_robot(self) -> BaseRobot:
        """
        Loads and returns the robot instance from the environment.
        Returns:
            BaseRobot: The robot instance loaded from the environment.
        """
        print("load robot...")
        robot = self.env.scene.object_registry("name", "robot_r1")
        return robot

    def load_task_instance(self, instance_id: int) -> None:
        """
        Loads the configuration for a specific task instance.

        Args:
            instance_id (int): The ID of the task instance to load.
        """
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        tro_file_path = os.path.join(
            get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
        )
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = tro_state
                robot_pos = presampled_robot_poses[self.robot.model_name][0]["position"]
                robot_quat = presampled_robot_poses[self.robot.model_name][0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)
                # Write robot poses to scene metadata
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        # Try to ensure that all task-relevant objects are stable
        # They should already be stable from the sampled instance, but there is some issue where loading the state
        # causes some jitter (maybe for small mass / thin objects?)
        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()


    def reset(self) -> None:
        self.env.reset()


    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # print stats
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        self.video_writer = None
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        self.__exit__(None, None, None)
        sys.exit(0)
        


class FindPath:
    def __init__(
        self,
        env,
        map_resolution,
        waypoint_resolution
    ):
        self.env = env
        self.map_resolution = map_resolution
        self.waypoint_resolution = waypoint_resolution
        self.waypoint_interval = int(waypoint_resolution/map_resolution)

    def build_map(self):
        # Compute the map dimensions by finding the AABB of all objects and calculating max distance from origin.
        floor_objs = {
            floor
            for floor_cat in FLOOR_CATEGORIES
            for floor in og.sim.scenes[0].object_registry("category", floor_cat, [])
        }
        roomless_floor_objs = [(floor, len(floor.in_rooms)) for floor in floor_objs if len(floor.in_rooms) != 1]
        assert not roomless_floor_objs, f"Found {len(roomless_floor_objs)} floor objects without exactly one room: {roomless_floor_objs}"
        aabb_corners = np.concatenate([floor.aabb for floor in floor_objs], axis=0)
        combined_low = np.min(list(aabb_corners), axis=0)
        combined_high = np.max(list(aabb_corners), axis=0)
        combined_aabb = np.array([combined_low, combined_high])
        aabb_dist_from_zero = np.abs(combined_aabb)
        dist_from_center = np.max(aabb_dist_from_zero)
        map_size_in_meters = dist_from_center * 2
        map_size_in_pixels = map_size_in_meters / self.map_resolution
        map_size_in_pixels = int(np.ceil(map_size_in_pixels / 2) * 2) + 2  # Round to nearest multiple of 2

        # Get the bounds of the part of the map that we will actually cast rays for (e.g. the occupied section)
        world_to_map_float = lambda xy: np.flip((np.array(xy) / self.map_resolution + map_size_in_pixels / 2.0))

        row_min, col_min = np.floor(world_to_map_float(combined_aabb[0][:2])).astype(int)
        row_max, col_max = np.ceil(world_to_map_float(combined_aabb[1][:2])).astype(int)

        # Assert that all the dimensions are within the map
        assert row_min >= 0 and row_max < map_size_in_pixels, f"Map row bounds: {row_min}, {row_max} vs {map_size_in_pixels}"
        assert col_min >= 0 and col_max < map_size_in_pixels, f"Map column bounds: {col_min}, {col_max} vs {map_size_in_pixels}"

        row_extent = row_max - row_min + 1
        col_extent = col_max - col_min + 1
        total_cells = row_extent * col_extent

        for pass_idx, map_pass in enumerate(MAP_GENERATION_PASSES):
            # Move the doors to the open position if necessary
            if map_pass[0][0] == "floor_trav_open_door_0.png":
                for door_cat in DOOR_CATEGORIES:
                    for door in og.sim.scenes[0].object_registry("category", door_cat, []):
                        if object_states.Open not in door.states:
                            continue
                        door.states[object_states.Open].set_value(True, fully=True)

                og.sim.step()

            allowed_hit_paths_by_fname = {}
            for fname, load_categories, not_load_categories in map_pass:
                # Using the load/not load params, build the set of allowed hits
                allowed_hit_paths_for_fname = {
                    link.prim_path: obj
                    for obj in og.sim.scenes[0].objects if obj.name!="robot_r1"
                    for link in obj.links.values()
                    if not load_categories or obj.category in load_categories
                }
                if not_load_categories:
                    for obj in og.sim.scenes[0].objects:
                        if obj.name!="robot_r1":
                            for link in obj.links.values():
                                if obj.category in not_load_categories:
                                    allowed_hit_paths_for_fname.pop(link.prim_path, None)

                # Add the allowed hit paths to the dictionary
                allowed_hit_paths_by_fname[fname] = allowed_hit_paths_for_fname

            # Prepare the arrays for the maps
            map_fnames = {fname for fname, _, _ in map_pass}

            map_arrays = {fname: np.zeros((map_size_in_pixels, map_size_in_pixels), dtype=np.uint8) for fname in map_fnames}

            # Do the actual ray casting (actually an overlap query). We make a single pass for each
            # map pass, relying on the callback to filter out the hits we don't want for each map file.
            with tqdm.tqdm(total=total_cells, desc=f"Overlap grid for pass {pass_idx}") as pbar:
                for row in range(row_min, row_max + 1):
                    for col in range(col_min, col_max + 1):
                        world_pos = self.map_to_world(np.array([row, col]), map_size_in_pixels)

                        hit_objs_by_fname = {fname: set() for fname in map_fnames}
                        def _check_hit(hit):
                            for fname, allowed_hit_paths_for_fname in allowed_hit_paths_by_fname.items():
                                if hit.rigid_body in allowed_hit_paths_for_fname:
                                    hit_objs_by_fname[fname].add(allowed_hit_paths_for_fname[hit.rigid_body])
                                
                            return True
                            
                        # Run the actual overlap query
                        og.sim.psqi.overlap_box(
                            halfExtent=np.array([self.map_resolution / 2, self.map_resolution / 2, HALF_HEIGHT]),
                            pos=np.array([world_pos[0], world_pos[1], HALF_Z]),
                            rot=np.array([0, 0, 0, 1.0]),
                            reportFn=_check_hit,
                        )

                        # Use the results from the hit_objs_by_fname to fill in the map arrays
                        for fname, load_categories, not_load_categories in map_pass:
                            # Get the hit object set for this map
                            hit_objs = hit_objs_by_fname[fname]

                            # Check whether or not we only hit a floor
                            only_hit_floor = int(hit_objs.issubset(floor_objs))
                            
                            # Assign the reshaped array to the scannable map
                            map_arrays[fname][row, col] = only_hit_floor * 255
                            
                        # Update the progress bar
                        pbar.update(1)

            return map_arrays
    

    def erode_trav_map(self, trav_map, robot=None):
        # Erode the traversability map to account for the robot's size
        if robot:
            robot_chassis_extent = robot.reset_joint_pos_aabb_extent[:2]
            radius = th.norm(robot_chassis_extent) / 2.0 + 0.2
        else:
            radius = self.default_erosion_radius
        radius_pixel = int(math.ceil(radius / self.map_resolution))
        trav_map = th.tensor(cv2.erode(trav_map.cpu().numpy(), th.ones((radius_pixel, radius_pixel)).cpu().numpy()))
        return trav_map

    
    def map_to_world(self, xy, map_size):
        """
        Transforms a 2D point in map reference frame into world (simulator) reference frame

        Args:
            xy (2-array or (N, 2)-array): 2D location(s) in map reference frame (in image pixel space)

        Returns:
            2-array or (N, 2)-array: 2D location(s) in world reference frame (in metric space)
        """
        axis = 0 if len(xy.shape) == 1 else 1
        return np.flip((xy - map_size / 2.0) * self.map_resolution, axis=axis)
    
        
    def world_to_map(self, xy, map_size):
        """
        Transforms a 2D point in world (simulator) reference frame into map reference frame

            xy: 2D location in world reference frame (metric)
        :return: 2D location in map reference frame (image)
        """

        point_wrt_map = xy / self.map_resolution + map_size / 2.0
        return th.flip(point_wrt_map, dims=tuple(range(point_wrt_map.dim()))).int()

    
    def get_shortest_path(self, obj):
        """
        Get the shortest path from one point to another point.
        If any of the given point is not in the graph, add it to the graph and
        create an edge between it to its closest node.

        Args:
            floor (int): floor number
            source_world (2-array): (x,y) 2D source location in world reference frame (metric)
            target_world (2-array): (x,y) 2D target location in world reference frame (metric)
            entire_path (bool): whether to return the entire path
            robot (None or BaseRobot): if given, erode the traversability map to account for the robot's size

        Returns:
            2-tuple:
                - (N, 2) array: array of path waypoints, where N is the number of generated waypoints
                - float: geodesic distance of the path
        """
        ## x, y turn into tuple
       
        # create a deep copy so that we don't erode the original map
        trav_map = th.tensor(self.build_map()["floor_trav_0.png"])
        map_size = trav_map.shape[0]
        trav_map = self.erode_trav_map(trav_map, robot=self.env.env.robots[0])
        source_map = tuple(self.world_to_map(self.env.env.robots[0].states[Pose].get_value()[0][:2], map_size).tolist())
        target_map = tuple(self.world_to_map(self.env.env.scene.object_registry("name", obj).get_position_orientation()[0][:2], map_size).tolist())
        path_map = astar(trav_map, source_map, target_map)
        breakpoint()
        if path_map is None:
            # No traversable path found
            return None, None
        path_world = self.map_to_world(path_map, map_size)
        geodesic_distance = th.sum(th.norm(path_world[1:] - path_world[:-1], dim=1))
        path_world = path_world[:: self.waypoint_interval]
        breakpoint()
        return path_world, geodesic_distance

if __name__ == "__main__":
    ###TODO first load the scene from the env
    # register_omegaconf_resolvers()
        # open yaml from task path
    with hydra.initialize_config_dir("/home/j84403411/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs", version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)
    # set headless mode
    gm.HEADLESS = config.headless

    env = TestEnv(config)
    env.reset()
    env.load_task_instance(0)

    obj_name = "radio_89"
    map_resolution = 0.01
    waypoint_resolution = 0.2
    path =  FindPath(env, map_resolution, waypoint_resolution)
    astar_path = path.get_shortest_path(obj_name)
    ## radio name
    # radio_89