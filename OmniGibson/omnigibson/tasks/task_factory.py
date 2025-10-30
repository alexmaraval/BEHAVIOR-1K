from functools import partial

import numpy as np
from omnigibson.tasks.custom_base_navigation_task import BaseNavigationTask
from omnigibson.tasks.custom_grasp_task import RobustGraspTask
from omnigibson.tasks.custom_open_close_task import SufficientlyClosedTask, SufficientlyOpenTask
from omnigibson.tasks.custom_point_reaching_task import MoveEEToObjectTask
from omnigibson.tasks.custom_predicate_task import OnTask
from omnigibson.tasks.custom_predicate_task import OnTopStableTask, NextToTask, OnTopTask, InsideTask

max_steps = 5000
task_factory = {}


def get_sub_tasks(task_name: str) -> list[dict[str, ...]]:
    return task_factory[task_name]


# -----turning_on_radio-----

name_radio = "radio_89"
name_coffe_table = "coffee_table_koagbh_0"
radio_handle_transform = np.array(
    [[0.25, 0.43, 0.042],
     [0.41, 0.68, 0.067],
     [0.050, 0.084, 0.0083]]
)

tr_move_to_radio = partial(
    BaseNavigationTask,
    target_object_name=name_radio,
    goal_tolerance=1,
    termination_config={"max_steps": 5000},
)

tr_grasp_radio = partial(
    RobustGraspTask,
    obj_name=name_radio,
    termination_config={"max_steps": 5000},
    transform_matrix=radio_handle_transform,
)

tr_radio_on = partial(
    OnTask,
    target_object_name=name_radio,
)

stages = [
    {"name": "move_to_radio", "factory": tr_move_to_radio},
    {"name": "pick_radio", "factory": tr_grasp_radio},
    {"name": "radio_on", "factory": tr_radio_on},
    # {"name": "place_radio", "kind": "task", "factory": place_radio},
]
task_factory.update({"turning_on_radio": stages})

# -----cook_bacon-----
name_fridge = "fridge_dszchb_0"
name_countertop = "countertop_kelker_0"
name_pan = "frying_pan_207"
name_tray = "tray_208"
name_burner = "burner_mjvqii_0"
name_bacon_1 = "bacon_209"
name_bacon_2 = "bacon_210"
name_bacon_3 = "bacon_211"
name_bacon_4 = "bacon_212"
name_bacon_5 = "bacon_213"
name_bacon_6 = "bacon_214"

cb_move_to_fridge = partial(
    BaseNavigationTask,
    target_object_name=name_fridge,
    goal_tolerance=1.3,
    termination_config={"max_steps": 10000},
)

cb_open_fridge = partial(
    SufficientlyOpenTask,
    target_object_name=name_fridge,
    allowed_deg=80,
    termination_config={"max_steps": 10000},
)

cb_close_fridge = partial(
    SufficientlyClosedTask,
    target_object_name=name_fridge,
    allowed_deg=0,
    termination_config={"max_steps": 10000},
)

cb_move_to_counter_top = partial(
    BaseNavigationTask,
    target_object_name=name_countertop,
    goal_tolerance=1,
    termination_config={"max_steps": 10000},
    skip_collision_with_objs=[
        name_tray,
        name_bacon_1,
        name_bacon_2,
        name_bacon_3,
        name_bacon_4,
        name_bacon_5,
        name_bacon_6,
    ],
)

cb_place_next_to_burner = partial(
    NextToTask,
    target_object_name=name_tray,
    source_object_name=name_burner,
    desired_value=True,
    termination_config={"max_steps": 10000},
)

cb_move_to_frying_pan = partial(
    MoveEEToObjectTask,
    target_object_name=name_pan,
    goal_tolerance=0.05,
    termination_config={"max_steps": 10000},
    skip_collision_with_objs=[name_pan],
)

cb_move_to_tray = partial(
    MoveEEToObjectTask,
    target_object_name=name_tray,
    goal_tolerance=0.05,
    termination_config={"max_steps": 10000},
    skip_collision_with_objs=[name_tray],
)

cb_place_frying_pan = partial(
    OnTopStableTask,
    target_object_name=name_burner,
    source_object_name=name_pan,
    termination_config={"max_steps": 10000},
)

cb_burner_on = partial(
    OnTask,
    target_object_name=name_burner,
    termination_config={"max_steps": 10000},
)

cb_pour_tray = partial(
    OnTopTask,
    target_object_name=name_bacon_3,
    source_object_name=name_pan,
    desired_value=True,
    termination_config={"max_steps": 10000},
)

cb_grasp_tray = partial(
    RobustGraspTask,
    obj_name=name_tray,
    termination_config={"max_steps": 10000},
)

cb_grasp_pan = partial(
    RobustGraspTask,
    obj_name=name_pan,
    termination_config={"max_steps": 10000},
)

stages = [
    {"name": "move_to_fridge", "factory": cb_move_to_fridge},
    {"name": "open_fridge", "factory": cb_open_fridge},
    {"name": "pick_tray", "factory": cb_grasp_tray},
    {"name": "close_fridge", "factory": cb_close_fridge},
    {"name": "move_to_counter_top", "factory": cb_move_to_counter_top},
    {"name": "place_on_next_to_burner1", "factory": cb_place_next_to_burner},
    {"name": "Move_to_frying_pan", "factory": cb_move_to_frying_pan},
    {"name": "pick_up_frying_pan", "factory": cb_grasp_pan},
    {"name": "place_frying_pan", "factory": cb_place_frying_pan},
    {"name": "move_to_tray", "factory": cb_move_to_tray},
    {"name": "pick_up_tray", "factory": cb_grasp_tray},
    {"name": "pour_tray", "factory": cb_pour_tray},
    {"name": "place_on_next_to_burner2", "factory": cb_place_next_to_burner},
    {"name": "burner_on_switch", "factory": cb_burner_on},
]

task_factory.update({"cook_bacon": stages})


# -----freeze_pies-----

name_cabinet = "bottom_cabinet_no_top_gjeoer_0"
name_fridge = "fridge_dszchb_0"
name_countertop = "countertop_kelzer_0"
name_tupperware_1 = "tupperware_230"
name_tupperware_2 = "tupperware_231"
name_apple_pie_1 = "apple_pie_235"
name_apple_pie_2 = "apple_pie_234"


fp_move_to_cabinet = partial(
    BaseNavigationTask,
    target_object_name=name_cabinet,
    goal_tolerance=1.3,
    termination_config={"max_steps": 10000},
)

fp_move_to_fridge = partial(
    BaseNavigationTask,
    target_object_name=name_fridge,
    goal_tolerance=1.3,
    termination_config={"max_steps": 10000},
)

fp_open_cabinet = partial(
    SufficientlyOpenTask,
    target_object_name=name_cabinet,
    allowed_deg=80,
    termination_config={"max_steps": 10000},
)

fp_close_cabinet = partial(
    SufficientlyClosedTask,
    target_object_name=name_cabinet,
    allowed_deg=0,
    termination_config={"max_steps": 10000},
)

fp_close_fridge = partial(
    SufficientlyClosedTask,
    target_object_name=name_fridge,
    allowed_deg=0,
    termination_config={"max_steps": 10000},
)

fp_open_fridge = partial(
    SufficientlyOpenTask,
    target_object_name=name_fridge,
    allowed_deg=80,
    termination_config={"max_steps": 10000},
)

fp_grasp_tupperware_1 = partial(
    RobustGraspTask,
    obj_name=name_tupperware_1,
    termination_config={"max_steps": 5000},
    transform_matrix=None,
)

fp_grasp_tupperware_2 = partial(
    RobustGraspTask,
    obj_name=name_tupperware_2,
    termination_config={"max_steps": 5000},
    transform_matrix=None,
)

fp_move_to_counter_top = partial(
    BaseNavigationTask,
    target_object_name=name_countertop,
    goal_tolerance=1,
    termination_config={"max_steps": 10000},
    skip_collision_with_objs=[],
)

fp_move_to_apple_pie_2 = partial(
    BaseNavigationTask,
    target_object_name=name_apple_pie_2,
    goal_tolerance=1,
    termination_config={"max_steps": 10000},
    skip_collision_with_objs=[],
)

fp_place_tupperware_1_countertop = partial(
    OnTopStableTask,
    target_object_name=name_countertop,
    source_object_name=name_tupperware_1,
    termination_config={"max_steps": 10000},
)

fp_place_tupperware_2_countertop = partial(
    OnTopStableTask,
    target_object_name=name_countertop,
    source_object_name=name_tupperware_2,
    termination_config={"max_steps": 10000},
)

fp_grasp_apple_pie_1 = partial(
    RobustGraspTask,
    obj_name=name_apple_pie_1,
    termination_config={"max_steps": 10000},
)

fp_grasp_apple_pie_2 = partial(
    RobustGraspTask,
    obj_name=name_apple_pie_2,
    termination_config={"max_steps": 10000},
)


fp_place_apple_pie_1 = partial(
    InsideTask,
    target_object_name=name_tupperware_1,
    source_object_name=name_apple_pie_1,
    termination_config={"max_steps": 10000},
)

fp_place_apple_pie_2 = partial(
    InsideTask,
    target_object_name=name_tupperware_2,
    source_object_name=name_apple_pie_2,
    termination_config={"max_steps": 10000},
)

fp_move_to_tupperware_1 = partial(
    MoveEEToObjectTask,
    target_object_name=name_tupperware_1,
    goal_tolerance=0.05,
    termination_config={"max_steps": 10000},
    skip_collision_with_objs=[name_tray],
)

fp_move_to_tupperware_2 = partial(
    MoveEEToObjectTask,
    target_object_name=name_tupperware_2,
    goal_tolerance=0.05,
    termination_config={"max_steps": 10000},
    skip_collision_with_objs=[name_tray],
)

fp_place_tupperware_1_fridge = partial(
    InsideTask,
    target_object_name=name_fridge,
    source_object_name=name_tupperware_1,
    termination_config={"max_steps": 10000},
)

fp_place_tupperware_2_fridge = partial(
    InsideTask,
    target_object_name=name_fridge,
    source_object_name=name_tupperware_2,
    termination_config={"max_steps": 10000},
)

stages = [
    {"name": "move_to_cabinet", "kind": "task", "factory": fp_move_to_cabinet},
    {"name": "open_cabinet_door", "kind": "task", "factory": fp_open_cabinet},
    {"name": "pick_up_tupperware", "kind": "task", "factory": fp_grasp_tupperware_1},
    {"name": "close_cabinet_door", "kind": "task", "factory": fp_close_cabinet},
    {"name": "move_to_counter_top", "kind": "task", "factory": fp_move_to_counter_top},
    {"name": "place_tupperware", "kind": "task", "factory": fp_place_tupperware_1_countertop},
    {"name": "pick_up_apple_pie", "kind": "task", "factory": fp_grasp_apple_pie_1},
    {"name": "place_in_apple_pie", "kind": "task", "factory": fp_place_apple_pie_1},
    {"name": "move_to_cabinet", "kind": "task", "factory": fp_move_to_cabinet},
    {"name": "open_cabinet_door", "kind": "task", "factory": fp_open_cabinet},
    {"name": "pick_up_tupperware", "kind": "task", "factory": fp_grasp_tupperware_2},
    {"name": "close_cabinet_door", "kind": "task", "factory": fp_close_cabinet},
    {"name": "move_to_counter_top", "kind": "task", "factory": fp_move_to_counter_top},
    {"name": "place_tupperware", "kind": "task", "factory": fp_place_tupperware_2_countertop},
    {"name": "move_to_apple_pie", "kind": "task", "factory": fp_move_to_apple_pie_2},
    {"name": "pick_up_apple_pie", "kind": "task", "factory": fp_grasp_apple_pie_2},
    {"name": "place_in_apple_pie", "kind": "task", "factory": fp_place_apple_pie_2},
    {"name": "Move_to_fridge", "kind": "task", "factory": fp_move_to_fridge},
    {"name": "open_frideg_door", "kind": "task", "factory": fp_open_fridge},
    {"name": "move_to_tupperware", "kind": "task", "factory": fp_move_to_tupperware_2},
    {"name": "pick_up_tupperware", "kind": "task", "factory": fp_grasp_tupperware_1},
    {"name": "pick_up_tupperware", "kind": "task", "factory": fp_grasp_tupperware_2},
    {"name": "move_to_frideg", "kind": "task", "factory": fp_move_to_fridge},
    {"name": "place_in_tupperware_1", "kind": "task", "factory": fp_place_tupperware_1_fridge},
    {"name": "place_in_tupperware_2", "kind": "task", "factory": fp_place_tupperware_2_fridge},
    {"name": "close_fridge_door", "kind": "task", "factory": fp_close_fridge},
]
task_factory.update({"freeze_pies": stages})