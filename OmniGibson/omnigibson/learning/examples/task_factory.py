from functools import partial

from omnigibson.tasks.custom_base_navigation_task import BaseNavigationTask
from omnigibson.tasks.custom_grasp_task import RobustGraspTask
from omnigibson.tasks.custom_open_close_task import SufficientlyClosedTask, SufficientlyOpenTask
from omnigibson.tasks.custom_point_reaching_task import MoveEEToObjectTask
from omnigibson.tasks.custom_predicate_task import OnTask
from omnigibson.tasks.custom_predicate_task import OnTopStableTask, NextToTask, OnTopTask

max_steps = 5000
task_factory = {}


def get_sub_tasks(task_name: str) -> list[dict[str, ...]]:
    return task_factory[task_name]


# -----turning_on_radio-------
name_radio = "radio_89"
name_coffe_table = "coffee_table_koagbh_0"

move_to_radio = partial(
    BaseNavigationTask,
    target_object_name=name_radio,
    goal_tolerance=1,
    termination_config={"max_steps": 5000},
    include_obs=False,
)

grasp_radio = partial(
    RobustGraspTask,
    obj_name=name_radio,
    termination_config={"max_steps": 5000},
    include_obs=False,
)

radio_on = partial(
    OnTask,
    target_object_name=name_radio,
    include_obs=False,
)

stages = [
    {"name": "move_to_radio", "factory": move_to_radio},
    {"name": "pick_radio", "factory": grasp_radio},
    {"name": "radio_on", "factory": radio_on},
    # {"name": "place_radio", "kind": "task", "factory": place_radio},
]
task_factory.update({"turning_on_radio": stages})

# ---------cook_bacon----------------
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

move_to_fridge = partial(
    BaseNavigationTask,
    target_object_name=name_fridge,
    goal_tolerance=1.3,
    termination_config={"max_steps": 10000},
    include_obs=False,
)

open_fridge = partial(
    SufficientlyOpenTask,
    target_object_name=name_fridge,
    allowed_deg=90,
    termination_config={"max_steps": 10000},
    include_obs=False,
)

close_fridge = partial(
    SufficientlyClosedTask,
    target_object_name=name_fridge,
    allowed_deg=0,
    termination_config={"max_steps": 10000},
    include_obs=False,
)

move_to_counter_top = partial(
    BaseNavigationTask,
    target_object_name=name_countertop,
    goal_tolerance=1,
    termination_config={"max_steps": 10000},
    include_obs=False,
    skip_collision_with_objs=[name_tray, name_bacon_1, name_bacon_2, name_bacon_3, name_bacon_4, name_bacon_5,
                              name_bacon_6],
)

place_next_to_burner = partial(
    NextToTask,
    target_object_name=name_tray,
    source_object_name=name_burner,
    desired_value=True,
    # reward_config=reward_config,
    termination_config={"max_steps": 10000},
    include_obs=False,
)

move_to_frying_pan = partial(
    MoveEEToObjectTask,
    target_object_name=name_pan,
    goal_tolerance=0.05,
    termination_config={"max_steps": 10000},
    include_obs=False,
    skip_collision_with_objs=[name_pan],
)

move_to_tray = partial(
    MoveEEToObjectTask,
    target_object_name=name_tray,
    goal_tolerance=0.05,
    termination_config={"max_steps": 10000},
    include_obs=False,
    skip_collision_with_objs=[name_tray],
)

place_frying_pan = partial(
    OnTopStableTask,
    target_object_name=name_burner,
    source_object_name=name_pan,
    termination_config={"max_steps": 10000},
    include_obs=False,
)

burner_on = partial(
    OnTask,
    target_object_name=name_burner,
    termination_config={"max_steps": 10000},
    include_obs=False,
)

pour_tray = partial(
    OnTopTask,
    target_object_name=name_bacon_3,
    source_object_name=name_pan,
    desired_value=True,
    termination_config={"max_steps": 10000},
    include_obs=False,
)

grasp_tray = partial(
    RobustGraspTask,
    obj_name=name_tray,
    termination_config={"max_steps": 10000},
    include_obs=False,
)

grasp_pan = partial(
    RobustGraspTask,
    obj_name=name_pan,
    termination_config={"max_steps": 10000},
    include_obs=False,
)

stages = [
    {"name": "move_to_fridge", "factory": move_to_fridge},
    {"name": "open_fridge", "factory": open_fridge},
    {"name": "pick_tray", "factory": grasp_tray},
    {"name": "close_fridge", "factory": close_fridge},
    {"name": "move_to_counter_top", "factory": move_to_counter_top},
    {"name": "place_on_next_to_burner1", "factory": place_next_to_burner},
    {"name": "Move_to_frying_pan", "factory": move_to_frying_pan},
    {"name": "pick_up_frying_pan", "factory": grasp_pan},
    {"name": "place_frying_pan", "factory": place_frying_pan},
    {"name": "move_to_tray", "factory": move_to_tray},
    {"name": "pick_up_tray", "factory": grasp_tray},
    {"name": "pour_tray", "factory": pour_tray},
    {"name": "place_on_next_to_burner2", "factory": place_next_to_burner},
    {"name": "burner_on_switch", "factory": burner_on},
]

task_factory.update({"cook_bacon": stages})
