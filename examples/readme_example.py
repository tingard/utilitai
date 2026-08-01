"""The example from the README - a robot deciding what to do next."""

import math
from dataclasses import dataclass

from utilitai import ToConsider, curves


@dataclass
class RobotState:
    remaining_battery_mAh: float
    meters_per_mAh: float
    remaining_battery_percentage: int  # int from 0 to 100
    robot_position: tuple[float, float]
    goal_position: tuple[float, float]


# Scoring functions are plain Python, so shared calculations are just
# functions - use them wherever they're useful
def distance_to_goal(state: RobotState) -> float:
    return math.sqrt(
        (state.robot_position[0] - state.goal_position[0]) ** 2
        + (state.robot_position[1] - state.goal_position[1]) ** 2
    )


def is_at_goal(state: RobotState) -> bool:
    return distance_to_goal(state) < 1e-2


def can_reach_goal_with_battery(state: RobotState) -> bool:
    movable_distance = state.remaining_battery_mAh * state.meters_per_mAh
    return movable_distance > distance_to_goal(state)


def has_valid_path(state: RobotState) -> bool:
    return True


# Everything the robot could choose to do, scored against the robot's state
actions: ToConsider[RobotState] = ToConsider()

# If we have no good choices, do nothing
actions.add_constant("do nothing", 1e-6)


@actions.add("recharge")
def recharge(state: RobotState) -> float:
    # As our battery gets lower, our desire to stop and recharge gets higher
    battery_used = 1 - state.remaining_battery_percentage / 100
    return curves.inverse_quadratic(battery_used)


@actions.add("move to goal")
def move_to_goal(state: RobotState) -> float:
    if is_at_goal(state) or not has_valid_path(state):
        return 0.0
    return 0.9 if can_reach_goal_with_battery(state) else 0.1


@actions.add("replan path")
def replan_path(state: RobotState) -> float:
    return 0.0 if has_valid_path(state) else 1.0


@actions.add("do a dance")
def do_a_dance(state: RobotState) -> float:
    # Dancing before we make it to the goal would be silly!
    return 1.0 if is_at_goal(state) else 0.0


def update_state_from_sensors(state: RobotState) -> RobotState:
    raise NotImplementedError()


def control_robot_from_action(action: str, state: RobotState):
    raise NotImplementedError()


def main():
    state = RobotState(
        remaining_battery_mAh=3000,
        meters_per_mAh=1.234,
        remaining_battery_percentage=100,
        robot_position=(0, 0),
        goal_position=(500, 0),
    )
    while True:
        # Perception
        state = update_state_from_sensors(state)
        # Planning (ish)
        best_action = actions.consider(state)
        # Control - maybe using btreeny!
        control_robot_from_action(best_action, state)


if __name__ == "__main__":
    main()
