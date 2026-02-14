# Welcome to utilitai!

This package is a minimal(ish) implementation of [Utility AI](https://en.wikipedia.org/wiki/Utility_system) in Python. It is designed to compliment [btreeny](https://github.com/tingard/btreeny), my Behaviour Trees implementation!

There actually aren't many options for utility AI in Python, which surprises me! In general for _serious_ systems you should be looking at C++ (or, maybe, Rust 🦀), but hopefully this library will help on the prototyping journey!

```python
import math
import utilitai

@dataclass
class MyContext:
    remaining_battery_mAh: float
    meters_per_mAh: float
    remaining_battery_percentage: int # int from 0 to 100
    robot_position: tuple[float, float]
    goal_position: tuple[float, float]

# Considerations take some computed value and map it to a normalized score usable
# when computing the utility of an action
@consideration(response_curve=lambda v: 1 - v)
def remaining_robot_battery(ctx: MyContext):
    """Return the robot's remaining battery as a float from 0 to 1
    """
    return ctx.remaining_battery_percentage / 100

@consideration(response_curve=partial(utilitai.negexp, scale=0.1))
def distance_to_goal(ctx: MyContext):
    return math.sqrt(
      (ctx.robot_position[0] - ctx.goal_position[0])**2
      + (ctx.robot_position[1] - ctx.goal_position[1])**2
    )

@consideration
def is_at_goal(ctx: MyContext):
    return 1.0 if distance_to_goal.raw(cts) < 1E-2 else 0.0

@consideration
def can_reach_goal_with_battery(ctx: MyContext):
    # Underlying functions from considerations can be used
    # in other considerations, and still be registered in the trace
    distance = distance_to_goal.raw(ctx)
    movable_distance = ctx.remaining_battery_mAh * ctx.meters_per_mAh
    return 1.0 if movable_distance > distance else 0.0

@consideration
def has_valid_path(ctx: MyContext):
    return 1.0

def main():
    actions = {
        # If we have no good choices, do nothing
        "do_nothing": utilitai.eps,
        # As our battery gets lower, our desire to stop and recharge
        # gets higher
        "recharge": remaining_robot_battery,
        # Make progress towards our goal location
        "move_to_goal": has_valid_path,
        "replan_path": lambda ctx: utilitai.curves.inverse_linear(has_valid_path(ctx)),
        # Dancing before we make it to the goal would be silly!
        "do_a_dance": is_at_goal,
    }
    
    planner = utilitai.utility_system(actions)
    
    while True:
        # Perception
        context = update_context_from_sensors(context)
        # Planning (ish)
        best_action = planner(context)
        # Control - maybe using btreeny!
        control_robot_from_action(best_action, context)
```
