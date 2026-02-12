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


@consideration("remaining_battery", transform=lambda v: 1 - v)
def remaining_robot_battery(context: MyContext):
    """Return the robot's remaining battery as a float from 0 to 1
    """
    return context.remaining_battery_percentage / 100

@consideration("distance_to_goal", transform=utilitai.negexp)
def distance_to_goal(context: MyContext):
    return math.sqrt(
      (robot_position[0] - goal_position[0])**2
      + (robot_position[1] - goal_position[1])**2
    )


@consideration("can_reach_goal_with_battery")
def can_reach_goal_with_battery(ctx: MyContext):
    # Underlying functions from considerations can be used
    # in other considerations, and still be registered in the trace
    distance = distance_to_goal.raw(ctx)
    movable_distance = ctx.remaining_battery_mAh * ctx.meters_per_mAh
    return 1 if movable_distance > distance else 0
    

def main():
    actions = [
        utilitai.Action(
            name="wait_for_battery_recharge",
            # We shouldn't worry as much about recharging if we're close
            # to the goal
            considerations=remaining_robot_battery * distance_to_goal
        ),
        utilitai.Action(
            name="move_to_goal",
            considerations=can_reach_goal_with_battery
        ),
    ]
    
    system = utilitai.utility_system(actions)
```
