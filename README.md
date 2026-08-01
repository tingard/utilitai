# Welcome to utilitai!

This package is a minimal(ish) implementation of [Utility AI](https://en.wikipedia.org/wiki/Utility_system) in Python. It is designed to compliment [btreeny](https://github.com/tingard/btreeny), my Behaviour Trees implementation!

There actually aren't many options for utility AI in Python, which surprises me! In general for _serious_ systems you should be looking at C++ (or, maybe, Rust 🦀), but hopefully this library will help on the prototyping journey!

## The idea

List the things your agent could do, score each of them against the current state of the world, and do whichever scores highest.

```python
from dataclasses import dataclass

from utilitai import ToConsider, curves

# In order to make decisions, we need context - this can be anything but
# a dataclass is a logical choice
@dataclass
class Context:
    hunger: int
    money: int
    food: int

# We create a registry object which is generic over the context
things: ToConsider[Context] = ToConsider()

# And then add options to consider
@things.add('go to the shops')
def go_to_the_shops(ctx: Context):
    return curves.exponential(ctx.hunger) * curves.is_gt_zero(ctx.money)

# Names can be inferred from function name
@things.add
def eat_food(ctx: Context):
    return curves.exponential(ctx.hunger) * curves.is_gt_zero(ctx.food)

# We can also add a constant baseline
things.add_constant('starve', 0.0)

current_context = Context(hunger=1, money=1, food=0)
action = things.consider(current_context)
assert action == 'go to the shops'
```

## Scoring functions

A scoring function takes your context and returns a float. That's the whole interface - no subclassing, no DSL, and the decorated function is returned unchanged so you can still call, test, or reuse it elsewhere.

Scores are only ever compared against each other, but keeping them in `[0, 1]` makes them much easier to reason about. The `utilitai.curves` module has the usual suspects for shaping a raw value into that range:

| curve | shape |
| --- | --- |
| `linear` / `inverse_linear` | identity / `1 - val` |
| `quadratic` / `inverse_quadratic` | accelerates / decelerates quickly |
| `logistic(val, midpoint, steepness)` | S-shaped, sharp transition around the midpoint |
| `exponential(val, base)` | strongly favours high inputs |
| `smoothstep` | ease-in / ease-out, clamped to `[0, 1]` |
| `is_gt_zero` / `is_le_zero` | step functions, handy as on/off multipliers |

Because scoring functions are plain Python, curves compose by calling them and combining the results - multiply to say "and", `max` to say "or", and multiply by a constant to weight an option against its rivals:

```python
@things.add('order a takeaway')
def order_a_takeaway(ctx: Context):
    # Weighted below the other options - it's a treat, not a necessity
    return 0.5 * curves.smoothstep(ctx.hunger) * curves.is_gt_zero(ctx.money)
```

## Choosing between options

- `things.consider(context)` returns the name of the highest scoring option. If two options tie, whichever was added first wins - so add any baselines first if you want them to break ties in their favour.
- `things.add_constant(name, value)` adds an option with a fixed score. This is the usual way to express "if nothing else appeals, do this".
- `things.score(context)` returns every option's score as a dict, which is useful in tests and when tuning curves.
- `consider` raises `ValueError` if nothing has been added, and `add` raises `ValueError` on a duplicate name.

Each call to `consider` logs the full set of scores at `DEBUG` level under the `utilitai` logger, so you can see why a decision was made:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

## A slightly bigger example

```python
actions: ToConsider[RobotState] = ToConsider()

# If we have no good choices, do nothing
actions.add_constant('do nothing', 1e-6)

@actions.add('recharge')
def recharge(state: RobotState) -> float:
    # As our battery gets lower, our desire to stop and recharge gets higher
    battery_used = 1 - state.remaining_battery_percentage / 100
    return curves.inverse_quadratic(battery_used)

@actions.add('move to goal')
def move_to_goal(state: RobotState) -> float:
    if is_at_goal(state) or not has_valid_path(state):
        return 0.0
    return 0.9 if can_reach_goal_with_battery(state) else 0.1

@actions.add('do a dance')
def do_a_dance(state: RobotState) -> float:
    # Dancing before we make it to the goal would be silly!
    return 1.0 if is_at_goal(state) else 0.0

while True:
    # Perception
    state = update_state_from_sensors(state)
    # Planning (ish)
    best_action = actions.consider(state)
    # Control - maybe using btreeny!
    control_robot_from_action(best_action, state)
```

The full version lives in [`examples/readme_example.py`](examples/readme_example.py), and [`examples/eat_sleep_forage.py`](examples/eat_sleep_forage.py) is a runnable simulation of a hunter-gatherer trying not to starve.
