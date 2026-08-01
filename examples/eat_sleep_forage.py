"""A hunter-gatherer who has to keep themselves fed.

Run with ``uv run python examples/eat_sleep_forage.py``. Set the log level to
DEBUG to see the score every option received on each tick.
"""

import logging
import random
from dataclasses import dataclass, field

from utilitai import ToConsider, curves


@dataclass
class FoodSupplies:
    fruit: int = 0
    nuts: int = 0


@dataclass
class HunterGathererState:
    energy: float = 100
    food_supplies: FoodSupplies = field(default_factory=FoodSupplies)


def hunger(state: HunterGathererState) -> float:
    """How hungry we are, as a float from 0 (stuffed) to 1 (starving)."""
    return curves.inverse_linear(state.energy / 100)


def total_food(state: HunterGathererState) -> int:
    return state.food_supplies.fruit + state.food_supplies.nuts


# Everything our hunter-gatherer could choose to do
goals: ToConsider[HunterGathererState] = ToConsider()


# Sleeping is a constant baseline - we do it whenever nothing else appeals.
# Adding it first means it also wins any ties.
goals.add_constant("sleep", 0.1)


@goals.add("eat")
def eat(state: HunterGathererState) -> float:
    # The hungrier we get the more we want to eat - but only if we have food
    return curves.smoothstep(hunger(state)) * curves.is_gt_zero(total_food(state))


@goals.add("forage for fruit")
def forage_for_fruit(state: HunterGathererState) -> float:
    # The less fruit we have in store, the more we want to find some.
    # Fruit is worth more than nuts, so we're keener to go looking for it
    return 0.6 * curves.inverse_linear(min(state.food_supplies.fruit / 3, 1.0))


@goals.add("forage for nuts")
def forage_for_nuts(state: HunterGathererState) -> float:
    return 0.5 * curves.inverse_linear(min(state.food_supplies.nuts / 5, 1.0))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("hunter_gatherer")

    state = HunterGathererState()
    for tick in range(1000):
        # Our energy drains every tick
        state.energy -= 1
        if state.energy <= 0:
            raise RuntimeError(f"You died on tick {tick}.")

        goal = goals.consider(state)
        logger.info(
            "[%4d] energy=%3.0f fruit=%d nuts=%d -> %s",
            tick,
            state.energy,
            state.food_supplies.fruit,
            state.food_supplies.nuts,
            goal,
        )

        match goal:
            # Fruit is harder to find than nuts
            case "forage for fruit":
                if random.random() > 0.78:
                    logger.info("Found fruit!")
                    state.food_supplies.fruit += 1
            case "forage for nuts":
                if random.random() > 0.2:
                    logger.info("Found nuts!")
                    state.food_supplies.nuts += 1
            case "eat":
                # One fruit is worth nearly three nuts!
                if state.food_supplies.fruit > 0:
                    state.food_supplies.fruit -= 1
                    state.energy = min(100, state.energy + 8)
                elif state.food_supplies.nuts > 0:
                    state.food_supplies.nuts -= 1
                    state.energy = min(100, state.energy + 3)
            case "sleep":
                pass

    logger.info("Survived 1000 ticks with %.0f energy to spare!", state.energy)


if __name__ == "__main__":
    main()
