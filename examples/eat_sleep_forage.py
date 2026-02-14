import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import utilitai


@dataclass
class FoodSupplies:
    fruit: int = 0
    nuts: int = 0


class TileState(str, Enum):
    EMPTY = "empty"
    FRUIT = "fruit"
    NUTS = "nuts"


@dataclass
class Map:
    known_tiles: dict[tuple[int, int], TileState] = field(default_factory=dict)

    def unexplored_neighbors(self, tile: tuple[int, int]) -> Iterable[tuple[int, int]]:
        for i in (-1, 1):
            for j in (-1, 1):
                key = (tile[0] + i, tile[1] + j)
                if key in self.known_tiles:
                    continue
                yield key


@dataclass
class HunterGathererState:
    energy: float = 100
    food_supplies: FoodSupplies = field(default_factory=FoodSupplies)
    map: Map = field(default_factory=Map)


@utilitai.consideration(response_curve=utilitai.curves.inverse_linear)
def remaining_energy(ctx: HunterGathererState):
    return ctx.energy / 100


@utilitai.consideration(response_curve=utilitai.curves.is_gt_zero)
def has_food(ctx: HunterGathererState):
    return max(ctx.food_supplies.fruit, ctx.food_supplies.nuts)


@utilitai.consideration(response_curve=utilitai.curves.is_le_zero)
def has_fruit(ctx: HunterGathererState):
    return ctx.food_supplies.fruit


@utilitai.consideration(response_curve=utilitai.curves.is_le_zero)
def has_nuts(ctx: HunterGathererState):
    return ctx.food_supplies.nuts


def main():
    logging.basicConfig(level=logging.DEBUG)
    goals = {
        "eat": has_food * remaining_energy,
        "sleep": utilitai.curves.eps,
        "forage_fruit": has_fruit,
        "forage_nuts": has_nuts,
    }
    system = utilitai.utility_system(goals)

    state = HunterGathererState()
    for _ in range(1000):
        # Our energy drains every tick.
        state.energy -= 1
        print(state)
        if state.energy <= 0:
            raise RuntimeError("You died.")
        action = system(state)
        print(f"Taking action {action}")
        match action:
            # Fruit is harder to find than nuts
            case "forage_fruit":
                if random.random() > 0.78:
                    print("Found fruit!")
                    state.food_supplies.fruit += 1
            case "forage_nuts":
                if random.random() > 0.2:
                    print("Found nuts!")
                    state.food_supplies.nuts += 1
            case "eat":
                # One fruit is worth five nuts!
                if state.food_supplies.fruit > 0:
                    state.food_supplies.fruit -= 1
                    state.energy = min(100, state.energy + 5)
                elif state.food_supplies.nuts > 0:
                    state.food_supplies.nuts -= 1
                    state.energy = min(100, state.energy + 2)
                else:
                    print("Nothing to eat!")
            case "sleep":
                pass

    pass


if __name__ == "__main__":
    main()
