from dataclasses import dataclass

from utilitai import ToConsider, curves


# In order to make decisions, we need context - this can be anything but
# a dataclass is a logical choice
@dataclass
class Context:
    hunger: int
    money: int
    food: int


MAX_HUNGER = 10

# We create a registry object which is generic over the context
things: ToConsider[Context] = ToConsider()


@things.consideration
def has_money(ctx: Context):
    # Note that by returning `None` we signal to `utilitai` that
    # any dependencies must also be None, and bypass any downstream
    # maths.
    # Feel free to return 0.0 if you don't want this.
    return None if ctx.money == 0 else 1.0


@things.consideration
def has_food(ctx: Context):
    # Note that by returning `None` we signal to `utilitai` that
    # any dependencies must also be None, and bypass any downstream
    # maths.
    # Feel free to return 0.0 if you don't want this.
    return None if ctx.food == 0 else 1.0


@things.consideration
def hunger_level(ctx: Context):
    return curves.logistic(ctx.hunger / MAX_HUNGER, midpoint=0.5)


# Note that now the utility of eat_food and go_to_the_shops will be
# identical - making it easy to spot undesirable ties.
@things.option('eat food', priority=1)
def eat_food(ctx: Context, has_food: float, hunger_level: float):
    return hunger_level


@things.option("go to the shops")
def go_to_the_shops(ctx: Context, has_money: float, hunger_level: float):
    return hunger_level


things.constant_option("do nothing", 0.2, priority=-1)

action = None
context = Context(hunger=0, money=2, food=0)
for _ in range(15):
    context.hunger += 1
    scores = things.score(context)
    action = things.consider_from_scores({k: v for k, (v, _) in scores.items()})
    print(f"Given {context}")
    for option, (score, deps) in scores.items():
        print(f"* I could {option} [{deps} -> {score}]")
    print(f"I will {action}")
    print()
    if action == "go to the shops":
        context.money -= 1
        context.food += 1
    if action == "eat food":
        context.food -= 1
        context.hunger = 0
