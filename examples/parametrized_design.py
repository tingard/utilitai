from dataclasses import dataclass

from utilitai import ToConsider


@dataclass
class Context:
    hunger: int = 0
    money: int = 0
    food: int = 0
    target_bar: str = 'x'

things = ToConsider[Context]()


@things.parameter
def foo(ctx: Context):
    yield 'a'
    yield 'b'

@things.parameter
def bar(ctx: Context):
    yield 'x'
    yield 'y'

@things.consideration
def baz(ctx: Context, foo: str, bar: str):
    match (foo, bar):
        case ('a', 'x'):
            return 0.8
        case ('b', 'x'):
            return 0.0
        case ('b', 'y'):
            return 0.2
        case _: # ignore (a, y)
            return None

@things.consideration
def bop(ctx: Context, bar: str):
    # This is contrived - in reality we would use
    # the context in bar to only yield the target
    if bar == ctx.target_bar:
        return 1.0
    return None

@things.option
def a(ctx: Context, bop: float):
    return max(0.1, bop)

@things.option
def b(ctx: Context, baz: float, bar: str):
    if bar == 'x':
        return baz
    return None

@things.option
def c(ctx: Context):
    return 0.1

# Add a new top_k kwarg to limit the number of options that are returned
# None means all?
scores = things.score(Context())
for score in scores:
    print(score)
# Return is scores with parameters, sorted from best to worst
assert scores == [
    ('a', {"bar": "x"}, 1.0),
    ('b', {"foo": "a", "bar": "x"}, 0.9),
    ('c', {}, 0.1),
    ('b', {"foo": "b", "bar": "x"}, 0.0),
]
