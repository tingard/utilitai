import contextvars
import copy
import logging
from typing import Callable, Generic, TypeVar, overload

from . import curves

# @CR: No __all__ is defined. Consider adding one to control the public API surface
# and prevent leaking implementation details (e.g. curves, contextvars, copy) via
# `from utilitai import *`.

ContextType = TypeVar("ContextType")
GoalType = TypeVar("GoalType")
# @CR: Mutable default — every context that hasn't set this var shares the *same* dict
# object. The copy.copy() in Consideration.__call__ mitigates this today, but a direct
# `_considerations_on_tick.get()[key] = val` elsewhere would silently corrupt all
# contexts. Safer to use a factory via `default=None` and initialise on first access,
# or wrap in a helper that always returns a fresh dict.
_considerations_on_tick = contextvars.ContextVar("considerations_on_tick", default={})

_logger = logging.getLogger(__name__)


#####################################################################################################
# Context Utility
#####################################################################################################
class Consideration(Generic[ContextType]):
    def __init__(
        self,
        func: Callable[[ContextType], float],
        response_curve: Callable[[float], float] = curves.linear,
        name: str | None = None,
    ):
        self._func = func
        self._name = name or self._func.__name__
        self._response_curve = response_curve

    def __call__(self, context: ContextType):
        considerations = copy.copy(_considerations_on_tick.get())
        if (cached := considerations.get(self._name, None)) is not None:
            return cached
        raw = self._func(context)
        # @CR: Bug — the raw value is stored under `self._func.__name__` but the cache
        # lookup (above) checks `self._name`. When a custom `name` is provided and
        # differs from `func.__name__`, these are two different keys and the raw entry
        # is never actually consulted by the cache. Worse, when `name` IS None the
        # fallback `self._name = self._func.__name__` makes both keys identical, so
        # the raw value written here is immediately overwritten by the utility value
        # two lines below, losing the raw data entirely.
        considerations[self._func.__name__] = raw
        utility = self._response_curve(raw)
        considerations[self._name] = utility
        _considerations_on_tick.set(considerations)
        # @CR: f-string is eagerly formatted even when DEBUG is disabled. Prefer lazy
        # formatting: _logger.debug("Consideration(%s, raw=%s, utility=%s)", self._name, raw, utility)
        _logger.debug(f"Consideration({self._name}, raw={raw}, utility={utility})")
        return utility

    def raw(self, context: ContextType) -> float:
        return self._func(context)

    # @CR: No __rmul__ is defined, so `2.0 * some_consideration` raises TypeError
    # while `some_consideration * 2.0` works. Consider adding __rmul__ for symmetry.
    def __mul__(
        self,
        # @CR: The type hint accepts `float` but `isinstance(other, float)` does NOT
        # match `int` in Python. Passing an int (e.g. `consideration * 2`) falls
        # through to the else branch, which calls `other.__name__` on an int and
        # crashes with AttributeError. Use `isinstance(other, (int, float))` or
        # `numbers.Real`, and update the type hint to `int | float`.
        # The same issue applies to `min` and `max` below.
        other: Callable[[ContextType], float] | float,
    ) -> "Consideration[ContextType]":
        if isinstance(other, Consideration):
            name = f"mul({self._name}, {other._name})"
        elif isinstance(other, float):
            name = f"mul({self._name}, {other})"
            return Consideration(
                func=lambda ctx: self(ctx) * other,
                name=name,
            )
        else:
            # @CR: Accessing `other.__name__` will AttributeError on lambdas-without-
            # meaningful-names or arbitrary callables (e.g. functools.partial, bound
            # methods). Consider a `getattr(other, '__name__', repr(other))` fallback.
            name = f"mul({self._name}, {other.__name__})"
        return Consideration(
            func=lambda ctx: self(ctx) * other(ctx),
            name=name,
        )

    # @CR: The bodies of __mul__, min, and max are structurally identical aside from
    # the operator. Consider extracting a private helper like
    # `_binary_op(self, other, op, op_name)` to reduce duplication.
    def min(
        self, other: Callable[[ContextType], float] | float
    ) -> "Consideration[ContextType]":
        if isinstance(other, Consideration):
            name = f"min({self._name}, {other._name})"
        elif isinstance(other, float):
            name = f"min({self._name}, {other})"
            return Consideration(
                func=lambda ctx: min(self(ctx), other),
                name=name,
            )
        else:
            name = f"min({self._name}, {other.__name__})"
        return Consideration(
            func=lambda ctx: min(self(ctx), other(ctx)),
            name=name,
        )

    def max(
        self, other: Callable[[ContextType], float] | float
    ) -> "Consideration[ContextType]":
        if isinstance(other, Consideration):
            name = f"max({self._name}, {other._name})"
        elif isinstance(other, float):
            name = f"max({self._name}, {other})"
            return Consideration(
                func=lambda ctx: max(self(ctx), other),
                name=name,
            )
        else:
            name = f"max({self._name}, {other.__name__})"
        return Consideration(
            func=lambda ctx: max(self(ctx), other(ctx)),
            name=name,
        )


@overload
def consideration(  # pyrefly: ignore
    func: None = None,
    *,
    response_curve: Callable[[float], float],
    name: str | None = None,
) -> Callable[[Callable[[ContextType], float]], Consideration[ContextType]]:
    raise NotImplementedError()


@overload
def consideration(
    func: Callable[[ContextType], float],
    *,
    response_curve: Callable[[float], float],
    name: str | None = None,
) -> Consideration[ContextType]:
    raise NotImplementedError()


# @CR: The overloads declare `response_curve` and `name` as keyword-only (after `*`),
# but the implementation accepts them as positional. This means
# `consideration(my_func, curves.quadratic)` would type-check against the overloads
# as an error but succeed at runtime — a confusing mismatch. Add `*` here too.
# Also, the implementation is missing a return type annotation.
def consideration(
    func: Callable[[ContextType], float] | None = None,
    response_curve: Callable[[float], float] = curves.linear,
    name: str | None = None,
):
    if func is None:

        def _inner(func: Callable[[ContextType], float]):
            return Consideration(
                func=func,
                response_curve=response_curve,
                name=name,
            )

        return _inner
    return Consideration(
        func=func,
        response_curve=response_curve,
        name=name,
    )


#####################################################################################################
# Utility System
#####################################################################################################
def utility_system(
    goals: dict[GoalType, Callable[[ContextType], float] | Consideration],
    # TODO logging configuration (rerun, rich, custom, ...)
) -> Callable[[ContextType], GoalType | None]:
    """Create a utility system that selects the best goal for a given context.

    This function accepts a mapping of goals to scoring functions (or
    ``Consideration`` instances) and returns a callable that, when invoked with a
    context object, evaluates every goal's utility and returns the one with the
    highest score.

    Parameters
    ----------
    goals : dict[GoalType, Callable[[ContextType], float] | Consideration]
        A dictionary mapping goal identifiers to callables (or
        :class:`Consideration` instances) that accept a context and return a
        float representing the utility of that goal.

    Returns
    -------
    Callable[[ContextType], GoalType | None]
        A function that takes a context object and returns the goal with
        the highest utility score, or ``None`` if *goals* is empty.
    """

    def tick(context: ContextType) -> GoalType | None:
        """Determine the best goal to take, given some context."""
        ctx = contextvars.copy_context()

        def _compute_within_context():
            best_utility = -float("inf")
            best_goal: GoalType | None = None
            # @CR: `goal_response_curve` is a misleading name — this variable holds a
            # scoring function or Consideration, not a response curve. Something like
            # `score_fn` or `scorer` would be clearer.
            for goal_id, goal_response_curve in goals.items():
                goal_utility = goal_response_curve(context)
                if goal_utility > best_utility:
                    best_goal = goal_id
                    best_utility = goal_utility
            return best_goal

        best_goal = ctx.run(_compute_within_context)
        _logger.debug(
            "Computed considerations",
            extra={
                "considerations": ctx[_considerations_on_tick],
                "best_goal": best_goal,
            },
        )
        return best_goal

    return tick
