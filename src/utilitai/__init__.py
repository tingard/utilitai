import contextvars
import copy
import logging
from typing import Callable, Generic, TypeVar, overload

from . import curves

ContextType = TypeVar("ContextType")
GoalType = TypeVar("GoalType")
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
        considerations[self._func.__name__] = raw
        utility = self._response_curve(raw)
        considerations[self._name] = utility
        _considerations_on_tick.set(considerations)
        _logger.debug(f"Consideration({self._name}, raw={raw}, utility={utility})")
        return utility

    def raw(self, context: ContextType) -> float:
        return self._func(context)

    def __mul__(
        self, other: Callable[[ContextType], float]
    ) -> "Consideration[ContextType]":
        if isinstance(other, Consideration):
            name = f"mul({self._name}, {other._name})"
        else:
            name = f"mul({self._name}, {other.__name__})"
        return Consideration(
            func=lambda ctx: self(ctx) * other(ctx),
            name=name,
        )

    def min(
        self, other: Callable[[ContextType], float]
    ) -> "Consideration[ContextType]":
        if isinstance(other, Consideration):
            name = f"min({self._name}, {other._name})"
        else:
            name = f"min({self._name}, {other.__name__})"
        return Consideration(
            func=lambda ctx: min(self(ctx), other(ctx)),
            name=name,
        )

    def max(
        self, other: Callable[[ContextType], float]
    ) -> "Consideration[ContextType]":
        if isinstance(other, Consideration):
            name = f"max({self._name}, {other._name})"
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
