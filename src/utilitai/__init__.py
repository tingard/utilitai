import contextvars
import logging
import sys
from typing import Callable, Generic, TypeVar, overload

ContextType = TypeVar("ContextType")
ActionType = TypeVar("ActionType")
_considerations_on_tick = contextvars.ContextVar("considerations_on_tick", default={})

_logger = logging.getLogger(__name__)


#####################################################################################################
# Response curves
#####################################################################################################
def eps(*a, **k):
    return sys.float_info.epsilon


def linear(val: float) -> float:
    return val


def inverse_linear(val: float) -> float:
    return 1 - val


# TODO: what other builtins should we offer?
# https://www.gameaipro.com/GameAIPro/GameAIPro_Chapter09_An_Introduction_to_Utility_Theory.pdf
# https://www.gameaipro.com/GameAIPro3/GameAIPro3_Chapter13_Choosing_Effective_Utility-Based_Considerations.pdf


#####################################################################################################
# Context Utility
#####################################################################################################
class Consideration(Generic[ContextType]):
    def __init__(
        self,
        func: Callable[[ContextType], float],
        response_curve: Callable[[float], float] = linear,
        name: str | None = None,
    ):
        self._func = func
        self._name = name or self._func.__name__
        self._response_curve = response_curve

    def __call__(self, context: ContextType):
        raw = self._func(context)
        considerations = _considerations_on_tick.get()
        considerations[self._func.__name__] = raw
        utility = self._response_curve(raw)
        considerations[self._name] = utility
        _considerations_on_tick.set(considerations)
        return utility

    def raw(self, context: ContextType) -> float:
        return self._func(context)

    def __mul__(
        self, other: Callable[[ContextType], float]
    ) -> "Consideration[ContextType]":
        if isinstance(other, Consideration):
            name = f"{self._name}x{other._name}"
        else:
            name = f"{self._name}x{other.__name__}"
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
    func: None,
    *,
    response_curve: Callable[[float], float] = linear,
    name: str | None = None,
) -> Callable[[Callable[[ContextType], float]], Consideration[ContextType]]:
    raise NotImplementedError()


@overload
def consideration(
    func: Callable[[ContextType], float],
    *,
    response_curve: Callable[[float], float] = linear,
    name: str | None = None,
) -> Consideration[ContextType]:
    raise NotImplementedError()


def consideration(
    func: Callable[[ContextType], float] | None,
    *,
    response_curve: Callable[[float], float] = linear,
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
    actions_dict: dict[ActionType, Callable[[ContextType], float] | Consideration],
    # TODO logging configuration (rerun, rich, custom, ...)
) -> Callable[[ContextType], ActionType | None]:
    """TODO"""

    def _inner(context: ContextType) -> ActionType | None:
        """Determine the best action to take, given some context.

        Parameters
        ----------
        TODO
        """
        ctx = contextvars.copy_context()

        def _compute_within_context():
            best_utility = -float("inf")
            best_action: ActionType | None = None
            for action_id, action_response_curve in actions_dict.items():
                action_utility = action_response_curve(context)
                if action_utility > best_utility:
                    best_action = action_id
                    best_utility = action_utility
            return best_action

        best_action = ctx.run(_compute_within_context)
        logging.debug(
            "Computed considerations",
            extra={
                "considerations": ctx[_considerations_on_tick],
                "best_action": best_action,
            },
        )
        return best_action

    return _inner
