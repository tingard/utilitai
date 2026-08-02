"""A minimal(ish) implementation of Utility AI.

Register the things an agent could do on a :class:`ToConsider` registry, then
ask it to :meth:`~ToConsider.consider` some context to find out which of them
is currently the most appealing.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Generic, Iterator, TypeVar, overload

from . import curves

__all__ = ["ScoreFunction", "ToConsider", "curves"]

ContextType = TypeVar("ContextType")

ScoreFunction = Callable[[ContextType], float]
"""Scores how appealing an option is, given some context.

Scores are compared against each other, so any float will do - but keeping
them normalised (usually to ``[0, 1]``, using the helpers in
:mod:`utilitai.curves`) makes them far easier to reason about.
"""

_logger = logging.getLogger(__name__)


def _log_nan(name: str, score: float):
    if math.isnan(score):
        _logger.warning("Option %s returned NaN", name)
        return None
    return score


class ToConsider(Generic[ContextType]):
    """A registry of the options an agent can choose between.

    Options are scoring functions which map a context to a float. The option
    with the highest score is the one that gets chosen.

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Context:
    ...     hunger: int
    >>> things: ToConsider[Context] = ToConsider()
    >>> MAX_HUNGER = 10
    >>> @things.add("eat food")
    ... def eat_food(ctx: Context) -> float:
    ...     return curves.clamped(curves.exponential(ctx.hunger / MAX_HUNGER))
    >>> things.add_constant("do nothing", 0.1)
    >>> things.consider(Context(hunger=7))
    'eat food'
    """

    def __init__(self) -> None:
        self._options: dict[str, ScoreFunction[ContextType]] = {}

    def __len__(self) -> int:
        return len(self._options)

    def __iter__(self) -> Iterator[str]:
        return iter(self._options)

    def __contains__(self, name: object) -> bool:
        return name in self._options

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self._options))})"

    @property
    def names(self) -> tuple[str, ...]:
        """The name of every option, in the order they were added."""
        return tuple(self._options)

    @overload
    def add(
        self, name_or_func: str, /
    ) -> Callable[[ScoreFunction[ContextType]], ScoreFunction[ContextType]]:
        raise NotImplementedError()

    @overload
    def add(
        self, name_or_func: ScoreFunction[ContextType], /
    ) -> ScoreFunction[ContextType]:
        raise NotImplementedError()

    def add(
        self, name_or_func: str | ScoreFunction[ContextType], /
    ) -> (
        ScoreFunction[ContextType]
        | Callable[[ScoreFunction[ContextType]], ScoreFunction[ContextType]]
    ):
        """Add an option to consider.

        Can be used either as a bare decorator, in which case the option takes
        the name of the decorated function, or called with a name to use
        instead::

            @things.add
            def eat_food(ctx: Context) -> float: ...

            @things.add("go to the shops")
            def go_to_the_shops(ctx: Context) -> float: ...

        The decorated function is returned unchanged, so it can still be
        called, tested, or reused by other scoring functions directly.

        Raises
        ------
        ValueError
            If an option with the same name has already been added.
        TypeError
            If *name_or_func* is neither a string nor a callable, or if a name
            cannot be inferred from the decorated function (lambdas, for
            example, must be given an explicit name).
        """
        if callable(name_or_func):
            return self._register(_infer_name(name_or_func), name_or_func)
        if not isinstance(name_or_func, str):
            raise TypeError(
                "Expected a name or a scoring function, got "
                f"{type(name_or_func).__name__}"
            )
        name = name_or_func

        def _decorate(func: ScoreFunction[ContextType]) -> ScoreFunction[ContextType]:
            return self._register(name, func)

        return _decorate

    def add_constant(self, name: str, value: float) -> None:
        """Add an option which always scores *value*.

        Useful as a baseline: a constant option wins whenever nothing else
        scores above it.
        """
        if not isinstance(name, str):
            raise ValueError("Name must be a string")
        value = float(value)
        self._register(name, lambda _context: value)

    def score(self, context: ContextType) -> dict[str, float]:
        """Score every option against *context*, keyed by option name.

        Options which return NaN or None will not be returned.
        """
        return {
            name: score
            for name, score in (
                (name, _log_nan(name, func(context)))
                for name, func in self._options.items()
            )
            if score is not None
        }

    def consider(self, context: ContextType) -> str:
        """Return the name of the highest scoring option for *context*.

        Ties are broken in favour of whichever option was added first.

        Raises
        ------
        ValueError
            If there is nothing to consider.
        """
        if not self._options:
            raise ValueError("Nothing to consider - no options have been added")
        scores = self.score(context)
        if len(scores) == 0:
            # Score functions may have returned NaN / None.
            raise ValueError("No good options - no valid options available.")
        # `max` returns the first maximal element, so earlier options win ties
        best = max(scores, key=scores.__getitem__)
        _logger.debug(
            "Considered %s options, chose %r",
            scores,
            best,
        )
        return best

    @staticmethod
    def consider_from_scores(scores: dict[str, float]) -> str:
        """Utility function allowing re-use of a scoring dict. This exists to avoid
        re-computing scores while allowing external access to the scores dict.
        """
        if len(scores) == 0:
            raise ValueError("Nothing to consider - no options have been added")
        best = max(scores, key=scores.__getitem__)
        _logger.debug(
            "Considered %s options, chose %r",
            scores,
            best,
        )
        return best

    def _register(
        self, name: str, func: ScoreFunction[ContextType]
    ) -> ScoreFunction[ContextType]:
        if name in self._options:
            raise ValueError(f"An option named {name!r} has already been added")
        if not isinstance(name, str):
            raise ValueError("Name must be a string.")
        self._options[name] = func
        return func

    def __getitem__(self, name: str) -> ScoreFunction[ContextType]:
        return self._options[name]


def _infer_name(func: Callable[..., float]) -> str:
    name = getattr(func, "__name__", None)
    if not isinstance(name, str) or name == "<lambda>":
        raise TypeError(
            f"Could not infer a name for {func!r} - pass one explicitly, "
            "e.g. @things.add('my option')"
        )
    return name
