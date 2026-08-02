"""A minimal(ish) implementation of Utility AI.

Register the things an agent could do on a :class:`ToConsider` registry, then
ask it to :meth:`~ToConsider.consider` some context to find out which of them
is currently the most appealing.
"""

from __future__ import annotations

from dataclasses import dataclass
import graphlib
import inspect
import logging
import math
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    NamedTuple,
    TypeVar,
    overload,
)

from . import curves

__all__ = ["ScoreFunction", "ToConsider", "curves"]

ContextType = TypeVar("ContextType", contravariant=True)

"""Scores how appealing an option is, given some context.

Scores are compared against each other, so any float will do - but keeping
them normalised (usually to ``[0, 1]``, using the helpers in
:mod:`utilitai.curves`) makes them far easier to reason about.
"""
ScoreFunction = Callable[..., float | None]


_logger = logging.getLogger(__name__)


def _log_nan(name: str, score: float):
    if math.isnan(score):
        _logger.warning("Option %s returned NaN", name)
        return None
    return score


@dataclass
class _DAGNode[ContextType]:
    f: ScoreFunction
    typ: Literal["option", "consideration", "requirement"]
    considerations: tuple[str, ...]
    requirements: tuple[str, ...]
    priority: int = 0


class ScoreWithDeps(NamedTuple):
    score: float
    deps: dict[str, float]


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
    >>> @things.option("eat food")
    ... def eat_food(ctx: Context) -> float:
    ...     return curves.clamped(curves.exponential(ctx.hunger / MAX_HUNGER))
    >>> things.constant_option("do nothing", 0.1)
    >>> things.consider(Context(hunger=7))
    'eat food'
    """

    def __init__(self) -> None:
        self._options: set[str] = set()
        self._nodes: dict[str, _DAGNode[ContextType]] = {}

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

    def __add_node(
        self,
        typ: Literal["option", "consideration", "requirement"],
        name_or_func: str | ScoreFunction,
        /,
        priority: int = 0,
    ) -> ScoreFunction | Callable[[ScoreFunction], ScoreFunction]:
        if callable(name_or_func):
            return self._register(
                _infer_name(name_or_func), name_or_func, typ, priority=priority
            )
        if not isinstance(name_or_func, str):
            raise TypeError(
                "Expected a name or a scoring function, got "
                f"{type(name_or_func).__name__}"
            )
        name = name_or_func

        def _decorate(func: ScoreFunction) -> ScoreFunction:
            return self._register(name, func, typ, priority=priority)

        return _decorate

    @overload
    def consideration(
        self, name_or_func: str, /
    ) -> Callable[[ScoreFunction], ScoreFunction]:
        raise NotImplementedError()

    @overload
    def consideration(self, name_or_func: ScoreFunction, /) -> ScoreFunction:
        raise NotImplementedError()

    def consideration(
        self, name_or_func: str | ScoreFunction, /
    ) -> ScoreFunction | Callable[[ScoreFunction], ScoreFunction]:
        """Add a consideration which one or many options,other considerations,
        or requirments may depend on.
        """
        return self.__add_node("consideration", name_or_func)

    @overload
    def option(
        self, name_or_func: str, /, priority: int = 0
    ) -> Callable[[ScoreFunction], ScoreFunction]:
        raise NotImplementedError()

    @overload
    def option(
        self, name_or_func: ScoreFunction, /, priority: int = 0
    ) -> ScoreFunction:
        raise NotImplementedError()

    def option(
        self, name_or_func: str | ScoreFunction, /, priority: int = 0
    ) -> ScoreFunction | Callable[[ScoreFunction], ScoreFunction]:
        """Add an option to consider.

        Can be used either as a bare decorator, in which case the option takes
        the name of the decorated function, or called with a name to use
        instead::

            @things.option
            def eat_food(ctx: Context) -> float: ...

            @things.option("go to the shops")
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
        return self.__add_node("option", name_or_func, priority=priority)

    def constant_option(self, name: str, value: float, priority: int = 0) -> None:
        """Add an option which always scores *value*.

        Useful as a baseline: a constant option wins whenever nothing else
        scores above it.
        """
        if not isinstance(name, str):
            raise ValueError("Name must be a string")
        value = float(value)

        def _inner(ctx: ContextType):
            return value

        self._register(name, _inner, "option", priority=priority)

    def score(self, context: ContextType) -> dict[str, ScoreWithDeps]:
        """Score every option against *context*, keyed by option name.

        Options which return NaN or None (or have dependencies which do)
        will not be returned.
        """
        order = self._get_dag_compute_order()
        cache: dict[str, float | None] = {}
        out: dict[str, ScoreWithDeps] = {}
        # Compute nodes in order to correctly populate the dependency tree
        for node_name in order:
            node = self._nodes[node_name]
            kw: dict[str, float] = {}
            considerations_met = True
            for c in node.considerations:
                if (cached_value := cache[c]) is None:
                    _logger.debug(
                        "%S %s requirement %s not met", node.typ, node_name, c
                    )
                    considerations_met = False
                    # Could break here for efficiency, but we'd lose debug information
                    continue
                kw[c] = cached_value
            if not considerations_met:
                cache[node_name] = None
                continue
            score = node.f(context, **kw)
            # A None is this node intentionally backing out
            if score is None:
                cache[node_name] = None
                continue
            # A NaN is likely an error in the function - be loud
            if math.isnan(score):
                _logger.warning("%S %s returned NaN", node.typ, node_name)
                cache[node_name] = None
                continue
            # Cache the computed value for downstream
            cache[node_name] = score
            if node_name in self._options:
                out[node_name] = ScoreWithDeps(score, kw)
        return out

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
        best = max(scores, key=lambda k: scores[k][0])
        _logger.debug(
            "Considered %s options, chose %r",
            scores,
            best,
        )
        return best

    def consider_from_scores(self, scores: dict[str, float]) -> str:
        """Utility function allowing re-use of a scoring dict. This exists to avoid
        re-computing scores while allowing external access to the scores dict.
        """
        if len(scores) == 0:
            raise ValueError("Nothing to consider - no options have been added")

        try:
            # Sort nodes from highest priority to lowest so that the max takes the higher
            # priority of matching scores.
            best = max(sorted(scores.keys()), key=lambda k: -self._nodes[k].priority)
        except KeyError as e:
            raise KeyError(
                "Unknown node - required to ensure priority is respected!"
            ) from e
        _logger.debug(
            "Considered %s options, chose %r",
            scores,
            best,
        )
        return best

    def _register(
        self,
        name: str,
        func: ScoreFunction,
        typ: Literal["option", "consideration", "requirement"],
        priority: int = 0,
    ) -> ScoreFunction:
        if name in self._nodes:
            raise ValueError(f"A function named {name!r} has already been added")
        if not isinstance(name, str):
            raise ValueError("Name must be a string.")
        sig = inspect.signature(func)
        considerations = []
        requirements = []
        # Zeroth arg must be ctx
        for arg in list(sig.parameters.values())[1:]:
            if (v := self._nodes.get(arg.name, None)) is None:
                raise ValueError(f'Unknown dependency "{arg.name}"')
            if not issubclass(arg.annotation, (int, float, inspect._empty)):
                raise ValueError(
                    f'Expected type annotation as a float or int, got "{arg.annotation}"'
                )
            match v.typ:
                case "consideration":
                    considerations.append(arg.name)
                case "requirement":
                    requirements.append(arg.name)
                case "option":
                    raise ValueError(
                        f"A {type} cannot depend on the output of an option - only of considerations or requirements."
                    )
                case _:
                    raise ValueError(f"Unknown type {v.typ}")
        self._nodes[name] = _DAGNode(
            func,
            typ=typ,
            considerations=tuple(considerations),
            requirements=tuple(requirements),
            priority=priority,
        )
        # Check that by adding this we do not create cycles, and revert if so.
        try:
            self._get_dag_compute_order()
        except RuntimeError:
            del self._nodes[name]
            raise
        if typ == "option":
            self._options.add(name)
        return func

    def _get_dag_compute_order(self):
        ts = graphlib.TopologicalSorter(
            {
                name: {*node.considerations, *node.requirements}
                for name, node in self._nodes.items()
            }
        )
        return tuple(ts.static_order())


def _infer_name(obj: Any) -> str:
    name = getattr(obj, "__name__", None)
    if not isinstance(name, str) or name == "<lambda>":
        raise TypeError(
            f"Could not infer a name for {obj!r} - pass one explicitly, "
            "e.g. @things.option('my option')"
        )
    return name
