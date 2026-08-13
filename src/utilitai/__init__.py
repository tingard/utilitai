"""A minimal(ish) implementation of Utility AI.

Register the things an agent could do on a :class:`ToConsider` registry, then
ask it to :meth:`~ToConsider.consider` some context to find out which of them
is currently the most appealing.
"""

from __future__ import annotations

from functools import lru_cache
import graphlib
import inspect
import itertools
import logging
import math
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from multiprocessing import Value
from typing import (
    Any,
    Concatenate,
    Literal,
    NamedTuple,
    overload,
)

from . import curves

__all__ = ["ScoreFunction", "ScoreWithDeps", "ToConsider", "curves"]

type ScoreFunction[ContextType] = Callable[Concatenate[ContextType, ...], float | None]
"""Scores how appealing an option is, given some context.

Scores are compared against each other, so any float will do - but keeping
them normalised (usually to ``[0, 1]``, using the helpers in
:mod:`utilitai.curves`) makes them far easier to reason about.
"""

type ParamFunction[ContextType] = Callable[Concatenate[ContextType, ...], Iterable[str]]
"""Responsible for returning parameter options given a context.
"""


type _Scorer[C, **P, R] = Callable[Concatenate[C, P], R]
type _Parametrizer[C, **P, R] = Callable[Concatenate[C, P], R]

_logger = logging.getLogger(__name__)


@dataclass
class _DAGNode[ContextType]:
    f: ScoreFunction[ContextType]
    typ: Literal["option", "consideration"]
    considerations: tuple[str, ...]
    parameters: tuple[str, ...]
    priority: int = 0


class ScoreWithDeps(NamedTuple):
    score: float
    deps: dict[str, float]


class ToConsider[ContextType]:
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
        self._parameters: dict[str, ParamFunction[ContextType]] = {}
        self._sort_cache: None | tuple[str, ...] = None

    @property
    def options(self) -> frozenset[str]:
        return frozenset(self._options)

    @property
    def considerations(self) -> frozenset[str]:
        return frozenset(c for c, v in self._nodes.items() if v.typ == "consideration")

    @property
    def parameters(self) -> frozenset[str]:
        return frozenset(self._parameters)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self._nodes))})"

    @property
    def names(self) -> tuple[str, ...]:
        """The name of every option or consideration (in no particular order)."""
        return tuple(self._nodes)

    def __add_node[**P, R: float | None](
        self,
        typ: Literal["option", "consideration"],
        name_or_func: str | _Scorer[ContextType, P, R],
        /,
        priority: int = 0,
    ) -> (
        _Scorer[ContextType, P, R]
        | Callable[[_Scorer[ContextType, P, R]], _Scorer[ContextType, P, R]]
    ):
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

        def _decorate(
            func: _Scorer[ContextType, P, R],
        ) -> _Scorer[ContextType, P, R]:
            return self._register(name, func, typ, priority=priority)

        return _decorate

    @overload
    def consideration(
        self, name_or_func: str, /
    ) -> Callable[[ScoreFunction[ContextType]], ScoreFunction[ContextType]]:
        raise NotImplementedError()

    @overload
    def consideration(
        self, name_or_func: ScoreFunction[ContextType], /
    ) -> ScoreFunction[ContextType]:
        raise NotImplementedError()

    def consideration(
        self, name_or_func: str | ScoreFunction[ContextType], /
    ) -> (
        ScoreFunction[ContextType]
        | Callable[[ScoreFunction[ContextType]], ScoreFunction[ContextType]]
    ):
        """Add a consideration which one or many options or other
        considerations, may depend on.
        """
        return self.__add_node("consideration", name_or_func)

    @overload
    def option[**P, R: float | None](
        self, name_or_func: str, /, priority: int = 0
    ) -> Callable[[_Scorer[ContextType, P, R]], _Scorer[ContextType, P, R]]:
        raise NotImplementedError()

    @overload
    def option[**P, R: float | None](
        self, name_or_func: _Scorer[ContextType, P, R], /, priority: int = 0
    ) -> _Scorer[ContextType, P, R]:
        raise NotImplementedError()

    def option[**P, R: float | None](
        self,
        name_or_func: str | _Scorer[ContextType, P, R],
        /,
        priority: int = 0,
    ) -> (
        _Scorer[ContextType, P, R]
        | Callable[[_Scorer[ContextType, P, R]], _Scorer[ContextType, P, R]]
    ):
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
        value = float(value)
        if math.isnan(value):
            raise ValueError("Constant option value cannot be NaN.")
        if not isinstance(name, str):
            raise TypeError("Name must be a string")

        def _inner(ctx: ContextType):
            return value

        self._register(name, _inner, "option", priority=priority)

    @overload
    def parameter[**P, R: Iterable[str]](
        self, name_or_func: str, /,
    ) -> Callable[[_Parametrizer[ContextType, P, R]], _Parametrizer[ContextType, P, R]]:
        raise NotImplementedError()

    @overload
    def parameter[**P, R: Iterable[str]](
        self, name_or_func: _Parametrizer[ContextType, P, R], /,
    ) -> _Parametrizer[ContextType, P, R]:
        raise NotImplementedError()

    def parameter[**P, R: Iterable[str]](
        self,
        name_or_func: str | _Parametrizer[ContextType, P, R],
        /,
    ) -> (
        _Parametrizer[ContextType, P, R]
        | Callable[[_Parametrizer[ContextType, P, R]], _Parametrizer[ContextType, P, R]]
    ):
        """Add a parameter generator.

        Can be used either as a bare decorator, in which case the parameter takes
        the name of the decorated function, or called with a name to use
        instead::

            @things.parameter
            def a(ctx: Context) -> Iterable[str]: ...

            @things.parameter("b")
            def _(ctx: Context) -> Iterable[str]: ...

        The decorated function is returned unchanged, so it can still be
        called, tested, or reused by other scoring functions directly.

        Raises
        ------
        ValueError
            If an parameter, option or consideration with the same name has
            already been added.
        TypeError
            If *name_or_func* is neither a string nor a callable, or if a name
            cannot be inferred from the decorated function (lambdas, for
            example, must be given an explicit name).
        """
        if callable(name_or_func):
            if (name := _infer_name(name_or_func)) in self.names:
                raise ValueError(f"Name {name} is already in use.")
            self._parameters[_infer_name(name_or_func)] = name_or_func
            return name_or_func
        if not isinstance(name_or_func, str):
            raise TypeError(
                "Expected a name or a scoring function, got "
                f"{type(name_or_func).__name__}"
            )
        name = name_or_func

        def _decorate(
            func: _Scorer[ContextType, P, R],
        ) -> _Scorer[ContextType, P, R]:
            if name in self.names:
                raise ValueError(f"Name {name} is already in use.")
            self._parameters[name] = func
            return func

        return _decorate

    def score(self, context: ContextType) -> list[tuple[str, dict[str, Any], ScoreWithDeps]]:
        """Score every option against *context*, keyed by option name.

        Options which return NaN or None (or have dependencies which do)
        will not be returned.
        """
        order = self._get_dag_compute_order()
        out: list[tuple[str, dict[str, Any], ScoreWithDeps]] = []

        # Naive cartesian product over all parametrizations as a first pass
        for params in map(dict, itertools.product(*(
            [(p, v) for v in ps(context)]
            for p, ps in self._parameters.items()
        ))):
            # Cache from consideration name to its values for each parameter
            # combination
            cache: dict[str, float | None] = {}
            # Compute nodes in order to correctly populate the dependency tree
            for node_name in order:
                node = self._nodes[node_name]
                kw: dict[str, Any] = {
                    p: v
                    for p, v in params.items()
                    if p in node.parameters
                }
                # We need to understand all of the possible parametrizations of this node
                # this is driven by
                # * the node's own parameterization
                # * the active parametrizations of its dependency tree
                considerations_met = True
                for c in node.considerations:
                    if (cached_value := cache[c]) is None:
                        _logger.debug(
                            "%s %s consideration %s not met", node.typ, node_name, c
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
                score = float(score)
                if math.isnan(score):
                    _logger.warning("%s %s returned NaN", node.typ, node_name)
                    cache[node_name] = None
                    continue
                # Cache the computed value for downstream
                cache[node_name] = score
                if node_name in self._options:
                    out.append((node_name, params, ScoreWithDeps(score, kw)))
        return sorted(out, key=lambda v: -v[2].score)

    def consider(self, context: ContextType) -> tuple[str, dict[str, Any]]:
        """Return the name of the highest scoring option for *context*.

        Ties are broken using priority specification, followed by
        sorting by name.

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
        best = self.consider_from_scores(scores)
        _logger.debug(
            "Considered %s options, chose %r",
            scores,
            best,
        )
        return best

    def consider_from_scores(
        self, scores: list[tuple[str, dict[str, Any], ScoreWithDeps]]
    ) -> tuple[str, dict[str, Any]]:
        """Utility function allowing re-use of a scoring dict. This exists to avoid
        re-computing scores while allowing external access to the scores dict.
        """
        if len(scores) == 0:
            raise ValueError("Nothing to consider - no options have been added")
        if not all(s in self._options for s in scores):
            missing = [s for s, _, _ in scores if s not in self._options]
            raise ValueError(f"Unrecognised option name(s): {', '.join(missing)}")

        def _key(v: tuple[str, dict[str, Any], ScoreWithDeps]):
            option, params, score_with_deps = v
            score = score_with_deps.score
            if math.isnan(score):
                raise ValueError("Score dict contains NaNs")
            return (-score, -self._nodes[option].priority, option)

        try:
            best = min(scores, key=_key)[:2]
        except KeyError as e:
            raise KeyError(
                "Unknown node - required to ensure priority is respected!"
            ) from e
        return best

    def _register[**P, R: float | None](
        self,
        name: str,
        func: _Scorer[ContextType, P, R],
        typ: Literal["option", "consideration"],
        priority: int = 0,
    ) -> _Scorer[ContextType, P, R]:
        if not isinstance(name, str):
            raise TypeError("Name must be a valid string identifier.")
        if typ == "consideration" and not name.isidentifier():
            raise TypeError(
                "Consideration name must be a valid string identifier to be used in"
                " the dependency tree! This means only names you could use as a"
                " variable."
            )
        if name in self._nodes:
            raise ValueError(f"A function named {name!r} has already been added")
        sig = inspect.signature(func)
        considerations = []
        parameters: list[str] = []
        # Zeroth arg must be ctx
        if len(sig.parameters) == 0:
            raise TypeError(
                f"The first argument accepted by a {typ} function should be a context"
                " type object, but this function accepts no arguments!"
            )
        for arg in list(sig.parameters.values())[1:]:
            if arg.kind not in (arg.POSITIONAL_OR_KEYWORD, arg.KEYWORD_ONLY):
                raise TypeError(
                    f"Args must support keyword injection, got {arg.kind} arg {arg.name}"
                )
            # Check that this is a known this is a consideration or option
            p = None
            if (
                (v := self._nodes.get(arg.name, None)) is None
                and
                (p := self._parameters.get(arg.name, None)) is None
            ):
                    raise ValueError(f'Unknown dependency "{arg.name}"')
            if v is not None:
                match v.typ:
                    case "consideration":
                        self._check_arg_ok(arg, v.typ)
                        considerations.append(arg.name)
                    case "option":
                        raise ValueError(
                            f"A {typ} cannot depend on the output of an option - only considerations."
                        )
                    case _:
                        raise ValueError(f"Unknown type {v.typ}")
            elif p is not None:
                self._check_param_arg_ok(arg)
                parameters.append(arg.name)

        self._nodes[name] = _DAGNode(
            func,
            typ=typ,
            considerations=tuple(considerations),
            priority=priority,
            parameters=tuple(parameters),
        )
        # We _could_ do a cycle detection check, but since inserting a node
        # requires all dependency nodes to already be present, and you can't
        # override an existing node name, we should be safe.
        if typ == "option":
            self._options.add(name)
        self._sort_cache = None
        return func

    def _check_arg_ok(self, arg, typ: Literal['option', 'consideration']):
        if arg.annotation is inspect.Parameter.empty:
            pass  # unannotated is fine
        elif isinstance(arg.annotation, str):
            if arg.annotation != "float":
                raise TypeError(f'Cannot handle {typ} type annotation {arg.annotation}"')
        elif not isinstance(arg.annotation, type):
            raise TypeError("Cannot handle this kind of type annotation.")
        elif arg.annotation is not float:
            raise TypeError(
                f'Expected {typ} annotation as a float, got "{arg.annotation}"'
            )

    def _check_param_arg_ok(self, arg):
        if arg.annotation is inspect.Parameter.empty:
            pass  # unannotated is fine
        elif isinstance(arg.annotation, str):
            if arg.annotation != "str":
                raise TypeError(f'Cannot handle parameter type annotation {arg.annotation}"')
        elif not isinstance(arg.annotation, type):
            raise TypeError("Cannot handle this kind of type annotation.")
        elif arg.annotation is not str:
            raise TypeError(
                f'Expected parameter annotation as a float, got "{arg.annotation}"'
            )

    def _get_dag_compute_order(self) -> tuple[str, ...]:
        if self._sort_cache is not None:
            return self._sort_cache
        to_include = {*self._options}
        to_search = deque(self._options)
        while len(to_search):
            parent = to_search.pop()
            not_yet_searched = [
                node
                for node in self._nodes[parent].considerations
                if node not in to_include
            ]
            to_search.extend(not_yet_searched)
            to_include.update(not_yet_searched)
        ts = graphlib.TopologicalSorter(
            {name: self._nodes[name].considerations for name in sorted(to_include)}
        )
        self._sort_cache = tuple(ts.static_order())
        return self._sort_cache


def _infer_name(obj: Any) -> str:
    name = getattr(obj, "__name__", None)
    if not isinstance(name, str) or name == "<lambda>":
        raise TypeError(
            f"Could not infer a name for {obj!r} - pass one explicitly, "
            "e.g. @things.option('my option')"
        )
    return name
