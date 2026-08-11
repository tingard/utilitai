"""
:class:`Commitment` Is a plain latch over a :class:`~utilitai.ToConsider`,
useful from any tick loop.
"""

from __future__ import annotations

import logging
from typing import Protocol, Self

from . import ToConsider

__all__ = ["Commitment", "NoValidOptionError"]

_logger = logging.getLogger(__name__)


class NoValidOptionError(RuntimeError):
    """Raised when nothing scores and no fallback was configured.

    A ``key_fn`` runs inside the tree tick, so raising here will tear down the
    whole tree. Pass ``default=`` to :class:`Commitment` to make this
    impossible.
    """


class _Reader[ContextType](Protocol):
    def __call__(self, context: ContextType, /) -> str | None: ...


class _Writer[ContextType](Protocol):
    def __call__(self, context: ContextType, value: str | None, /) -> None: ...


class Commitment[ContextType]:
    """A latch which holds a chosen option until it is released.

    The current choice is stored on the caller's context (btreeny's
    blackboard), not on this object, so it stays visible to actions and to
    anything else inspecting the world state.

    Parameters
    ----------
    options:
        The registry to consult when a fresh decision is needed.
    get_current, set_current:
        Read and write the committed option name on the context. Use
        :meth:`on_attribute` for the common case of a plain attribute.
    default:
        Option to fall back to when nothing scores - because every option
        short-circuited on ``None``, returned NaN, or the registry is empty.
        Strongly recommended: without it, :meth:`key_for` can raise
        :class:`NoValidOptionError` mid-tick.
    preempt:
        An optional second registry checked *before* the latch. Anything which
        scores here interrupts the committed option immediately. Use it for
        conditions that must not wait, and keep it small - every option in it
        is evaluated on every tick.
    abandon_below:
        Release the commitment early if the committed option's own score falls
        below this value. Together with the ordinary selection threshold this
        gives hysteresis: an option must score well to be chosen, but only
        this well to be continued.
    log_scores:
        Evaluate and log the full score table even while latched, so near
        misses stay visible. Costs a full DAG evaluation per tick.
    """

    def __init__(
        self,
        to_consider: ToConsider[ContextType],
        *,
        get_current: _Reader[ContextType],
        set_current: _Writer[ContextType],
        default: str | None = None,
        preempt: ToConsider[ContextType] | None = None,
        abandon_below: float | None = None,
        log_scores: bool = False,
    ) -> None:
        if default is not None and default not in to_consider.options:
            raise ValueError(f"Default option {default!r} is not registered")
        self._to_consider = to_consider
        self._preempt = preempt
        self._get_current = get_current
        self._set_current = set_current
        self._default = default
        self._abandon_below = abandon_below
        self._log_scores = log_scores

    @classmethod
    def on_attribute(
        cls,
        options: ToConsider[ContextType],
        attribute: str = "current_action",
        *,
        default: str | None = None,
        preempt: ToConsider[ContextType] | None = None,
        abandon_below: float | None = None,
        log_scores: bool = False,
    ) -> Self:
        """Store the commitment in an attribute of the context::

        @dataclass
        class Blackboard:
            current_action: str | None = None

        commitment = Commitment.on_attribute(goals, default="idle")
        """

        def _get(context: ContextType, /) -> str | None:
            return getattr(context, attribute)

        def _set(context: ContextType, value: str | None, /) -> None:
            setattr(context, attribute, value)

        return cls(
            options,
            get_current=_get,
            set_current=_set,
            default=default,
            preempt=preempt,
            abandon_below=abandon_below,
            log_scores=log_scores,
        )

    @property
    def options(self) -> ToConsider[ContextType]:
        return self._to_consider

    @property
    def default(self) -> str | None:
        return self._default

    @property
    def preempt(self) -> ToConsider[ContextType] | None:
        return self._preempt

    @property
    def known_names(self) -> frozenset[str]:
        """Every name which :meth:`key_for` could return.

        Includes considerations as well as options, since ``ToConsider`` does
        not currently expose the two separately.
        """
        names = set(self._to_consider.names)
        if self._preempt is not None:
            names |= set(self._preempt.names)
        if self._default is not None:
            names.add(self._default)
        return frozenset(names)

    def release(self, context: ContextType) -> None:
        """Drop the current commitment, so the next tick decides afresh."""
        self._set_current(context, None)

    def key_for(self, context: ContextType) -> str:
        """Return the option to run now. Never raises if ``default`` is set.

        Suitable as ``btreeny.keyed``'s ``key_fn``.
        """
        current = self._get_current(context)

        if self._preempt is not None:
            preempt_scores = self._preempt.score(context)
            urgent = (
                self._preempt.consider_from_scores(preempt_scores)
                if len(preempt_scores)
                else None
            )
            if urgent is not None and urgent != current:
                _logger.info("preempting %r with %r", current, urgent)
                self._set_current(context, urgent)
                return urgent

        scores = None
        if current is not None:
            if self._abandon_below is None and not self._log_scores:
                return current
            scores = self._to_consider.score(context)
            if self._log_scores:
                _logger.debug("committed to %r, scores were %s", current, scores)
            if current not in self._to_consider.options:
                # An interrupt is running. It is scored against `preempt`, not
                # against the routine registry, so it would look like it had
                # collapsed on every tick. Interrupts run to completion.
                return current
            live = scores[current].score if current in scores else None
            if live is not None and (
                self._abandon_below is None or live >= self._abandon_below
            ):
                return current
            _logger.debug(
                "%r scored %s, below %s - reconsidering",
                current,
                live,
                self._abandon_below,
            )
            self._set_current(context, None)

        if scores is None:
            scores = self._to_consider.score(context)
        chosen = (
            self._to_consider.consider_from_scores(scores)
            if len(scores)
            else self._default
        )
        if chosen is None:
            raise NoValidOptionError(
                "No option scored and no default was configured. Pass "
                "default= to Commitment so the tree cannot be torn down "
                "by an undecidable tick."
            )
        if chosen != current:
            # Reconsidering often re-picks the option we were already running,
            # which is not a switch - keyed will not rebuild anything.
            _logger.info("committing to %r (was %r)", chosen, current)
        _logger.debug("chose %r from %s", chosen, scores)
        self._set_current(context, chosen)
        return chosen
