"""Drive a :mod:`btreeny` behaviour tree from a :class:`~utilitai.ToConsider`.

This module is optional. Install it with::

    pip install 'utilitai[btreeny]'

The idea is simple: ``btreeny.keyed`` picks a subtree from a key, and
``ToConsider.consider`` produces exactly such a key. Wiring the two together
directly is a trap, though - ``keyed`` only rebuilds a subtree when the key
*changes*, so re-selecting the option that just finished re-ticks a completed
behaviour and raises ``BehaviourCompleteError``. Utility scores also tend to sit
close together, so a naive ``key_fn`` will thrash: tearing down and rebuilding
subtrees every tick and never letting a RUNNING action finish.

:func:`committed` handles both. An option, once chosen, is *committed to* until
the subtree running it completes, and completion triggers a fresh decision::

    goals: ToConsider[Blackboard] = ToConsider()

    @goals.option("forage")
    def forage(bb: Blackboard) -> float: ...

    tree = committed(
        Commitment.on_attribute(goals, default="idle"),
        {"forage": forage_tree, "idle": idle_tree},
    )

    with tree as tick:
        while True:
            tick(blackboard)

Branch authors do not touch the commitment: it is released for them whenever
their subtree stops RUNNING.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from .commitment import Commitment

try:
    import btreeny as _bt
except ModuleNotFoundError as e:  # pragma: no cover - exercised in a bare env
    raise ImportError(
        "utilitai.btreeny requires btreeny, which is not installed. "
        "Install it with:  pip install 'utilitai[btreeny]'"
    ) from e

if TYPE_CHECKING:
    from btreeny import IdType, TreeNode, TreeStatus

__all__ = ["Commitment", "NoValidOptionError", "committed"]

_logger = logging.getLogger(__name__)


class NoValidOptionError(RuntimeError):
    """Raised when nothing scores and no fallback was configured.

    A ``key_fn`` runs inside the tree tick, so raising here will tear down the
    whole tree. Pass ``default=`` to :class:`Commitment` to make this
    impossible.
    """


@_bt.action
def _releasing[ContextType](
    node_id: IdType,
    commitment: Commitment[ContextType],
    factory: Callable[[], TreeNode[ContextType]],
):
    """Release *commitment* whenever the wrapped subtree stops RUNNING.

    Centralising this means a branch cannot strand the latch by forgetting to
    clear it on its failure path - which would lock the agent out of its own
    decision layer permanently.
    """
    with factory() as tick:

        def _inner(context: ContextType) -> TreeStatus:
            result = tick(context)
            if result is not _bt.RUNNING:
                _logger.debug("subtree finished with %s, releasing commitment", result)
                commitment.release(context)
            return result

        yield _inner


def committed[ContextType](
    commitment: Commitment[ContextType],
    trees: Mapping[str, Callable[[], TreeNode[ContextType]]],
    *,
    absorb_failures: bool = True,
) -> TreeNode[ContextType]:
    """Build a tree which runs one option at a time, to completion.

    Each tick, the committed option's subtree is ticked. When that subtree
    stops RUNNING the commitment is released and the next tick picks again.

    Parameters
    ----------
    commitment:
        Supplies the ``key_fn`` and owns the latch.
    trees:
        Maps option name to a *factory* for that option's subtree. Factories,
        not nodes: a fresh subtree is built for each run, since btreeny nodes
        are single use.
    absorb_failures:
        Map FAILURE to SUCCESS at the root so a failing option is treated as
        input to the next decision rather than as a reason to stop the agent.
        Turn this off only if a caller above is prepared to handle FAILURE;
        without it the tick after a failure raises ``BehaviourCompleteError``.

    Notes
    -----
    The returned node is wrapped in ``btreeny.redo``. That is what makes
    re-selecting the option which just finished safe: the whole ``keyed`` node
    is rebuilt on completion, so its key comparison starts fresh.
    """
    if unknown := sorted(set(trees) - commitment.known_names):
        raise ValueError(
            f"No option named {', '.join(map(repr, unknown))} is registered - "
            "tree keys must match option names in the registry (or in preempt)"
        )

    if commitment.default is not None and commitment.default not in trees:
        raise ValueError(
            f"Default option {commitment.default!r} has no subtree - the "
            "fallback must always be runnable"
        )

    def _value_fn(key: str) -> TreeNode[ContextType]:
        try:
            factory = trees[key]
        except KeyError:
            raise KeyError(
                f"Option {key!r} was chosen but has no subtree. Known subtrees: "
                f"{', '.join(map(repr, sorted(trees)))}"
            ) from None
        return _releasing(commitment, factory)

    def _node() -> TreeNode[ContextType]:
        node = _bt.keyed(commitment.key_for, _value_fn)
        if absorb_failures:
            # remap, *not* always_return: always_return would report SUCCESS
            # while the subtree is still RUNNING, making redo rebuild it every
            # tick so multi-tick actions could never progress past their first.
            node = _bt.remap(node, {_bt.FAILURE: _bt.SUCCESS})
        return node

    return _bt.redo(_node)
