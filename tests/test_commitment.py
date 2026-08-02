"""Tests for :class:`utilitai.btreeny.Commitment` - the latch which holds a
chosen option until something releases it.

``Commitment`` needs nothing from btreeny at runtime, but it currently lives in
``utilitai.btreeny``, so importing it pulls btreeny in. These tests skip when
the optional extra is not installed.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from utilitai import ToConsider
from utilitai.commitment import Commitment, NoValidOptionError


@dataclass
class Context:
    energy: float = 1.0
    danger: bool = False
    current_action: str | None = None


@pytest.fixture
def goals() -> ToConsider[Context]:
    things: ToConsider[Context] = ToConsider()
    things.option("burn")(lambda ctx: ctx.energy)
    things.constant_option("idle", 0.2)
    return things


@pytest.fixture
def commitment(goals: ToConsider[Context]) -> Commitment[Context]:
    return Commitment.on_attribute(goals, default="idle")


class TestConstruction:
    def test_rejects_an_unregistered_default(self, goals: ToConsider[Context]):
        with pytest.raises(ValueError, match="Default option 'nope' is not registered"):
            Commitment.on_attribute(goals, default="nope")

    def test_accepts_no_default(self, goals: ToConsider[Context]):
        assert Commitment.on_attribute(goals).default is None

    def test_exposes_its_registries(self, goals: ToConsider[Context]):
        urgent: ToConsider[Context] = ToConsider()
        urgent.constant_option("flee", 1.0)
        c = Commitment.on_attribute(goals, default="idle", preempt=urgent)
        assert c.options is goals
        assert c.preempt is urgent
        assert c.default == "idle"


class TestStorage:
    def test_on_attribute_uses_the_named_attribute(self, goals: ToConsider[Context]):
        c = Commitment.on_attribute(goals, default="idle")
        ctx = Context()
        assert c.key_for(ctx) == "burn"
        assert ctx.current_action == "burn"

    def test_on_attribute_accepts_a_custom_attribute_name(self):
        @dataclass
        class Other:
            goal: str | None = None

        things: ToConsider[Other] = ToConsider()
        things.constant_option("wait", 1.0)
        c = Commitment.on_attribute(things, "goal", default="wait")

        ctx = Other()
        assert c.key_for(ctx) == "wait"
        assert ctx.goal == "wait"

    def test_accepts_explicit_getter_and_setter(self, goals: ToConsider[Context]):
        store: dict[str, str | None] = {"key": None}
        c = Commitment(
            goals,
            get_current=lambda ctx: store["key"],
            set_current=lambda ctx, value: store.__setitem__("key", value),
            default="idle",
        )

        assert c.key_for(Context()) == "burn"
        assert store == {"key": "burn"}

    def test_release_clears_the_latch(self, commitment: Commitment[Context]):
        ctx = Context(current_action="burn")
        commitment.release(ctx)
        assert ctx.current_action is None

    def test_state_lives_on_the_context_not_the_commitment(
        self, commitment: Commitment[Context]
    ):
        one, other = Context(), Context()
        commitment.key_for(one)
        assert one.current_action == "burn"
        assert other.current_action is None


class TestDeciding:
    def test_chooses_the_highest_scoring_option(self, commitment: Commitment[Context]):
        assert commitment.key_for(Context(energy=1.0)) == "burn"
        assert commitment.key_for(Context(energy=0.05)) == "idle"

    def test_holds_the_latch_once_committed(self, commitment: Commitment[Context]):
        ctx = Context(energy=1.0)
        assert commitment.key_for(ctx) == "burn"
        # 'burn' now scores far below 'idle', but we are committed
        ctx.energy = 0.0
        assert commitment.key_for(ctx) == "burn"

    def test_decides_again_after_release(self, commitment: Commitment[Context]):
        ctx = Context(energy=1.0)
        commitment.key_for(ctx)
        ctx.energy = 0.0
        commitment.release(ctx)
        assert commitment.key_for(ctx) == "idle"

    def test_does_not_score_while_latched(self, goals: ToConsider[Context]):
        """The whole point of the latch is to skip the DAG on committed ticks."""
        calls = 0

        things: ToConsider[Context] = ToConsider()

        @things.option
        def counted(ctx: Context) -> float:
            nonlocal calls
            calls += 1
            return 1.0

        c = Commitment.on_attribute(things, default="counted")
        ctx = Context()

        c.key_for(ctx)
        assert calls == 1
        c.key_for(ctx)
        c.key_for(ctx)
        assert calls == 1


class TestFallback:
    """The fallback only fires when *every* option short-circuits.

    A constant option always scores, so a registry containing one never
    reaches this branch - which makes it easy to write a test that passes
    without exercising anything.
    """

    @pytest.fixture
    def all_dead(self) -> ToConsider[Context]:
        things: ToConsider[Context] = ToConsider()
        things.option("impossible")(lambda ctx: None)
        things.option("also_impossible")(lambda ctx: None)
        return things

    def test_falls_back_to_the_default(self, all_dead: ToConsider[Context]):
        c = Commitment.on_attribute(all_dead, default="impossible")
        ctx = Context()

        assert all_dead.score(ctx) == {}, "precondition: nothing scores"
        assert c.key_for(ctx) == "impossible"
        assert ctx.current_action == "impossible"

    def test_a_scoring_option_beats_the_default(self, goals: ToConsider[Context]):
        c = Commitment.on_attribute(goals, default="idle")
        assert c.key_for(Context(energy=1.0)) == "burn"

    def test_raises_when_nothing_scores_and_there_is_no_default(
        self, all_dead: ToConsider[Context]
    ):
        c = Commitment.on_attribute(all_dead)

        with pytest.raises(NoValidOptionError) as excinfo:
            c.key_for(Context())

        # The message should point at the fix, since this fires mid-tick
        assert "no default was configured" in str(excinfo.value)
        assert "default=" in str(excinfo.value)

    def test_the_latch_is_not_set_when_the_decision_fails(
        self, all_dead: ToConsider[Context]
    ):
        c = Commitment.on_attribute(all_dead)
        ctx = Context()

        with pytest.raises(NoValidOptionError):
            c.key_for(ctx)

        assert ctx.current_action is None


class TestPreemption:
    @pytest.fixture
    def urgent(self) -> ToConsider[Context]:
        emergencies: ToConsider[Context] = ToConsider()
        emergencies.option("flee")(lambda ctx: 1.0 if ctx.danger else None)
        return emergencies

    @pytest.fixture
    def preempting(
        self, goals: ToConsider[Context], urgent: ToConsider[Context]
    ) -> Commitment[Context]:
        return Commitment.on_attribute(goals, default="idle", preempt=urgent)

    def test_stays_out_of_the_way_when_nothing_is_urgent(
        self, preempting: Commitment[Context]
    ):
        assert preempting.key_for(Context()) == "burn"

    def test_interrupts_a_committed_option(self, preempting: Commitment[Context]):
        ctx = Context()
        assert preempting.key_for(ctx) == "burn"
        ctx.danger = True
        assert preempting.key_for(ctx) == "flee"
        assert ctx.current_action == "flee"

    def test_holds_the_urgent_option_while_it_persists(
        self, preempting: Commitment[Context]
    ):
        ctx = Context(danger=True)
        assert preempting.key_for(ctx) == "flee"
        assert preempting.key_for(ctx) == "flee"

    def test_falls_back_to_normal_goals_once_clear(
        self, preempting: Commitment[Context]
    ):
        ctx = Context(danger=True)
        preempting.key_for(ctx)
        ctx.danger = False
        # still latched to flee - preemption commits like anything else
        assert preempting.key_for(ctx) == "flee"
        preempting.release(ctx)
        assert preempting.key_for(ctx) == "burn"

    def test_preempt_names_count_as_known(self, preempting: Commitment[Context]):
        assert "flee" in preempting.known_names
        assert {"burn", "idle"} <= preempting.known_names


class TestAbandonBelow:
    @pytest.fixture
    def fickle(self, goals: ToConsider[Context]) -> Commitment[Context]:
        return Commitment.on_attribute(goals, default="idle", abandon_below=0.5)

    def test_holds_while_the_score_stays_high(self, fickle: Commitment[Context]):
        ctx = Context(energy=1.0)
        assert fickle.key_for(ctx) == "burn"
        ctx.energy = 0.6
        assert fickle.key_for(ctx) == "burn"

    def test_reconsiders_when_the_score_decays(self, fickle: Commitment[Context]):
        ctx = Context(energy=1.0)
        assert fickle.key_for(ctx) == "burn"
        ctx.energy = 0.1
        assert fickle.key_for(ctx) == "idle"
        assert ctx.current_action == "idle"

    def test_reconsidering_may_re_pick_the_same_option(
        self, fickle: Commitment[Context]
    ):
        """Below the threshold but still the best available - stay put."""
        ctx = Context(energy=1.0)
        assert fickle.key_for(ctx) == "burn"
        # 0.3 is under abandon_below, but still beats idle's 0.2
        ctx.energy = 0.3
        assert fickle.key_for(ctx) == "burn"
        assert ctx.current_action == "burn"

    def test_releases_when_the_committed_option_short_circuits(self):
        things: ToConsider[Context] = ToConsider()
        things.option("burn")(lambda ctx: None if ctx.energy <= 0 else ctx.energy)
        things.constant_option("idle", 0.2)
        c = Commitment.on_attribute(things, default="idle", abandon_below=0.5)

        ctx = Context(energy=1.0)
        assert c.key_for(ctx) == "burn"
        ctx.energy = 0.0  # 'burn' now returns None and vanishes from scores
        assert c.key_for(ctx) == "idle"

    def test_does_not_abandon_a_committed_interrupt(self):
        """Regression: an interrupt is scored against ``preempt``, never
        against the routine registry, so checking it against the latter makes
        it look collapsed on every single tick - and the robot thrashes
        between the interrupt and its duties.
        """
        duties: ToConsider[Context] = ToConsider()
        duties.constant_option("patrol", 0.8)
        urgent: ToConsider[Context] = ToConsider()
        urgent.option("investigate")(lambda ctx: 0.6 if ctx.danger else None)

        c = Commitment.on_attribute(
            duties, default="patrol", preempt=urgent, abandon_below=0.1
        )
        ctx = Context(danger=True)

        assert c.key_for(ctx) == "investigate"
        assert "investigate" not in duties, "precondition: not a routine option"
        # Held, not abandoned, even though it scores nothing in `duties`
        assert c.key_for(ctx) == "investigate"
        assert c.key_for(ctx) == "investigate"

    def test_does_not_abandon_a_committed_interrupt_when_logging_scores(self):
        """log_scores reaches the same branch by a different route."""
        duties: ToConsider[Context] = ToConsider()
        duties.constant_option("patrol", 0.8)
        urgent: ToConsider[Context] = ToConsider()
        urgent.option("investigate")(lambda ctx: 0.6 if ctx.danger else None)

        c = Commitment.on_attribute(
            duties, default="patrol", preempt=urgent, log_scores=True
        )
        ctx = Context(danger=True)

        assert c.key_for(ctx) == "investigate"
        assert c.key_for(ctx) == "investigate"

    def test_scores_on_every_tick_when_set(self, goals: ToConsider[Context]):
        calls = 0

        things: ToConsider[Context] = ToConsider()

        @things.option
        def counted(ctx: Context) -> float:
            nonlocal calls
            calls += 1
            return 1.0

        c = Commitment.on_attribute(things, default="counted", abandon_below=0.5)
        ctx = Context()

        c.key_for(ctx)
        c.key_for(ctx)
        c.key_for(ctx)
        assert calls == 3


@pytest.fixture
def logs() -> Iterator[list[str]]:
    """Collect the module's own log records.

    Deliberately not ``caplog``: that relies on records propagating up to the
    root logger, and anything in the host application - or a library imported
    by it - can switch propagation off, at which point ``caplog.text`` is
    silently empty and these tests either fail spuriously or pass vacuously.
    Attaching a handler to the logger itself sidesteps all of that.
    """
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    # Keyed off the module, so relocating Commitment does not silently
    # disconnect the capture.
    logger = logging.getLogger(Commitment.__module__)
    handler = _Collect()
    previous_level = logger.level
    # setLevel(), not `logger.level = ...`: only the former clears
    # Logger._cache, which otherwise remembers that INFO was disabled from
    # whenever this logger was last used and silently drops every record.
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


class TestLogging:
    def test_logs_the_committed_option(
        self, commitment: Commitment[Context], logs: list[str]
    ):
        commitment.key_for(Context())
        assert any("burn" in line for line in logs)

    def test_logs_preemption(self, goals: ToConsider[Context], logs: list[str]):
        emergencies: ToConsider[Context] = ToConsider()
        emergencies.option("flee")(lambda ctx: 1.0 if ctx.danger else None)
        c = Commitment.on_attribute(goals, default="idle", preempt=emergencies)

        ctx = Context()
        c.key_for(ctx)
        ctx.danger = True
        c.key_for(ctx)

        assert any("preempting" in line for line in logs)

    def test_does_not_log_a_switch_when_the_choice_is_unchanged(
        self, goals: ToConsider[Context], logs: list[str]
    ):
        c = Commitment.on_attribute(goals, default="idle", abandon_below=0.5)
        ctx = Context(energy=0.3)  # under the threshold, but still the best
        c.key_for(ctx)
        logs.clear()

        c.key_for(ctx)
        c.key_for(ctx)

        assert not any("committing" in line for line in logs)
        # ...and the capture really is live, so this is not passing vacuously
        assert any("reconsidering" in line for line in logs)

    def test_log_scores_forces_evaluation_while_latched(self, logs: list[str]):
        calls = 0

        things: ToConsider[Context] = ToConsider()

        @things.option
        def counted(ctx: Context) -> float:
            nonlocal calls
            calls += 1
            return 1.0

        c = Commitment.on_attribute(things, default="counted", log_scores=True)
        ctx = Context()

        c.key_for(ctx)
        c.key_for(ctx)
        assert calls == 2
        assert any("committed to" in line for line in logs)


class TestKnownNames:
    def test_covers_options(self, commitment: Commitment[Context]):
        assert {"burn", "idle"} <= commitment.known_names

    def test_excludes_unregistered_names(self, commitment: Commitment[Context]):
        assert "flee" not in commitment.known_names

    def test_is_immutable(self, commitment: Commitment[Context]):
        assert isinstance(commitment.known_names, frozenset)
