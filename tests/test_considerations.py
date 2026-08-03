"""Tests for considerations - the shared-dependency DAG behind options.

Tests marked ``xfail(strict=True)`` describe behaviour the library documents
but does not currently implement. They will start failing (as XPASS) once the
underlying bug is fixed, at which point the marker should be removed.
"""

import logging
import math
from dataclasses import dataclass

import pytest

from utilitai import ToConsider, curves


@dataclass
class Context:
    hunger: int = 0
    money: int = 0
    food: int = 0


@pytest.fixture
def things() -> ToConsider[Context]:
    return ToConsider()


class TestRegistration:
    def test_infers_name_from_function(self, things: ToConsider[Context]):
        @things.consideration
        def has_food(ctx: Context) -> float:
            return 1.0

        @things.option
        def eat(ctx: Context, has_food: float) -> float:
            return has_food

        assert things.score(Context())["eat"].deps == {"has_food": 1.0}

    def test_uses_explicit_name(self, things: ToConsider[Context]):
        @things.consideration("has_food")
        def food_check(ctx: Context) -> float:
            return 1.0

        # Dependencies are matched on the registered name, not the function's
        things.option("eat")(lambda ctx, has_food: has_food)
        assert things.score(Context())["eat"].deps == {"has_food": 1.0}

    def test_names_which_are_not_identifiers_cannot_be_depended_on(
        self, things: ToConsider[Context]
    ):
        """Dependencies are resolved via parameter names, so a consideration
        named with a space is unreachable. Worth documenting, or validating at
        registration time rather than at the point of use.
        """
        with pytest.raises(
            TypeError, match="Consideration name must be a valid string identifier"
        ):
            things.consideration("has food")(lambda ctx: 1.0)

    def test_returns_function_unchanged(self, things: ToConsider[Context]):
        def hunger_level(ctx: Context) -> float:
            return 0.25

        assert things.consideration("named")(hunger_level) is hunger_level
        assert things.consideration(hunger_level) is hunger_level

    def test_rejects_unnameable_function(self, things: ToConsider[Context]):
        with pytest.raises(TypeError, match="pass one explicitly"):
            things.consideration(lambda ctx: 1.0)

    def test_shares_a_namespace_with_options(self, things: ToConsider[Context]):
        things.consideration("thing")(lambda ctx: 1.0)
        with pytest.raises(ValueError, match="already been added"):
            things.constant_option("thing", 1.0)

    def test_considerations_are_included(self, things: ToConsider[Context]):
        things.consideration("hunger_level")(lambda ctx: 1.0)
        assert len(things) == 1
        assert things.names == ("hunger_level",)
        assert "hunger_level" in things
        with pytest.raises(ValueError, match="Nothing to consider"):
            things.consider(Context())

    def test_considerations_are_absent_from_scores(self, things: ToConsider[Context]):
        @things.consideration
        def hunger_level(ctx: Context) -> float:
            return 0.5

        @things.option
        def eat(ctx: Context, hunger_level: float) -> float:
            return hunger_level

        assert set(things.score(Context())) == {"eat"}


class TestDependencyValidation:
    def test_rejects_unknown_dependency(self, things: ToConsider[Context]):
        with pytest.raises(ValueError, match='Unknown dependency "has_food"'):

            @things.option
            def eat(ctx: Context, has_food: float) -> float:
                return has_food

    def test_dependencies_must_be_registered_first(self, things: ToConsider[Context]):
        """Forward references are not supported - order of definition matters."""
        with pytest.raises(ValueError, match="Unknown dependency"):

            @things.consideration
            def early(ctx: Context, late: float) -> float:
                return late

        @things.consideration
        def late(ctx: Context) -> float:
            return 1.0

    def test_rejects_depending_on_an_option(self, things: ToConsider[Context]):
        things.constant_option("do_nothing", 0.1)
        with pytest.raises(
            ValueError, match="cannot depend on the output of an option"
        ):

            @things.consideration
            def derived(ctx: Context, do_nothing: float) -> float:
                return do_nothing

    def test_rejects_depending_on_an_option_by_registered_name(
        self, things: ToConsider[Context]
    ):
        things.constant_option("eat", 0.1)
        with pytest.raises(
            ValueError, match="cannot depend on the output of an option"
        ):
            things.consideration("derived")(lambda ctx, eat: eat)

    @pytest.mark.parametrize("annotation", [complex, int, bool])
    def test_does_not_accept_numeric_annotations_other_than_float(
        self, things: ToConsider[Context], annotation: type
    ):
        things.consideration("dep")(lambda ctx: 1.0)

        def option(ctx: Context, dep):
            return dep

        option.__annotations__["dep"] = annotation
        with pytest.raises(TypeError):
            things.option("opt")(option)

    def test_accepts_unannotated_dependencies(self, things: ToConsider[Context]):
        things.consideration("dep")(lambda ctx: 1.0)
        things.option("opt")(lambda ctx, dep: dep)
        assert things.score(Context())["opt"].score == 1.0

    def test_rejects_non_numeric_annotations(self, things: ToConsider[Context]):
        things.consideration("dep")(lambda ctx: 1.0)
        with pytest.raises(TypeError, match="Expected type annotation as a float"):

            @things.option
            def opt(ctx: Context, dep: str) -> float:
                return 1.0

    @pytest.mark.xfail(
        strict=True,
        reason="issubclass() is called on the raw annotation, so PEP 563 string"
        " annotations and unions such as `float | None` or `float | int` raise"
        " TypeError, despite being valid as they would accept a float.",
    )
    def test_accepts_union_annotations(self, things: ToConsider[Context]):
        things.consideration("dep")(lambda ctx: 1.0)

        def option(ctx: Context, dep):
            return dep

        option.__annotations__["dep"] = float | None

        things.option("opt")(option)
        assert things.score(Context())["opt"].score == 1.0

    def test_accepts_stringified_annotations(self, things: ToConsider[Context]):
        things.consideration("dep")(lambda ctx: 1.0)

        def option(ctx: Context, dep):
            return dep

        option.__annotations__["dep"] = "float"  # what PEP 563 produces
        things.option("opt")(option)
        assert things.score(Context())["opt"].score == 1.0


class TestEvaluation:
    def test_passes_the_consideration_value_by_name(self, things: ToConsider[Context]):
        @things.consideration
        def hunger_level(ctx: Context) -> float:
            return curves.linear(ctx.hunger / 10)

        @things.option
        def eat(ctx: Context, hunger_level: float) -> float:
            return hunger_level

        assert things.score(Context(hunger=7))["eat"].score == pytest.approx(0.7)

    def test_passes_the_context_to_considerations(self, things: ToConsider[Context]):
        seen: list[Context] = []
        things.consideration("watcher")(lambda ctx: seen.append(ctx) or 1.0)
        things.option("opt")(lambda ctx, watcher: watcher)

        context = Context(hunger=3)
        things.score(context)
        assert seen == [context]

    def test_records_dependencies_alongside_the_score(
        self, things: ToConsider[Context]
    ):
        things.consideration("has_money")(lambda ctx: curves.is_gt_zero(ctx.money))
        things.consideration("hunger_level")(lambda ctx: ctx.hunger / 10)
        things.option("shop")(
            lambda ctx, has_money, hunger_level: has_money * hunger_level
        )

        score, deps = things.score(Context(hunger=6, money=5))["shop"]
        assert score == pytest.approx(0.6)
        assert deps == {"has_money": 1.0, "hunger_level": pytest.approx(0.6)}

    def test_constant_options_have_no_dependencies(self, things: ToConsider[Context]):
        things.constant_option("sleep", 0.1)
        assert things.score(Context())["sleep"].deps == {}

    def test_considerations_may_depend_on_considerations(
        self, things: ToConsider[Context]
    ):
        @things.consideration
        def raw(ctx: Context) -> float:
            return ctx.hunger / 10

        @things.consideration
        def shaped(ctx: Context, raw: float) -> float:
            return curves.quadratic(raw)

        @things.option
        def eat(ctx: Context, shaped: float) -> float:
            return shaped

        assert things.score(Context(hunger=4))["eat"].score == pytest.approx(0.16)

    def test_a_shared_consideration_is_evaluated_once_per_score(
        self, things: ToConsider[Context]
    ):
        calls = 0

        @things.consideration
        def shared(ctx: Context) -> float:
            nonlocal calls
            calls += 1
            return 0.5

        @things.consideration
        def left(ctx: Context, shared: float) -> float:
            return shared

        @things.consideration
        def right(ctx: Context, shared: float) -> float:
            return shared

        @things.option
        def opt(ctx: Context, left: float, right: float) -> float:
            return left + right

        assert things.score(Context())["opt"].score == 1.0
        assert calls == 1

    def test_nothing_is_cached_between_calls(self, things: ToConsider[Context]):
        """Considerations see fresh context on every tick."""
        calls = 0

        @things.consideration
        def counter(ctx: Context) -> float:
            nonlocal calls
            calls += 1
            return float(calls)

        things.option("opt")(lambda ctx, counter: counter)

        assert things.score(Context())["opt"].score == 1.0
        assert things.score(Context())["opt"].score == 2.0

    def test_unused_considerations_do_not_affect_options(
        self, things: ToConsider[Context]
    ):
        things.consideration("unused")(lambda ctx: None)
        things.constant_option("sleep", 0.1)
        assert things.score(Context()) == {"sleep": (0.1, {})}


class TestShortCircuiting:
    """Returning ``None`` from a consideration drops everything downstream."""

    def test_none_drops_the_dependent_option(self, things: ToConsider[Context]):
        things.consideration("has_food")(lambda ctx: None if ctx.food == 0 else 1.0)
        things.option("eat")(lambda ctx, has_food: has_food)
        things.constant_option("sleep", 0.1)

        assert set(things.score(Context(food=0))) == {"sleep"}
        assert set(things.score(Context(food=1))) == {"eat", "sleep"}

    def test_none_propagates_through_a_chain(self, things: ToConsider[Context]):
        things.consideration("root")(lambda ctx: None)
        things.consideration("middle")(lambda ctx, root: root)
        things.option("leaf")(lambda ctx, middle: middle)
        things.constant_option("sleep", 0.1)

        assert set(things.score(Context())) == {"sleep"}

    def test_none_only_drops_the_dependent_branch(self, things: ToConsider[Context]):
        things.consideration("blocked")(lambda ctx: None)
        things.consideration("fine")(lambda ctx: 0.5)
        things.option("skipped")(lambda ctx, blocked: blocked)
        things.option("kept")(lambda ctx, fine: fine)

        assert set(things.score(Context())) == {"kept"}

    def test_an_option_may_return_none_directly(self, things: ToConsider[Context]):
        things.option("opts out")(lambda ctx: None)
        things.constant_option("sleep", 0.1)
        assert set(things.score(Context())) == {"sleep"}

    def test_nan_from_a_consideration_drops_the_option(
        self, things: ToConsider[Context]
    ):
        things.consideration("broken")(lambda ctx: math.nan)
        things.option("eat")(lambda ctx, broken: broken)
        things.constant_option("sleep", 0.1)

        assert set(things.score(Context())) == {"sleep"}

    def test_nan_from_an_option_drops_it(self, things: ToConsider[Context]):
        things.option("broken")(lambda ctx: math.nan)
        things.constant_option("sleep", 0.1)
        assert set(things.score(Context())) == {"sleep"}

    def test_consider_raises_when_every_option_is_dropped(
        self, things: ToConsider[Context]
    ):
        things.consideration("gate")(lambda ctx: None)
        things.option("only")(lambda ctx, gate: gate)

        with pytest.raises(ValueError, match="No good options"):
            things.consider(Context())

    def test_partial_dependency_failure_drops_the_option(
        self, things: ToConsider[Context]
    ):
        things.consideration("ok")(lambda ctx: 1.0)
        things.consideration("blocked")(lambda ctx: None)
        things.option("eat")(lambda ctx, ok, blocked: ok + blocked)
        things.constant_option("sleep", 0.1)

        assert set(things.score(Context())) == {"sleep"}


class TestConsiderWithConsiderations:
    def test_chooses_the_highest_scoring_surviving_option(
        self, things: ToConsider[Context]
    ):
        things.consideration("has_money")(lambda ctx: None if ctx.money == 0 else 1.0)
        things.consideration("has_food")(lambda ctx: None if ctx.food == 0 else 1.0)
        things.consideration("hunger_level")(lambda ctx: ctx.hunger / 10)

        things.option("eat")(lambda ctx, has_food, hunger_level: hunger_level)
        things.option("shop")(lambda ctx, has_money, hunger_level: 0.5 * hunger_level)
        things.constant_option("sleep", 0.01)

        assert things.consider(Context(hunger=8, money=1, food=1)) == "eat"
        assert things.consider(Context(hunger=8, money=1, food=0)) == "shop"
        assert things.consider(Context(hunger=8, money=0, food=0)) == "sleep"

    def test_ties_favour_alphanumeric_sorting(self, things: ToConsider[Context]):
        things.consideration("gate")(lambda ctx: 1.0)
        things.constant_option("b", 1.0)
        things.option("a")(lambda ctx, gate: 1.0)

        assert things.consider(Context()) == "a"

    def test_priority_breaks_ties(self, things: ToConsider[Context]):
        things.constant_option("low", 1.0, priority=0)
        things.constant_option("high", 1.0, priority=10)

        assert things.consider(Context()) == "high"


class TestLogging:
    def test_logs_which_dependency_blocked_an_option(
        self, things: ToConsider[Context], caplog: pytest.LogCaptureFixture
    ):
        things.consideration("gate")(lambda ctx: None)
        things.option("blocked")(lambda ctx, gate: gate)
        things.constant_option("sleep", 0.1)

        with caplog.at_level(logging.DEBUG, logger="utilitai"):
            things.score(Context())

        assert "gate" in caplog.text
        assert "blocked" in caplog.text

    def test_warns_when_a_consideration_returns_nan(
        self, things: ToConsider[Context], caplog: pytest.LogCaptureFixture
    ):
        things.consideration("broken")(lambda ctx: math.nan)
        things.option("eat")(lambda ctx, broken: broken)

        with caplog.at_level(logging.WARNING, logger="utilitai"):
            things.score(Context())

        assert "NaN" in caplog.text


class TestConsiderFromScores:
    def test_chooses_the_highest_score(self, things: ToConsider[Context]):
        things.constant_option("zzz best", 1.0)
        things.constant_option("aaa worst", 0.0)

        scores = {name: value.score for name, value in things.score(Context()).items()}
        assert things.consider_from_scores(scores) == "zzz best"

    def test_priority_breaks_ties(self, things: ToConsider[Context]):
        things.constant_option("low", 1.0, priority=0)
        things.constant_option("high", 1.0, priority=10)

        assert things.consider_from_scores({"low": 1.0, "high": 1.0}) == "high"

    def test_raises_on_NaN(self, things: ToConsider[Context]):
        things.constant_option("low", 1.0, priority=0)
        things.constant_option("high", 1.0, priority=10)
        with pytest.raises(ValueError):
            assert things.consider_from_scores({"low": float("nan"), "high": 1.0})

    def test_agrees_with_consider(self, things: ToConsider[Context]):
        things.consideration("hunger_level")(lambda ctx: ctx.hunger / 10)
        things.option("eat")(lambda ctx, hunger_level: hunger_level)
        things.constant_option("sleep", 0.3)

        context = Context(hunger=9)
        scores = {name: value.score for name, value in things.score(context).items()}
        assert things.consider_from_scores(scores) == things.consider(context)

    def test_rejects_unknown_names(self, things: ToConsider[Context]):
        things.constant_option("sleep", 0.1)
        with pytest.raises(
            ValueError, match=r"Unrecognised option name\(s\): not registered"
        ):
            things.consider_from_scores({"not registered": 1.0})

    def test_rejects_an_empty_mapping(self, things: ToConsider[Context]):
        with pytest.raises(ValueError):
            things.consider_from_scores({})

    def test_mixing_contexts_is_a_type_error(self, things: ToConsider[Context]):
        @things.option  # pyrefly: ignore
        def foo(ctx: None):
            return 1.0
