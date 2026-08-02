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


class TestAdd:
    def test_infers_name_from_function(self, things: ToConsider[Context]):
        @things.option
        def eat_food(ctx: Context) -> float:
            return 1.0

        assert things.names == ("eat_food",)

    def test_uses_explicit_name(self, things: ToConsider[Context]):
        @things.option("go to the shops")
        def go_to_the_shops(ctx: Context) -> float:
            return 1.0

        assert things.names == ("go to the shops",)

    def test_returns_function_unchanged(self, things: ToConsider[Context]):
        def score(ctx: Context) -> float:
            return 0.25

        assert things.option("named")(score) is score
        assert things.option(score) is score

    def test_decorated_function_is_still_callable(self, things: ToConsider[Context]):
        @things.option("named")
        def score(ctx: Context) -> float:
            return 0.25

        assert score(Context()) == 0.25

    def test_rejects_duplicate_names(self, things: ToConsider[Context]):
        things.constant_option("thing", 1.0)
        with pytest.raises(ValueError, match="already been added"):
            things.constant_option("thing", 2.0)

    def test_rejects_duplicate_inferred_names(self, things: ToConsider[Context]):
        @things.option
        def eat_food(ctx: Context) -> float:
            return 1.0

        with pytest.raises(ValueError, match="already been added"):
            things.option(eat_food)

    def test_rejects_non_name_non_callable(self, things: ToConsider[Context]):
        with pytest.raises(TypeError):
            things.option(1.0)  # pyrefly: ignore

    def test_rejects_unnameable_function(self, things: ToConsider[Context]):
        with pytest.raises(TypeError, match="pass one explicitly"):
            things.option(lambda ctx: 1.0)

    def test_lambda_can_be_added_with_a_name(self, things: ToConsider[Context]):
        things.option("anonymous")(lambda ctx: 1.0)
        assert things.consider(Context()) == "anonymous"


class TestAddConstant:
    def test_scores_the_constant(self, things: ToConsider[Context]):
        things.constant_option("starve", 0.25)
        assert things.score(Context()) == {"starve": (0.25, {})}

    def test_acts_as_a_baseline(self, things: ToConsider[Context]):
        things.constant_option("starve", 0.5)

        @things.option("eat food")
        def eat_food(ctx: Context) -> float:
            return curves.is_gt_zero(ctx.food)

        assert things.consider(Context(food=0)) == "starve"
        assert things.consider(Context(food=1)) == "eat food"


class TestConsider:
    def test_chooses_the_highest_scoring_option(self, things: ToConsider[Context]):
        @things.option("go to the shops")
        def go_to_the_shops(ctx: Context) -> float:
            return curves.exponential(ctx.hunger) * curves.is_gt_zero(ctx.money)

        @things.option
        def eat_food(ctx: Context) -> float:
            return curves.exponential(ctx.hunger) * curves.is_gt_zero(ctx.food)

        things.constant_option("starve", 0.0)

        assert things.consider(Context(hunger=1, money=1, food=0)) == "go to the shops"
        assert things.consider(Context(hunger=1, money=0, food=1)) == "eat_food"

    def test_a_zero_baseline_only_ties_with_zero_scoring_options(
        self, things: ToConsider[Context]
    ):
        @things.option
        def eat_food(ctx: Context) -> float:
            return curves.exponential(ctx.hunger) * curves.is_gt_zero(ctx.food)

        things.constant_option("starve", 0.0)

        assert things.score(Context(hunger=1, food=0)) == {
            "eat_food": (0.0, {}),
            "starve": (0.0, {}),
        }
        # ...so the earlier option wins. Add baselines first to win ties.
        assert things.consider(Context(hunger=1, food=0)) == "eat_food"

    def test_ties_favour_the_first_added_option(self, things: ToConsider[Context]):
        things.constant_option("first", 1.0)
        things.constant_option("second", 1.0)
        assert things.consider(Context()) == "first"

    def test_handles_negative_scores(self, things: ToConsider[Context]):
        things.constant_option("bad", -10.0)
        things.constant_option("less bad", -1.0)
        assert things.consider(Context()) == "less bad"

    def test_raises_when_nothing_to_consider(self, things: ToConsider[Context]):
        with pytest.raises(ValueError, match="Nothing to consider"):
            things.consider(Context())

    def test_passes_the_context_through(self, things: ToConsider[Context]):
        seen: list[Context] = []
        things.option("watcher")(lambda ctx: seen.append(ctx) or 1.0)

        context = Context(hunger=3)
        things.consider(context)

        assert seen == [context]


class TestScore:
    def test_scores_every_option_in_order(self, things: ToConsider[Context]):
        things.constant_option("a", 0.1)
        things.constant_option("b", 0.2)
        assert list(things.score(Context()).items()) == [
            ("a", (0.1, {})),
            ("b", (0.2, {})),
        ]

    def test_empty_registry_scores_nothing(self, things: ToConsider[Context]):
        assert things.score(Context()) == {}


class TestContainer:
    def test_len(self, things: ToConsider[Context]):
        assert len(things) == 0
        things.constant_option("thing", 1.0)
        assert len(things) == 1

    def test_contains(self, things: ToConsider[Context]):
        things.constant_option("thing", 1.0)
        assert "thing" in things
        assert "other" not in things

    def test_iterates_over_names(self, things: ToConsider[Context]):
        things.constant_option("a", 1.0, priority=0)
        things.constant_option("b", 1.0, priority=1)
        assert list(things) == ["b", "a"]

    def test_repr_lists_options(self, things: ToConsider[Context]):
        things.constant_option("a", 1.0)
        assert repr(things) == "ToConsider('a')"

    def test_registries_are_independent(self):
        one: ToConsider[Context] = ToConsider()
        other: ToConsider[Context] = ToConsider()
        one.constant_option("thing", 1.0)
        assert other.names == ()
