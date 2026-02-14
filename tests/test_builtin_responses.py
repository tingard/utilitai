import pytest

from utilitai import curves


class TestEps:
    def test_returns_float(self):
        assert isinstance(curves.eps(), float)

    def test_returns_machine_epsilon(self):
        import sys

        assert curves.eps() == sys.float_info.epsilon

    def test_ignores_positional_args(self):
        assert curves.eps(1, 2, 3) == curves.eps()

    def test_ignores_keyword_args(self):
        assert curves.eps(a=1, b=2) == curves.eps()

    def test_greater_than_zero(self):
        assert curves.eps() > 0


class TestLinear:
    def test_zero(self):
        assert curves.linear(0.0) == 0.0

    def test_one(self):
        assert curves.linear(1.0) == 1.0

    def test_identity(self):
        assert curves.linear(0.42) == 0.42

    def test_negative(self):
        assert curves.linear(-0.5) == -0.5


class TestInverseLinear:
    def test_zero(self):
        assert curves.inverse_linear(0.0) == 1.0

    def test_one(self):
        assert curves.inverse_linear(1.0) == 0.0

    def test_midpoint(self):
        assert curves.inverse_linear(0.5) == pytest.approx(0.5)

    def test_complement(self):
        assert curves.inverse_linear(0.3) == pytest.approx(0.7)


class TestQuadratic:
    def test_zero(self):
        assert curves.quadratic(0.0) == 0.0

    def test_one(self):
        assert curves.quadratic(1.0) == 1.0

    def test_midpoint(self):
        assert curves.quadratic(0.5) == pytest.approx(0.25)

    def test_accelerates(self):
        # quadratic should be below linear for values in (0, 1)
        assert curves.quadratic(0.5) < curves.linear(0.5)

    def test_known_value(self):
        assert curves.quadratic(0.3) == pytest.approx(0.09)


class TestInverseQuadratic:
    def test_zero(self):
        assert curves.inverse_quadratic(0.0) == 0.0

    def test_one(self):
        assert curves.inverse_quadratic(1.0) == 1.0

    def test_midpoint(self):
        assert curves.inverse_quadratic(0.5) == pytest.approx(0.75)

    def test_decelerates(self):
        # inverse quadratic should be above linear for values in (0, 1)
        assert curves.inverse_quadratic(0.5) > curves.linear(0.5)


class TestLogistic:
    def test_midpoint_default(self):
        # At the default midpoint (0.5), output should be ~0.5
        assert curves.logistic(0.5) == pytest.approx(0.5)

    def test_zero_below_half(self):
        assert curves.logistic(0.0) < 0.5

    def test_one_above_half(self):
        assert curves.logistic(1.0) > 0.5

    def test_monotonically_increasing(self):
        values = [curves.logistic(v / 10.0) for v in range(11)]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1]

    def test_custom_midpoint(self):
        assert curves.logistic(0.3, midpoint=0.3) == pytest.approx(0.5)

    def test_low_steepness_is_flatter(self):
        # With lower steepness, values near 0 should be closer to 0.5
        steep = curves.logistic(0.0, steepness=10.0)
        flat = curves.logistic(0.0, steepness=1.0)
        assert flat > steep  # flatter curve stays closer to 0.5

    def test_high_steepness_near_step(self):
        # Very high steepness should act like a step function
        assert curves.logistic(0.49, steepness=100.0) < 0.3
        assert curves.logistic(0.51, steepness=100.0) > 0.7


class TestExponential:
    def test_zero(self):
        assert curves.exponential(0.0) == pytest.approx(0.0)

    def test_one(self):
        assert curves.exponential(1.0) == pytest.approx(1.0)

    def test_normalized_zero_and_one_for_various_bases(self):
        for base in [1.5, 2.0, 5.0, 10.0]:
            assert curves.exponential(0.0, base=base) == pytest.approx(0.0)
            assert curves.exponential(1.0, base=base) == pytest.approx(1.0)

    def test_monotonically_increasing(self):
        values = [curves.exponential(v / 10.0) for v in range(11)]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1]

    def test_higher_base_more_aggressive(self):
        # Higher base should produce lower values in the middle (curve bows down more)
        low_base = curves.exponential(0.5, base=2.0)
        high_base = curves.exponential(0.5, base=10.0)
        assert high_base < low_base


class TestSmoothstep:
    def test_zero(self):
        assert curves.smoothstep(0.0) == 0.0

    def test_one(self):
        assert curves.smoothstep(1.0) == 1.0

    def test_midpoint(self):
        assert curves.smoothstep(0.5) == pytest.approx(0.5)

    def test_clamps_below_zero(self):
        assert curves.smoothstep(-1.0) == 0.0

    def test_clamps_above_one(self):
        assert curves.smoothstep(2.0) == 1.0

    def test_monotonically_increasing(self):
        values = [curves.smoothstep(v / 10.0) for v in range(11)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]

    def test_ease_in_ease_out(self):
        # Derivative at 0 and 1 should be 0 (ease in/out), meaning values near
        # boundaries should be very close to the boundary values
        assert curves.smoothstep(0.01) < 0.01  # slower than linear near 0
        assert curves.smoothstep(0.99) > 0.99  # slower than linear near 1
