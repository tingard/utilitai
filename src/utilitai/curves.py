"""Response curves: small functions which shape a raw value into a score.

Scoring functions are plain Python, so curves are composed by calling them::

    @things.option("eat food")
    def eat_food(ctx: Context) -> float:
        return curves.clamped(curves.exponential(ctx.hunger) * curves.is_gt_zero(ctx.food))
"""

import math

__all__ = [
    "exponential",
    "inverse_linear",
    "inverse_quadratic",
    "is_gt_zero",
    "is_le_zero",
    "linear",
    "logistic",
    "quadratic",
    "smoothstep",
]


def linear(val: float) -> float:
    """Linear (identity) response curve. Returns the input unchanged."""
    return val


def inverse_linear(val: float) -> float:
    """Inverse linear response curve. Returns ``1 - val``."""
    return 1 - val


def quadratic(val: float) -> float:
    """Quadratic (polynomial power of 2) response curve. Accelerates quickly."""
    return val * val


def inverse_quadratic(val: float) -> float:
    """Inverse quadratic response curve. Decelerates quickly."""
    return 1 - (1 - val) * (1 - val)


def logistic(val: float, midpoint: float = 0.5, steepness: float = 10.0) -> float:
    """Logistic (sigmoid) response curve. Creates an S-shaped curve that
    transitions sharply around the midpoint.

    Parameters
    ----------
    val : float
        Input value, typically in [0, 1].
    midpoint : float
        The input value at which the output is ~0.5.
    steepness : float
        How steep the transition is around the midpoint.
    """
    z = steepness * (val - midpoint)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def exponential(val: float, base: float = 2.0) -> float:
    """Exponential response curve. Strongly favours high input values.

    The output is normalised so that f(0)=0 and f(1)=1 for any base > 1.

    Parameters
    ----------
    val : float
        Input value, typically in [0, 1].
    base : float
        Controls how aggressively the curve rises (must be > 1).

    Raises
    ------
    ValueError
        If *base* is not greater than one.
    """
    if base <= 1.0:
        raise ValueError(f"base must be greater than one, got {base}")
    return (base**val - 1.0) / (base - 1.0)


def smoothstep(val: float) -> float:
    """Hermite smoothstep response curve (3t² - 2t³).

    Provides a smooth ease-in / ease-out transition between 0 and 1,
    useful when you want a softer version of a linear ramp.

    Unlike the other curves, the input is clamped to [0, 1] - outside that
    range the polynomial turns back on itself and stops being monotonic.
    """
    val = max(0.0, min(1.0, val))
    return val * val * (3.0 - 2.0 * val)


def is_gt_zero(val: float) -> float:
    """Step function which returns one if the value is greater than
    zero else zero.
    """
    return 1.0 if val > 0 else 0.0


def is_le_zero(val: float) -> float:
    """Step function which returns one if the value is less than or
    equal to zero, else zero.
    """
    return 1.0 if val <= 0 else 0.0


def clamped(val: float) -> float:
    """Hard-clamp a value to [0, 1]. Bounding utility scores between
    0 and 1 is recommended to aid composition.
    """
    return min(1.0, max(0.0, val))
