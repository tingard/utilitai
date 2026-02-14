import logging
import sys

_logger = logging.getLogger(__name__)


def eps(*_a, **_k):
    """Return machine epsilon, ignoring any arguments.

    This is useful as a near-zero floor value that avoids true zero in
    calculations (e.g. to prevent division-by-zero).
    """
    return sys.float_info.epsilon


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
    import math

    return 1.0 / (1.0 + math.exp(-steepness * (val - midpoint)))


def exponential(val: float, base: float = 2.0) -> float:
    """Exponential response curve. Strongly favours high input values.

    The output is normalised so that f(0)=0 and f(1)=1 for any base > 1.

    Parameters
    ----------
    val : float
        Input value, typically in [0, 1].
    base : float
        Controls how aggressively the curve rises (must be > 1).
    """
    return (base**val - 1.0) / (base - 1.0)


def smoothstep(val: float) -> float:
    """Hermite smoothstep response curve (3t² - 2t³).

    Provides a smooth ease-in / ease-out transition between 0 and 1,
    useful when you want a softer version of a linear ramp.
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
