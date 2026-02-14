import logging
import sys

_logger = logging.getLogger(__name__)  # @CR: Unused — remove or use it.


def eps(
    *_a, **_k
):  # @CR: Accepting *args/**kwargs silently swallows caller mistakes. Since this is used as a response curve (Callable[[float], float]), consider a single ignored parameter like `(_val: float = 0.0)` to match the expected signature and let type checkers help.
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


# @CR: `logistic` and `exponential` have extra parameters beyond `val`, so they don't
# satisfy `Callable[[float], float]` as far as strict type checkers are concerned (even
# though defaults make them callable with one arg at runtime). Consider either providing
# factory functions (e.g. `def logistic(midpoint, steepness) -> Callable`) or using
# `functools.partial` helpers so the returned curves cleanly match the expected signature.
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
    import math  # @CR: Move `import math` to the top of the file.

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
    # @CR: The docstring says `base` must be > 1 but there's no validation.
    # base == 1 causes ZeroDivisionError; base < 1 silently produces wrong results.
    # Add a guard: `if base <= 1: raise ValueError(...)`.
    return (base**val - 1.0) / (base - 1.0)


def smoothstep(val: float) -> float:
    """Hermite smoothstep response curve (3t² - 2t³).

    Provides a smooth ease-in / ease-out transition between 0 and 1,
    useful when you want a softer version of a linear ramp.
    """
    val = max(
        0.0, min(1.0, val)
    )  # @CR: This is the only curve that clamps input to [0, 1]. The inconsistency could surprise users — consider either clamping all curves or documenting why this one is special.
    return val * val * (3.0 - 2.0 * val)


def is_gt_zero(
    val: float,
) -> float:  # @CR: The `is_` prefix reads like a boolean predicate. Consider `step_gt_zero` / `step_le_zero` (or similar) to signal that these return 0.0/1.0 floats.
    """Step function which returns one if the value is greater than
    zero else zero.
    """
    return 1.0 if val > 0 else 0.0


def is_le_zero(val: float) -> float:
    """Step function which returns one if the value is less than or
    equal to zero, else zero.
    """
    return 1.0 if val <= 0 else 0.0
