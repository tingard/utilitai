"""A warehouse inspection robot, deciding with utilitai and acting with btreeny.

Run with ``uv run python examples/warehouse_inspection.py``. Set ``LOG_LEVEL``
to DEBUG to see every score behind every decision.

The robot has four things it can do:

``patrol``
    Walk an aisle looking for problems. Takes several ticks.
``recharge``
    Return to the charger. Wants attention more and more as the battery drains.
``investigate``
    Something looked wrong - go and take a closer look. Interrupts patrolling.
``evacuate``
    The fire alarm went off. Interrupts everything.

The interesting part is the interrupt hierarchy. ``investigate`` and
``evacuate`` both preempt the routine duties, ``evacuate`` preempts
``investigate``, and ``investigate`` must *not* preempt an evacuation already
under way. None of that is special-cased: it falls out of two ordinary scoring
functions and one consideration.
"""

import logging
import os
from dataclasses import dataclass

import btreeny

from utilitai import ToConsider, curves
from utilitai.btreeny import Commitment, committed

logger = logging.getLogger("warehouse")


# ------------------------------------------------------------------------------
# The world
# ------------------------------------------------------------------------------


@dataclass
class Warehouse:
    """Everything the robot knows. This is both the utilitai context and the
    btreeny blackboard - they are the same object."""

    battery: float = 1.0
    """Remaining charge, 0 (flat) to 1 (full)."""
    anomaly: float = 0.0
    """Confidence that something needs a closer look, 0 to 1."""
    fire_alarm: bool = False
    aisles_left: int = 6
    at_charger: bool = False
    at_exit: bool = False
    current_action: str | None = None
    """Whichever option the robot has committed to. Owned by `Commitment`."""


# ------------------------------------------------------------------------------
# Routine duties - what to do when nothing is on fire
# ------------------------------------------------------------------------------

duties: ToConsider[Warehouse] = ToConsider()


@duties.consideration
def battery_low(wh: Warehouse) -> float:
    """Rises smoothly as charge falls away."""
    return curves.smoothstep(curves.inverse_linear(wh.battery))


@duties.consideration
def work_remaining(wh: Warehouse) -> float | None:
    """Drops out entirely once every aisle has been walked."""
    if wh.aisles_left <= 0:
        return None
    return curves.clamped(wh.aisles_left / 6)


@duties.option("patrol")
def patrol(wh: Warehouse, work_remaining: float, battery_low: float) -> float:
    # Keen to work, but less so the flatter the battery gets
    return curves.clamped(0.8 * work_remaining * curves.inverse_linear(battery_low))


@duties.option("recharge")
def recharge(wh: Warehouse, battery_low: float) -> float:
    # Quadratic so it stays out of the way until the battery genuinely matters,
    # then climbs hard
    return curves.quadratic(battery_low)


# If there is nothing to patrol and no need to charge, stand still
duties.constant_option("hold", 0.05)


# ------------------------------------------------------------------------------
# Interrupts - allowed to cut in on a committed duty
# ------------------------------------------------------------------------------

interrupts: ToConsider[Warehouse] = ToConsider()


@interrupts.consideration
def not_already_evacuating(wh: Warehouse) -> float | None:
    """The commitment lives on the blackboard, so scoring functions can read it.

    This is what stops ``investigate`` from yanking the robot out of an
    evacuation that is already under way - it short-circuits itself rather than
    needing the framework to rank interrupts.
    """
    return None if wh.current_action == "evacuate" else 1.0


@interrupts.option("evacuate")
def evacuate(wh: Warehouse) -> float | None:
    # Scores 1.0 or not at all, so it outranks anything else that can interrupt
    return 1.0 if wh.fire_alarm else None


@interrupts.option("investigate")
def investigate(wh: Warehouse, not_already_evacuating: float) -> float | None:
    if wh.anomaly < 0.4:
        return None  # not worth breaking off a patrol for
    # Capped below evacuate's 1.0: the ordering *is* the scores
    return curves.clamped(0.6 * curves.smoothstep(wh.anomaly))


# ------------------------------------------------------------------------------
# Actions - what the robot actually does, once it has decided
# ------------------------------------------------------------------------------


@btreeny.action
def patrol_aisle(node_id: btreeny.IdType, steps: int = 3):
    """Walk one aisle. Multi-tick, so it is something worth committing to."""
    logger.debug("setting up patrol")
    walked = 0
    try:

        def tick(wh: Warehouse) -> btreeny.TreeStatus:
            nonlocal walked
            walked += 1
            wh.at_charger = wh.at_exit = False
            wh.battery = max(0.0, wh.battery - 0.04)
            if walked < steps:
                return btreeny.RUNNING
            wh.aisles_left -= 1
            logger.info("aisle swept, %d left", wh.aisles_left)
            return btreeny.SUCCESS

        yield tick
    finally:
        logger.debug("tearing down patrol")


@btreeny.action
def go_and_charge(node_id: btreeny.IdType):
    try:

        def tick(wh: Warehouse) -> btreeny.TreeStatus:
            if not wh.at_charger:
                wh.at_charger = True
                wh.at_exit = False
                logger.info("docking at the charger")
                return btreeny.RUNNING
            wh.battery = min(1.0, wh.battery + 0.15)
            if wh.battery < 0.9:
                return btreeny.RUNNING
            logger.info("charged to %.0f%%", wh.battery * 100)
            return btreeny.SUCCESS

        yield tick
    finally:
        logger.debug("leaving the charger")


@btreeny.action
def take_a_closer_look(node_id: btreeny.IdType, steps: int = 3):
    scanned = 0
    try:

        def tick(wh: Warehouse) -> btreeny.TreeStatus:
            nonlocal scanned
            scanned += 1
            wh.at_charger = wh.at_exit = False
            wh.battery = max(0.0, wh.battery - 0.03)
            if scanned < steps:
                return btreeny.RUNNING
            logger.info("anomaly checked out, nothing wrong")
            wh.anomaly = 0.0
            return btreeny.SUCCESS

        yield tick
    finally:
        logger.debug("ending inspection")


@btreeny.action
def head_for_the_exit(node_id: btreeny.IdType, steps: int = 2):
    travelled = 0
    try:

        def tick(wh: Warehouse) -> btreeny.TreeStatus:
            nonlocal travelled
            wh.at_charger = False
            if not wh.at_exit:
                travelled += 1
                wh.battery = max(0.0, wh.battery - 0.05)
                if travelled < steps:
                    return btreeny.RUNNING
                wh.at_exit = True
                logger.info("clear of the building")
                return btreeny.RUNNING
            # Wait outside. The action decides for itself when it is done -
            # which is what releases the commitment.
            if wh.fire_alarm:
                return btreeny.RUNNING
            logger.info("all clear, going back in")
            return btreeny.SUCCESS

        yield tick
    finally:
        logger.debug("ending evacuation")


@btreeny.simple_action
def stand_still(wh: Warehouse) -> btreeny.TreeStatus:
    return btreeny.SUCCESS


TREES = {
    "patrol": patrol_aisle,
    "recharge": go_and_charge,
    "investigate": take_a_closer_look,
    "evacuate": head_for_the_exit,
    "hold": stand_still,
}


# ------------------------------------------------------------------------------
# Wiring it together
# ------------------------------------------------------------------------------


class RecordingCommitment(Commitment[Warehouse]):
    """Remembers what it last chose, purely so the printout can show the action
    that ran on a tick - the latch itself is cleared the moment a subtree
    finishes, so reading it afterwards would show ``None``."""

    last_choice: str | None = None

    def key_for(self, context: Warehouse) -> str:
        self.last_choice = super().key_for(context)
        return self.last_choice


def build_tree() -> tuple[btreeny.TreeNode[Warehouse], RecordingCommitment]:
    commitment = RecordingCommitment.on_attribute(
        duties,
        preempt=interrupts,
        # key_fn runs inside the tick, so it must never raise
        default="hold",
        # Abandon a duty early if it stops being worth doing at all. Interrupts
        # do not need this - they cut in regardless.
        abandon_below=0.1,
    )
    return committed(commitment, TREES), commitment


# A scripted shift, so the run is reproducible
EVENTS: dict[int, tuple[str, str]] = {
    8: ("anomaly", "a pallet looks out of place"),
    10: ("fire_alarm_on", "FIRE ALARM"),
    16: ("fire_alarm_off", "alarm cleared"),
}


def apply_event(wh: Warehouse, name: str) -> None:
    match name:
        case "anomaly":
            wh.anomaly = 0.7
        case "fire_alarm_on":
            wh.fire_alarm = True
        case "fire_alarm_off":
            wh.fire_alarm = False


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(), format="        %(message)s"
    )
    wh = Warehouse()

    print(f"{'tick':>4}  {'battery':>7}  {'anomaly':>7}  {'alarm':>5}  action")
    print("-" * 52)

    tree, commitment = build_tree()
    with tree as tick:
        for t in range(26):
            if (event := EVENTS.get(t)) is not None:
                apply_event(wh, event[0])
                print(f"{'':>4}  >>> {event[1]}")

            tick(wh)

            print(
                f"{t:>4}  {wh.battery:>6.0%}   {wh.anomaly:>6.1f}   "
                f"{'YES' if wh.fire_alarm else '-':>5}  {commitment.last_choice}"
            )

    print("-" * 52)
    print(f"Shift over: {wh.aisles_left} aisles left, battery {wh.battery:.0%}")


if __name__ == "__main__":
    main()
