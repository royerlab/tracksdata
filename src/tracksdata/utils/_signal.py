from collections.abc import Iterable
from typing import Any

from psygnal import Signal, SignalInstance


def is_signal_on(sig: Signal | SignalInstance) -> bool:
    """Check if a signal is connected and not blocked."""
    return len(sig._slots) > 0 and not sig._is_blocked


def emit_node_added_events(
    sig: Signal | SignalInstance,
    event_args: Iterable[tuple[int, dict[str, Any]]],
) -> None:
    """
    Emit a single batched ``node_added`` event.

    Connected slots always receive one call ``(list_of_ids, list_of_attrs)``,
    regardless of how many nodes were added. No-op if there are no events or the
    signal has no active listeners.

    Parameters
    ----------
    sig : Signal | SignalInstance
        The ``node_added`` signal to emit on.
    event_args : Iterable[tuple[int, dict[str, Any]]]
        The ``(node_id, attrs)`` pairs to emit.
    """
    events = list(event_args)
    if len(events) == 0 or not is_signal_on(sig):
        return

    node_ids, attrs = zip(*events, strict=True)
    sig.emit(list(node_ids), list(attrs))


def emit_node_updated_events(
    sig: Signal | SignalInstance,
    event_args: Iterable[tuple[int, dict[str, Any], dict[str, Any]]],
    changed_keys: set[str],
) -> None:
    """
    Emit a single batched ``node_updated`` event.

    Connected slots receive one call
    ``(list_of_ids, list_of_old_attrs, list_of_new_attrs, changed_keys)``. No-op
    if there are no events or the signal has no active listeners.

    Parameters
    ----------
    sig : Signal | SignalInstance
        The ``node_updated`` signal to emit on.
    event_args : Iterable[tuple[int, dict[str, Any], dict[str, Any]]]
        The ``(node_id, old_attrs, new_attrs)`` triples to emit.
    changed_keys : set[str]
        The attribute keys actually written by this update (uniform across the
        batch). Lets connectors skip work when none of the keys they track changed.
    """
    events = list(event_args)
    if len(events) == 0 or not is_signal_on(sig):
        return

    node_ids, old_attrs, new_attrs = zip(*events, strict=True)
    sig.emit(list(node_ids), list(old_attrs), list(new_attrs), set(changed_keys))


def emit_node_removed_events(
    sig: Signal | SignalInstance,
    event_args: Iterable[tuple[int, dict[str, Any]]],
) -> None:
    """
    Emit a single batched ``node_removed`` event.

    Connected slots always receive one call ``(list_of_ids, list_of_attrs)``,
    regardless of how many nodes were removed. No-op if there are no events or
    the signal has no active listeners.

    Parameters
    ----------
    sig : Signal | SignalInstance
        The ``node_removed`` signal to emit on.
    event_args : Iterable[tuple[int, dict[str, Any]]]
        The ``(node_id, old_attrs)`` pairs to emit.
    """
    events = list(event_args)
    if len(events) == 0 or not is_signal_on(sig):
        return

    node_ids, attrs = zip(*events, strict=True)
    sig.emit(list(node_ids), list(attrs))
