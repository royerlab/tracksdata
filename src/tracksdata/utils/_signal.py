from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from psygnal import Signal, SignalInstance


def _is_batched(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray | dict)


def reduce_node_added_events(
    event_args: Iterable[tuple[int, dict[str, Any]]],
) -> tuple[int | list[int], dict[str, Any] | list[dict[str, Any]]]:
    events = list(event_args)
    if len(events) == 1:
        return events[0]

    node_ids, attrs = zip(*events, strict=True)
    return list(node_ids), list(attrs)


def reduce_node_updated_events(
    event_args: Iterable[tuple[int, dict[str, Any], dict[str, Any]]],
) -> tuple[int | list[int], dict[str, Any] | list[dict[str, Any]], dict[str, Any] | list[dict[str, Any]]]:
    events = list(event_args)
    if len(events) == 1:
        return events[0]

    node_ids, old_attrs, new_attrs = zip(*events, strict=True)
    return list(node_ids), list(old_attrs), list(new_attrs)


def emit_node_added_events(
    sig: Signal | SignalInstance,
    event_args: Iterable[tuple[int, dict[str, Any]]],
) -> None:
    events = list(event_args)
    if len(events) == 0 or not is_signal_on(sig):
        return

    with sig.paused(reduce_node_added_events):
        for node_id, attrs in events:
            sig.emit(node_id, attrs)


def emit_node_updated_events(
    sig: Signal | SignalInstance,
    event_args: Iterable[tuple[int, dict[str, Any], dict[str, Any]]],
) -> None:
    events = list(event_args)
    if len(events) == 0 or not is_signal_on(sig):
        return

    with sig.paused(reduce_node_updated_events):
        for node_id, old_attrs, new_attrs in events:
            sig.emit(node_id, old_attrs, new_attrs)


def iter_node_added_events(
    node_ids: int | Sequence[int],
    attrs: dict[str, Any] | Sequence[dict[str, Any]],
) -> Iterator[tuple[int, dict[str, Any]]]:
    if _is_batched(node_ids):
        if not _is_batched(attrs):
            raise TypeError("Expected a sequence of node attributes for batched node_added events.")

        yield from zip(node_ids, attrs, strict=True)
        return

    if _is_batched(attrs):
        raise TypeError("Expected a single node attributes dict for node_added events.")

    yield node_ids, attrs


def iter_node_updated_events(
    node_ids: int | Sequence[int],
    old_attrs: dict[str, Any] | Sequence[dict[str, Any]],
    new_attrs: dict[str, Any] | Sequence[dict[str, Any]],
) -> Iterator[tuple[int, dict[str, Any], dict[str, Any]]]:
    if _is_batched(node_ids):
        if not _is_batched(old_attrs) or not _is_batched(new_attrs):
            raise TypeError("Expected sequences of node attribute dicts for batched node_updated events.")

        yield from zip(node_ids, old_attrs, new_attrs, strict=True)
        return

    if _is_batched(old_attrs) or _is_batched(new_attrs):
        raise TypeError("Expected single node attribute dicts for node_updated events.")

    yield node_ids, old_attrs, new_attrs


def is_signal_on(sig: Signal | SignalInstance) -> bool:
    """Check if a signal is connected and not blocked."""
    return len(sig._slots) > 0 and not sig._is_blocked
