from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from psygnal import Signal, SignalInstance


def _is_batched(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray | dict)


def reduce_node_added_events(
    event_args: Iterable[tuple[int, dict[str, Any]]],
) -> tuple[int | list[int], dict[str, Any] | list[dict[str, Any]]]:
    """
    Collapse a stream of ``node_added`` event args into a single batched event.

    A single event is returned unchanged as ``(node_id, attrs)``; multiple events
    are combined into ``(list_of_ids, list_of_attrs)``. Used as the reducer passed
    to ``SignalInstance.paused`` so listeners receive one batched emission.

    Parameters
    ----------
    event_args : Iterable[tuple[int, dict[str, Any]]]
        The ``(node_id, attrs)`` pairs collected while the signal was paused.

    Returns
    -------
    tuple[int | list[int], dict[str, Any] | list[dict[str, Any]]]
        Either a single ``(node_id, attrs)`` event or the batched
        ``(list_of_ids, list_of_attrs)`` form.
    """
    events = list(event_args)
    if len(events) == 1:
        return events[0]

    node_ids, attrs = zip(*events, strict=True)
    return list(node_ids), list(attrs)


def reduce_node_updated_events(
    event_args: Iterable[tuple[int, dict[str, Any], dict[str, Any]]],
) -> tuple[int | list[int], dict[str, Any] | list[dict[str, Any]], dict[str, Any] | list[dict[str, Any]]]:
    """
    Collapse a stream of ``node_updated`` event args into a single batched event.

    A single event is returned unchanged as ``(node_id, old_attrs, new_attrs)``;
    multiple events are combined into ``(list_of_ids, list_of_old, list_of_new)``.
    Used as the reducer passed to ``SignalInstance.paused``.

    Parameters
    ----------
    event_args : Iterable[tuple[int, dict[str, Any], dict[str, Any]]]
        The ``(node_id, old_attrs, new_attrs)`` triples collected while paused.

    Returns
    -------
    tuple[int | list[int], dict[str, Any] | list[dict[str, Any]], dict[str, Any] | list[dict[str, Any]]]
        Either a single event or the batched list form.
    """
    events = list(event_args)
    if len(events) == 1:
        return events[0]

    node_ids, old_attrs, new_attrs = zip(*events, strict=True)
    return list(node_ids), list(old_attrs), list(new_attrs)


def emit_node_added_events(
    sig: Signal | SignalInstance,
    event_args: Iterable[tuple[int, dict[str, Any]]],
) -> None:
    """
    Emit ``node_added`` events as a single batched emission.

    The signal is paused and reduced via :func:`reduce_node_added_events`, so
    connected slots receive one call: ``(node_id, attrs)`` for a single node or
    ``(list_of_ids, list_of_attrs)`` for many. No-op if there are no events or
    the signal has no active listeners.

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

    with sig.paused(reduce_node_added_events):
        for node_id, attrs in events:
            sig.emit(node_id, attrs)


def emit_node_updated_events(
    sig: Signal | SignalInstance,
    event_args: Iterable[tuple[int, dict[str, Any], dict[str, Any]]],
) -> None:
    """
    Emit ``node_updated`` events as a single batched emission.

    The signal is paused and reduced via :func:`reduce_node_updated_events`, so
    connected slots receive one call: ``(node_id, old_attrs, new_attrs)`` for a
    single node or the list form for many. No-op if there are no events or the
    signal has no active listeners.

    Parameters
    ----------
    sig : Signal | SignalInstance
        The ``node_updated`` signal to emit on.
    event_args : Iterable[tuple[int, dict[str, Any], dict[str, Any]]]
        The ``(node_id, old_attrs, new_attrs)`` triples to emit.
    """
    events = list(event_args)
    if len(events) == 0 or not is_signal_on(sig):
        return

    with sig.paused(reduce_node_updated_events):
        for node_id, old_attrs, new_attrs in events:
            sig.emit(node_id, old_attrs, new_attrs)


def emit_node_removed_events(
    sig: Signal | SignalInstance,
    event_args: Iterable[tuple[int, dict[str, Any]]],
) -> None:
    """
    Emit one ``node_removed`` event per removed node.

    Unlike :func:`emit_node_added_events` and :func:`emit_node_updated_events`,
    removal events are emitted individually rather than reduced into a single
    batched event: connected slots (e.g. spatial-filter and array invalidation)
    expect one ``(node_id, old_attrs)`` call per node. This helper centralises
    the per-node emission loop shared by the graph backends. No-op if the signal
    has no active listeners.

    Parameters
    ----------
    sig : Signal | SignalInstance
        The ``node_removed`` signal to emit on.
    event_args : Iterable[tuple[int, dict[str, Any]]]
        The ``(node_id, old_attrs)`` pairs to emit, one emission each.
    """
    if not is_signal_on(sig):
        return

    for node_id, old_attrs in event_args:
        sig.emit(node_id, old_attrs)


def iter_node_added_events(
    node_ids: int | Sequence[int],
    attrs: dict[str, Any] | Sequence[dict[str, Any]],
) -> Iterator[tuple[int, dict[str, Any]]]:
    """
    Normalise possibly-batched ``node_added`` payloads into individual events.

    Accepts either a single ``(node_id, attrs)`` or the batched
    ``(list_of_ids, list_of_attrs)`` form produced by
    :func:`reduce_node_added_events`, and yields one ``(node_id, attrs)`` pair per
    node. This lets slots handle both single and batched emissions uniformly.

    Parameters
    ----------
    node_ids : int | Sequence[int]
        A single node id or a sequence of ids.
    attrs : dict[str, Any] | Sequence[dict[str, Any]]
        The matching attributes; a single dict or a sequence of dicts.

    Yields
    ------
    tuple[int, dict[str, Any]]
        One ``(node_id, attrs)`` pair per node.

    Raises
    ------
    TypeError
        If the batched-ness of ``node_ids`` and ``attrs`` disagree.
    """
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
    """
    Normalise possibly-batched ``node_updated`` payloads into individual events.

    Accepts either a single ``(node_id, old_attrs, new_attrs)`` or the batched
    list form produced by :func:`reduce_node_updated_events`, and yields one
    ``(node_id, old_attrs, new_attrs)`` triple per node.

    Parameters
    ----------
    node_ids : int | Sequence[int]
        A single node id or a sequence of ids.
    old_attrs : dict[str, Any] | Sequence[dict[str, Any]]
        The previous attributes; a single dict or a sequence of dicts.
    new_attrs : dict[str, Any] | Sequence[dict[str, Any]]
        The new attributes; a single dict or a sequence of dicts.

    Yields
    ------
    tuple[int, dict[str, Any], dict[str, Any]]
        One ``(node_id, old_attrs, new_attrs)`` triple per node.

    Raises
    ------
    TypeError
        If the batched-ness of ``node_ids``, ``old_attrs`` and ``new_attrs`` disagree.
    """
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
