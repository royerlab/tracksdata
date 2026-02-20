from psygnal import Signal, SignalInstance


def is_signal_on(sig: Signal | SignalInstance) -> bool:
    """Check if a signal is connected and not blocked."""
    return len(sig._slots) > 0 and not sig._is_blocked
