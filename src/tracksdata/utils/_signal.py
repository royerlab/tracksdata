from psygnal import Signal


def is_signal_on(sig: Signal) -> bool:
    """Check if a signal is connected and not blocked."""
    return len(sig._slots) > 0 and not sig._is_blocked
