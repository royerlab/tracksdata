from collections.abc import Sequence


def normalize_attr_keys(
    attr_keys: Sequence[str] | str | None,
    available_keys: Sequence[str],
    required_keys: Sequence[str],
) -> list[str]:
    """Normalize attribute key selection for subgraph creation.

    Parameters
    ----------
    attr_keys : Sequence[str] | str | None
        User-specified attribute keys. ``None`` selects all available keys.
    available_keys : Sequence[str]
        Full set of attribute keys to fall back on when ``attr_keys`` is ``None``.
    required_keys : Sequence[str]
        Keys that must always be included in the result.

    Returns
    -------
    list[str]
        Normalized list of attribute keys including all ``required_keys``.
    """
    if attr_keys is None:
        keys = list(available_keys)
    elif isinstance(attr_keys, str):
        keys = [attr_keys]
    else:
        keys = list(attr_keys)

    # Remove duplicates while preserving order
    keys = list(dict.fromkeys(keys))

    for key in required_keys:
        if key not in keys:
            keys.append(key)
    return keys
