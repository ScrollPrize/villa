"""Settings removed from configurations that older run files and checkpoints still record."""

ANY = object()  # a retired setting every value of which is supported


def retire(values, retired, what):
    """``values`` without the ``retired`` settings ({name: supported value or ANY}); refuses any other value."""
    for key, kept in retired.items():
        if key in values and kept is not ANY and values[key] != kept:
            raise ValueError(f'The {what} setting {key}={values[key]!r} no longer exists (only {kept!r} is supported)')
    return {k: v for k, v in values.items() if k not in retired}
