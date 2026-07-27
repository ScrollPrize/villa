"""An ArgumentParser that accepts --foo-bar and --foo_bar interchangeably.

The CLIs in this repo use both conventions, and the split runs between stages of the same
pipeline: blend_logits takes --chunk_size and --num_workers, while finalize_outputs takes
--chunk-size and --num-workers for the same two parameters. predict is mixed internally
too. Whichever a user types first, half the time they get "unrecognized arguments".

Rather than hand-maintaining an alias on every affected flag, this derives the alternate
spelling automatically: any long option containing '-' also answers to '_', and vice
versa. Adding a flag later needs no thought.

    parser = HyphenUnderscoreParser(description=...)
    parser.add_argument('--chunk-size', ...)   # --chunk_size works too

The auto-derived spellings are hidden from --help, so the help text still shows one
canonical name per flag rather than doubling in width.
"""

from __future__ import annotations

import argparse


def _alternate(option: str) -> str | None:
    """--foo-bar -> --foo_bar, --foo_bar -> --foo-bar, or None if there is nothing to swap."""
    if not option.startswith("--"):
        return None                      # short options: -v has no separator to swap
    body = option[2:]
    if "-" in body:
        alt = body.replace("-", "_")
    elif "_" in body:
        alt = body.replace("_", "-")
    else:
        return None
    return f"--{alt}"


class _HideAutoAliases(argparse.HelpFormatter):
    """Show only the spellings that were declared, not the derived ones."""

    def _format_action_invocation(self, action):
        auto = getattr(action, "_auto_aliases", ())
        if not auto:
            return super()._format_action_invocation(action)
        declared = [o for o in action.option_strings if o not in auto]
        # Swap in the declared-only list for the duration of the call; argparse reads
        # option_strings directly and there is no cleaner seam than this.
        original, action.option_strings = action.option_strings, declared
        try:
            return super()._format_action_invocation(action)
        finally:
            action.option_strings = original


class HyphenUnderscoreParser(argparse.ArgumentParser):
    """ArgumentParser that also answers to the opposite separator on every long option."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", _HideAutoAliases)
        super().__init__(*args, **kwargs)

    def _release_derived(self, option: str) -> None:
        """Give up a derived spelling so an explicit declaration can claim it.

        A declared name always beats a derived one. Without this, adding --a_b and then
        --a-b would fail: the first flag's auto-alias has already taken the second's real
        name, and argparse raises on the duplicate.
        """
        owner = self._option_string_actions.get(option)
        if owner is None or option not in getattr(owner, "_auto_aliases", ()):
            return
        owner.option_strings = [o for o in owner.option_strings if o != option]
        owner._auto_aliases = tuple(o for o in owner._auto_aliases if o != option)
        del self._option_string_actions[option]

    def add_argument(self, *args, **kwargs):
        for opt in args:
            if isinstance(opt, str) and opt.startswith("-"):
                self._release_derived(opt)

        extra = []
        for opt in args:
            alt = _alternate(opt) if isinstance(opt, str) else None
            # Skip anything already spoken for: a flag that declares both spellings
            # explicitly, or a name another flag has taken. Registering a duplicate
            # raises, and a CLI that will not construct is worse than a missing alias.
            if alt and alt not in args and alt not in extra \
                    and alt not in self._option_string_actions:
                extra.append(alt)
        action = super().add_argument(*args, *extra, **kwargs)
        if extra:
            action._auto_aliases = tuple(extra)
        return action
