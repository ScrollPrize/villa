"""Setuptools command hooks for the Vesuvius package."""

from __future__ import annotations

import warnings

from setuptools.command.install import install


class CustomInstallCommand(install):
    """Display a post-install reminder to accept the package terms."""

    def run(self) -> None:  # type: ignore[override]
        super().run()
        message = """
        ============================================================
        Thank you for installing vesuvius!

        To complete the setup, please run the following command:

            vesuvius.accept_terms --yes

        This will display the terms and conditions to be accepted.
        ============================================================
        """
        warnings.warn(message, UserWarning)


__all__ = ["CustomInstallCommand"]
