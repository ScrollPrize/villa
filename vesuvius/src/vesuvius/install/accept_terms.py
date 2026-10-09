"""Installation-path helper.

The data licence used to be printed and "accepted" here (`vesuvius.accept_terms --yes`), but nothing
checked the record. It now lives in the library README under "Data licence".
"""

import site
from pathlib import Path


def is_colab():
    try:
        import google.colab

        return True
    except ImportError:
        return False


def get_installation_path():
    """
    Get the installation path of the package.

    Returns:
        str: The installation path.

    Note:
        - For editable installs, this returns the directory containing the package source.
        - For standard installs, this returns the site-packages directory
    """
    source_root = Path(__file__).resolve().parents[2]
    package_root = source_root / "vesuvius"
    if (package_root / "__init__.py").exists():
        return str(source_root)

    # Otherwise, use the site-packages location
    if is_colab():
        install_path = site.getsitepackages()[0]
    else:
        install_path = site.getsitepackages()[-1]
    return install_path
