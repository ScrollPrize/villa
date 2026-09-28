import argparse
import os
import site
import sys
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


def agreement_path() -> Path:
    """Where the licence acceptance is recorded: the user's config directory.

    ``VESUVIUS_AGREEMENT_FILE`` overrides it. It used to be written next to
    this module, which for the recommended editable install (``uv sync``) is
    inside the git checkout and shows up as an untracked file in every
    contributor's ``git status``, and for a wheel install is lost on upgrade.
    """

    override = os.environ.get("VESUVIUS_AGREEMENT_FILE")
    if override:
        return Path(override).expanduser()
    config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return Path(config_home) / "vesuvius" / "agreement.txt"


def save_agreement():
    agreement_file_path = agreement_path()
    agreement_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agreement_file_path, "w+") as file:
        file.write("yes")
    print(f"Agreement saved to {agreement_file_path}")


def display_terms_and_conditions(accept_terms):
    terms = """
    ## LICENSE

    All Vesuvius Challenge datasets are released under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt), except Scrolls 1–4 and Fragments 1–6, which are bound by the following terms:

    By registering for downloads from the EduceLab-Scrolls Dataset and Vesuvius Challenge Discord Server, I agree to license the data from Vesuvius Challenge* under the following licensing terms:
    - I agree that all points below apply to both the EduceLab-Scrolls Dataset downloaded from our webserver, as well as any data (e.g. text, pictures, code) downloaded from the Vesuvius Challenge Discord Server.
    - I will not redistribute the data without the written approval of Vesuvius Challenge (if I am working in a team, every team member will sign this form separately).
    Vesuvius Challenge reserves the right to use in any way, including in an academic or other publication, all submissions or results produced from this dataset.
    - I will not make public (outside of Discord) any revelation of hidden text (or associated code) without the written approval of Vesuvius Challenge.
    - I agree all publications and presentations resulting from any use of the EduceLab-Scrolls Dataset must cite use of the EduceLab-Scrolls Dataset as follows:
    - In any published abstract, I will cite "EduceLab-Scrolls" as the source of the data in the abstract.
    - In any published manuscripts using data from EduceLab-Scrolls, I will reference the following paper: Parsons, S., Parker, C. S., Chapman, C., Hayashida, M., & Seales, W. B. (2023). EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT. ArXiv [Cs.CV]. https://doi.org/10.48550/arXiv.2304.02084.
    - I will include language similar to the following in the methods section of my manuscripts in order to accurately acknowledge the data source: "Data used in the preparation of this article were obtained from the EduceLab-Scrolls dataset [above citation]."
    - I understand that all submissions will be reviewed by the Vesuvius Challenge Review Team, and that prizes will be awarded as the sole discretion of Vesuvius Challenge.
    When I post or upload data in Discord (e.g. text, pictures, code), I agree to license it to other participants under these same terms.

    * All EduceLab-Scrolls data is copyrighted by EduceLab/The University of Kentucky. Permission to use the data linked herein according to the terms outlined above is granted to Vesuvius Challenge.
    """
    print(terms)

    if not accept_terms:
        print(
            "You must accept the terms and conditions to use this package. Run `$ vesuvius.accept_terms --yes` to accept."
        )
        sys.exit(1)
    else:
        save_agreement()
        print("Thank you for accepting the terms and conditions.")


def main():
    parser = argparse.ArgumentParser(
        description="Script to accept Terms and Conditions"
    )
    parser.add_argument(
        "--yes", action="store_true", help="Automatically accept terms and conditions"
    )

    args = parser.parse_args()

    display_terms_and_conditions(args.yes)


if __name__ == "__main__":
    main()
