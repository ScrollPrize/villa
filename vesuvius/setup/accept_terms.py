import sys
import os
import site
import argparse

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
        - For editable installs, this returns the directory containing the vesuvius.py file
        - For standard installs, this returns the site-packages directory
    """
    # Try to find if we're installed in editable mode
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vesuvius_py_path = os.path.join(current_dir, 'vesuvius.py')
    
    # If vesuvius.py exists in the parent directory, we're in editable mode
    if os.path.exists(vesuvius_py_path):
        return current_dir
    
    # Otherwise, use the site-packages location
    if is_colab():
        install_path = site.getsitepackages()[0]
    else:
        install_path = site.getsitepackages()[-1]
    return install_path

def save_agreement():
    install_path = get_installation_path()
    agreement_file_path = os.path.join(install_path, 'vesuvius', 'setup', 'agreement.txt')
    
    with open(agreement_file_path, 'w+') as file:
        file.write("yes")
    print(f"Agreement saved to {agreement_file_path}")

def display_terms_and_conditions(accept_terms):
    terms = """
    ## LICENSE

    The PHerc. 172 (Scroll 5) dataset Â© 2024 by Scroll Prize and the Bodleian Library is licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt.

    All other Vesuvius Challenge datasets are bound by the following terms:

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
        print("You must accept the terms and conditions to use this package. Run `$ vesuvius.accept_terms --yes` to accept.")
        sys.exit(1)
    else:
        save_agreement()
        print("Thank you for accepting the terms and conditions.")

def main():
    parser = argparse.ArgumentParser(description='Script to accept Terms and Conditions')
    parser.add_argument('--yes', action='store_true', help='Automatically accept terms and conditions')
    
    args = parser.parse_args()
    
    display_terms_and_conditions(args.yes)

if __name__ == "__main__":
    main()
