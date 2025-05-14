import os

from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

default_num_processes = 8 if 'nnUNet_def_n_proc' not in os.environ else int(os.environ['nnUNet_def_n_proc'])

ANISO_THRESHOLD = 5  # determines when a sample is considered anisotropic (5 means that the spacing in the low
# resolution axis must be 5x as large as the next largest spacing), the default value was 3

default_n_proc_DA = get_allowed_n_proc_DA()
