# Fibers Dataset

This project generates a fibers dataset from skeletons for volumetric annotation of Herculaneum papyri.

## Installation

To set up the environment, run the following commands:

```bash
conda env create -f environment.yml
conda activate fibers-dataset
vesuvius.accept_terms --yes
```

Copy your WebKnossos Auth Token into a `token.txt` file in this folder.

## Example Usage
```bash
python fibers-dataset-generator.py --nml_path fibers_s5_06500z_02000y_04000x_500_v03.nml --output_folder output
python hz-vt-generator.py --nml_path fibers_s5_06500z_02000y_04000x_500_v03.nml --output_folder output --radius 2 --workers 1
```

Select an appropriate number of workers considering your computer memory, number of processors and the annotation size.

The example annotation was done by Elian Rafael Dal Prá


