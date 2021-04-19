# Moiré lattice generator
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TAdeJong/moire-lattice-generator/HEAD?urlpath=lab)
![build](https://github.com/TAdeJong/moire-lattice-generator/workflows/build/badge.svg)

Easily generate renders of lattices, moiré lattices and even quasi-lattices in Python.

Magic angle bilayer graphene was shown to be superconducting in 2018 [[1](https://doi.org/10.1038/nature26160)]. 
Despite the considerable hype concerning this discovery, little code exists to visualize the moiré pattern of two graphene layers.

To illustrate the work as done in our own paper ["Direct evidence for flat bands in twisted bilayer graphene from nano-ARPES"](https://www.nature.com/articles/s41567-020-01041-x) ([arXiv version here](https://arxiv.org/abs/2002.02289)), I created this repository.

This repository contains Python code to generate lattices with values reasonably like experimental (e.g. STM or TEM) results.
*If you are looking to generate more schematic like hexagonal lattice drawings, have a look at [@alexkaz2](https://github.com/alexkaz2)'s excellent [hexalattice](https://github.com/alexkaz2/hexalattice).*

- Trigonal, hexagonal, square lattices as well as quasi lattices can be created and combined.
- Linear distortions, such as a uniaxial strain along an arbitrary direction and rotations are supported. In addition, arbitrary deformations can be rendered (by passing a deformation tensor to the `shift` parameter).
- Edge dislocations can be added to the lattice as well.

A simple Python notebook to interactively generate visualizations of moire patterns of hexagonal lattices at different angles is included.

A high resolution resulting movie of varying twist angle can be found [here](https://www.youtube.com/watch?v=c4n1pMsDNaU).

Furthermore, the effect of uniaxial deformation along a single direction as described in e.g. ["Measuring local moiré lattice heterogeneity of twisted bilayer graphene
"](https://doi.org/10.1103/PhysRevResearch.3.013153) can be visualized.

Click the "Launch binder" button above to open an interactive notebook directly in your browser. (Note: performance in the mybinder environment is somewhat slow. Download and run the notebook on a local machine for better performance.)

![moire pattern](https://repository-images.githubusercontent.com/292806144/bd108280-7081-11eb-8e03-2018853e1909)

## Local installation 

```
git clone https://github.com/TAdeJong/moire-lattice-generator.git
cd moire-lattice-generator
pip install .
```

If you want to be able to play around with the functions themselves, consider using `pip install -e .`.

### Using conda:

- Clone the repository and open a terminal in the folder `moire-lattice-generator`
- Create the conda environment: `conda env create -f binder/environment.yml`
- Activate the environment: `conda activate moire-gen`
- Run jupyter lab: `jupyter lab` (or use a classical notebook if you prefer.)
- Update the following line in the notebook: `cluster = LocalCluster(n_workers=1, threads_per_worker=4, memory_limit='2GB')` to match your local machine. The default `cluster=LocalCluster()` typically works fine.

## Testing

This project uses `pytest` and `hypothesis` to run tests.

Install the test dependencies:

```
$ pip install -r requirements_test.txt
```
To run the tests:

```
$ pytest
```


# Acknowledgement

This work was financially supported by the [Netherlands Organisation for Scientific Research (NWO/OCW)](https://www.nwo.nl/en/science-enw) as part of the [Frontiers of Nanoscience (NanoFront)](https://www.universiteitleiden.nl/en/research/research-projects/science/frontiers-of-nanoscience-nanofront) program.
