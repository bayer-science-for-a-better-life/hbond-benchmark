![Tests](https://github.com/bayer-science-for-a-better-life/hbond-benchmark/actions/workflows/python-package-conda.yml/badge.svg) 

# hbond-benchmark 

This is a fork of the [molnet-gometric-lightning](https://github.com/bayer-science-for-a-better-life/molnet-geometric-lightning) repository modified to add possible H-donor-acceptor interactions as edges to molecular graphs.
Large parts of this code are borrowed from PyTorch Geometric and OGB examples, therefore this package is available under the same license (MIT).

## Why?

Molecular graphs used for training graph neural networks typically use covalent bonds as graph edges.
However, for tasks such as solubility, intra-molecular forces can play a role.

## Installation

After cloning this repo, you should be able to install with:

```conda env create```

Note: depending on your hardware, you may need to install the CUDA toolkit as well.
For instance, you might have to add a line `- cudatoolkit=10.2` to `environment.yml`.

## Example Usage

The following will train 5 models on the `bbbp` dataset with the default parameters.
The models will be stored in `example_models/`, and the data will be downloaded to `datasets/`.

```shell script
 python hbond_benchmark/train.py --default_root_dir=example_model/ --dataset_name=bbbp --dataset_root=datasets/ --gpus=1 --max_epochs=100 --n_runs=5 
```

Replace the directories to your liking, and `bbbp` with any name from MoleculeNet, for example `tox21`, `muv`, `hiv`, `pcba`, `bace`, `esol`.

## Model evaluation

Validation curves and test set performance are logged to `default_root_dir`.
Start a Tensorboard server with `default_root_dir` as the log directory.
From the above example, something like:

```shell script
tensorboard --logdir=/full/path/to/example_model/
```