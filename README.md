# PILOT: Physics-Informed Learning of Operators

Welcome to **PILOT**, a hands-on framework for training and benchmarking physics-informed neural operators.

While the documentation is still a work in progress, a detailed explanation of most modules in the source code can be found in `docs/PILOT.pdf`.

## Download and usage instructions
### Option 1: Clone github repository using `git clone`

In your machine's terminal, enter the folder in which you wish to download the repository and run the 

```bash
git clone https://github.com/NikTimes/PILOT.git
```

Continue by accessing the folder using your terminal by running: 

```bash
cd PILOT 
```

The PILOT framework was built using python version 3.12. While previous versions are probably compatible 
with the framework it is recommended that you build a new virtual environment containing python >= 3.12 to ensure compatibility. If you wish to download it in an already existing virtual environment, activate it and run: 

```bash
pip install -e .
```

If you wish to create a new virtual environment to install PILOT follow any of the following subchapters: 

#### Conda environment (Recommended)

If using Conda run the following command: 

```bash
conda create -n pilot python=3.12
```

Activate it:

```bash
conda activate pilot
```

Then install PILOT 

```bash
pip install -e .
```

#### Python venv: 

Create the environment:

```bash
python -m venv .venv
```

If using Linux or macOS activate environment using: 

```bash
source .venv/bin/activate
```

If using windows: 

```bash
.venv\Scripts\activate
```

Then install 

```bash
pip install -e .
```
#### Homebrew environment: 

Install pyenv:

```bash
brew install pyenv pyenv-virtualenv
```

Install a Python version:

```bash
pyenv install 3.12.0
```

Create a virtual environment:

```bash
pyenv virtualenv 3.12.0 pilot
```

Activate it:

```bash
pyenv activate pilot
```

Then install the project:

```bash
pip install -e .
```

## Experiments 

## Experiments

All experiments shown in `docs/PILOT.pdf` can be found in the `experiments/` directory.
Each file is named according to the physical system we aimed to train an operator for. If the 
file name contains the word `naive` that means the resulting operator was trained without using 
Physics Information. The opposite is true if it contains the word `PINNS`. 

Each experiment contains three files:

- Sweep configuration (where applicable)
- Operator trainer of best sweep configuration.
- Benchmarking of trained Operator. 

Any file named `*_sweep.py` contains examples of hyperparameter sweeps carried out using PILOT.

Note that in order to run these sweep files, the user must have an account with [Weights & Biases](https://wandb.ai/). Each sweep may take from several minutes to hours depending on the machine and configuration. As such, these files are primarily intended as code examples and templates for reproducible experimentation rather than as scripts to be executed directly without modification.

The configurations that yielded the best results after running `*_sweep.py` were subsequently used in separate `*.ipynb` training notebooks. These notebooks save the resulting model weights in the `/weights` directory.

Files named `*_benchmark.ipynb` contain full benchmarking workflows for evaluating the learned operator using PILOT and the saved weights in `/weights`. These notebooks are the recommended entry point for verifying that the codebase is functioning correctly. 

Note, that in order to use the jupyter notebooks in an interface such as `spyder` you need to ensure that you have the required packages to support it. In the case of `spyder` you can follow how to use notebook instructions here: https://pypi.org/project/spyder-notebook/
