# Setting up conda environment
From any directory, run
```bash
/path/to/create_env.sh <the environment name (default: ditto)>
```
to create a conda environment all dependencies installed.

* For example, running `./conda/create_env.sh venv` from the repository root will create a conda environment named "venv".

* You can also use other alternatives for conda such as
[micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) as
`CONDA=micromamba ./conda/create_env.sh venv`.

Activate the generated conda environment.
```bash
conda activate <the environment name>
```
