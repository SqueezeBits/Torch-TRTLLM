# Setting up conda environment
From any directory, run
```bash
/path/to/create_env.sh <the-environment-name>
```
to create a conda environment all dependencies installed.

* For example, running `./conda/create_env.sh ditto` from the repository root will create a conda environment named "ditto".

* You can also use other alternatives for conda, for example, [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) like `CONDA=micromamba ./conda/create_env.sh ditto`.
