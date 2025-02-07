# Setting up conda environment
From any directory, run
```bash
/path/to/create_env.sh
```
to create a conda environment with all dependencies installed.

## Full Usage
```
Usage: ./conda/create_env.sh [options]
Options:
  -n, --name      Set the environment name (default: ditto)
  -v, --version   Specify the version of Ditto to install (default: nightly from remote or from source)
  -c, --conda     Specify the conda executable to use (default: conda)
  -r, --remote    Install Ditto from the remote repository (which is the default behavior when not attached to a terminal)
  -e, --editable  Install Ditto in editable mode (only when installing from script directory)
  -h, --help      Display this help message and exit
```

* For example, running `./conda/create_env.sh -n venv` from the repository root will create a conda environment named "venv".

* You can also use other alternatives for conda such as
[micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) as
`./conda/create_env.sh -n venv -c micromamba`.

Then, activate the generated conda environment.
```bash
conda activate <the environment name>
```
