#!/bin/bash

##------------------------##
#### PYTHON Environment ####
##------------------------##
### See: https://docs.alliancecan.ca/wiki/Python#Python_version_supported

# Environment to create
ENV_NAME="ENV_SKYGAN"

# Prevent error when SLURM_TMPDIR is not defined.
if [ ! -d "$SLURM_TMPDIR" ]; then
    SLURM_TMPDIR=$(pwd)
fi

# Purge existing environment
if [ -d "$SLURM_TMPDIR/$ENV_NAME" ]; then
    read -t 10 -p "Do you want delete the environment $SLURM_TMPDIR/$ENV_NAME? (yes/no) " answer
    answer=${answer:-n}
    case $answer in
        [Yy]* ) rm -rf $SLURM_TMPDIR/$ENV_NAME && echo -e "\n Deleted $SLURM_TMPDIR/$ENV_NAME";;
        [Nn]* ) echo -e "\n Skipping deletion of $SLURM_TMPDIR/$ENV_NAME";;
        * ) echo "Please answer yes or no.";;
    esac
fi

# Clear modules
module --force purge

# Load modules
# See https://docs.alliancecan.ca/wiki/Available_software
# or use 'module spider <your_package>'
PYTHON_VERSION="3.9"
PYTHON="python$PYTHON_VERSION"
module load StdEnv/2020 gcc cuda/11.8.0 opencv python/3.9.6
# Warning!
# Python3.9 is not compatible with StdEnv/2023

if [ $? == 1 ]; then
    echo "Environment base modules failed to be loaded."
    exit
fi

# Create the environment
virtualenv --no-download $SLURM_TMPDIR/$ENV_NAME

# Activate the environment
source $SLURM_TMPDIR/$ENV_NAME/bin/activate

# Prevent python installs outside of an environment
export PIP_REQUIRE_VIRTUALENV=1

# Upgrade pip
$PYTHON -m pip install --no-index --upgrade pip

# 1. Fastest & Most Reliable Source
# Install dependencies (internal; --no-index searches computecanada)
# For available python wheels, see https://docs.alliancecan.ca/wiki/Available_Python_wheels
# DANGER! Names don't necessarily match those of wheels on PYPI!
$PYTHON -m pip install --no-index -r ./.drac/requirements_python_internal.txt

# 2. For missing, but dependencies already installed
# Install dependencies (external; use pypi and skip dependencies)
$PYTHON -m pip install --no-deps --upgrade -r ./.drac/requirements_python_external_w_internalDeps.txt

# 3. Worst case
# Install dependencies (external; use pypi)
$PYTHON -m pip install --upgrade -r ./.drac/requirements_python_external.txt

# Install custom packages
cd libs
# Use SkyLibs branch main (if possible)
$PYTHON -m pip install --compile Parametric_SkyModels/

# Save modules
module save ${ENV_NAME}_modules

# Deactivate the environment
deactivate
