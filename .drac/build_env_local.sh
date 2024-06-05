#!/bin/bash -l

echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@ Intializing @@@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"

### ------------------------------------------------------------------------------ ###
##-------------------------##
#### CONSTANTS (SET ME!) ####
##-------------------------##

ENV_NAME="ENV_SKYGAN"

# Prevent error when SLURM_TMPDIR is not defined.
if [ ! -d "$SLURM_TMPDIR" ]; then
    SLURM_TMPDIR=$(pwd)
fi


### ------------------------------------------------------------------------------ ###
##------------------------##
#### Export Env. Vars.   ####
##------------------------##

echo -e "\n--- Constants ---"
echo "(* required constant)"
echo "SLURM_JOB_ID* =" $SLURM_JOB_ID
echo "SLURM_JOB_NAME =" $SLURM_JOB_NAME
echo "SLURM_PROCID* =" $SLURM_PROCID
echo "SLURM_JOB_NODELIST = " $SLURM_JOB_NODELIST
echo "SLURM_NNODES* = " $SLURM_NNODES
echo "SLURM_JOB_NUM_NODES = " $SLURM_JOB_NUM_NODES
echo "SLURM_NTASKS = " $SLURM_NTASKS
echo "SLURM_NTASKS_PER_NODE = " $SLURM_NTASKS_PER_NODE
echo "SLURM_LOCALID* = " $SLURM_LOCALID
echo "SLURM_GTIDS = " $SLURM_GTIDS
echo -e "--- End Constants --- \n"


### ------------------------------------------------------------------------------ ###
##------------------------##
#### Virtual Environment ####
##------------------------##
# Makes a local copy of the environment for faster operation
echo -e "--- Creating local python environment ---\n"

# DANGER!
# If an internet connection is required, the
# environment must be built using ./build_env.sh
# on the login node!
if [ ! -d $ENV_NAME ]; then
    echo -e "--- No global environment found ---\n"
    echo -e "--- Creating a new python environment ---\n"
    ./build_env.sh
    echo -e "--- New python environment created!re
     ---\n"
fi

# Restore modules
module restore ${ENV_NAME}_modules

# Prevent error when SLURM_TMPDIR is not defined.
if [ -d $SLURM_TMPDIR ]; then
    LOCAL_ENV="${SLURM_TMPDIR}/${ENV_NAME}_LOCAL"

    # Create local copy of environment
    if [ $SLURM_LOCALID -eq "0" ]; then
        # Activate the environment
        source $ENV_NAME/bin/activate

        echo "Cloning $ENV_NAME to SLURM_TMPDIR ($SLURM_TMPDIR)"
        if [ -d $LOCAL_ENV ]; then
            rm -rf $LOCAL_ENV # Remove the old environment
        fi
        # Clone the new environment
        time virtualenv-clone $ENV_NAME $LOCAL_ENV

        # Deactivate the old environment
        deactivate
    fi
else
    LOCAL_ENV=$ENV_NAME
fi

echo -e "\n--- Creating local python environment -- Done!\n"
# Note! Activate the environment AFTER barrier!


### ------------------------------------------------------------------------------ ###
##------------------------##
#### Data to local storage ####
##------------------------##
# Note! Each task will add to the local nodes cache, therefore cloning data is not necessary.

# None


### ------------------------------------------------------------------------------ ###
##------------------------##
#### WAIT               ####
##------------------------##
echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@ Run waiting @@@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"

FILE_LOCK=".drac_locks/lock_${SLURM_JOB_ID}_${SLURM_JOB_NAME}.txt"
echo "Waiting for all tasks to start..."
echo $SLURM_PROCID >> $FILE_LOCK
until [ $(cat $FILE_LOCK | wc -l) -ge $SLURM_NTASKS ]
do
    sleep 1
done
echo "$SLURM_PROCID Barrier Done!"
echo "$SLURM_PROCID Barrier Done!" >> $FILE_LOCK

### ------------------------------------------------------------------------------ ###


source ${LOCAL_ENV}/bin/activate
echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@@@ Intialization Complete @@@@@@@@@@@@@@@@@@@@"
date
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import sys; print(f'Python path: {sys.executable}')"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"
echo ""

### ------------------------------------------------------------------------------ ###
