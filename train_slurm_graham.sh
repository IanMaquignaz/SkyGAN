#!/bin/bash -l

### See: https://slurm.schedmd.com/sbatch.html

# For Testing
# GLOBAL_COMMAND=" --TEST --limit_train_batches=5"
GLOBAL_COMMAND=""
CREATE_ENV_LOCAL=true
ENV_NAME="ENV_SKYGAN"

RESUME_FROM="logs/DeepClouds_17995032/DeepClouds/hpc_ckpt_3.ckpt"

sbatch << EOT
#!/bin/bash -l

### NAME ###
#SBATCH --job-name="Train_DeepClouds"

### TIME ###
### Time formats:
###     "minutes", "minutes:seconds"
###     "hours:minutes:seconds",
###     "days-hours", "days-hours:minutes"
###     "days-hours:minutes:seconds".
#SBATCH --time="8:0:0"

### ACCOUNT ###
#SBATCH --account=def-jlalonde
#SBATCH --mail-user=ian.maquignaz.1@ulaval.ca
#SBATCH --mail-type=ALL

### OUTPUT x=job-name, j=job-ID, n=node-number ###
#SBATCH --output="logs/sbatch_%x_id%j.txt"

### REQUEING ###
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@360

### NODES/TASKS ###
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

### HARDWARE ###
#SBATCH --cpus-per-task=16
#SBATCH --mem=124G
#SBATCH --gres="gpu:p100:1"

### OPTIONAL ###
## --test-only Validate the batch script and return an queue time estimate
###SBATCH --test-only


echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run Started @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"


### ------------------------------------------------------------------------------ ###
##-------------------------##
#### CONSTANTS (SET ME!) ####
##-------------------------##

# Echo all commands
set -x

# Prevent error when SLURM_TMPDIR is not defined.
if [ ! -d "\$SLURM_TMPDIR" ]; then
    SLURM_TMPDIR=\$(pwd)
fi

ENV_NAME=$ENV_NAME


### ------------------------------------------------------------------------------ ###
##------------------------##
#### Export Env. Vars.   ####
##------------------------##

echo -e "\n--- Constants ---"
echo "(* required constant)"
echo "SLURM_JOB_ID* =" \$SLURM_JOB_ID
echo "SLURM_JOB_NAME =" \$SLURM_JOB_NAME
echo "SLURM_PROCID* =" \$SLURM_PROCID
echo "SLURM_JOB_NODELIST = " \$SLURM_JOB_NODELIST
echo "SLURM_NNODES* = " \$SLURM_NNODES
echo "SLURM_JOB_NUM_NODES = " \$SLURM_JOB_NUM_NODES
echo "SLURM_NTASKS = " \$SLURM_NTASKS
echo "SLURM_NTASKS_PER_NODE = " \$SLURM_NTASKS_PER_NODE
echo "SLURM_LOCALID* = " \$SLURM_LOCALID
echo "SLURM_GTIDS = " \$SLURM_GTIDS
echo -e "--- End Constants --- \n"


### ------------------------------------------------------------------------------ ###
##------------------------##
#### Cleanup             ####
##------------------------##
# Old lock file may be present in the case of a reqeued job.

# Remove lock file
FILE_LOCK=".drac/lock_\${SLURM_JOB_ID}_\${SLURM_JOB_NAME}.txt"
if test -f \$FILE_LOCK ; then
    echo "Removing lock file: \$FILE_LOCK"
    rm \$FILE_LOCK
    echo "Removing lock file -- Done!"
fi


### ------------------------------------------------------------------------------ ###
##------------------------------##
#### Network storage to local ####
##------------------------------##
# Note! Each task will add to the local nodes cache, therefore cloning data is not necessary.

# None


### ------------------------------------------------------------------------------ ###
##------------------------##
#### Start               ####
##------------------------##
echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run Started @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"
TAG="DeepClouds_\${SLURM_JOB_ID}"

if $CREATE_ENV_LOCAL; then
    ENV_ACTIVATE_PATH=\${SLURM_TMPDIR}/\${ENV_NAME}_LOCAL/bin/activate
    echo "Creating local environment!"
    ENV_COMMAND=" \
        .drac/build_env_local.sh && \
        module restore \${ENV_NAME}_modules && \
        source \$ENV_ACTIVATE_PATH \
    "
else
    ENV_ACTIVATE_PATH=\${ENV_NAME}/bin/activate
    echo "SKIPPING creating local environment!"
    ENV_COMMAND=" \
        module restore \${ENV_NAME}_modules && \
        source \$ENV_ACTIVATE_PATH \
    "
fi
echo "Environment will be sourced from \$ENV_ACTIVATE_PATH"

# Check RESUME_FROM
if [ ! -z $RESUME_FROM ]; then
    # Check RESUME_FROM
    if [ \$(find "logs/\${TAG}/" -type f -name 'hpc_ckpt_*.ckpt' | wc -l) -gt 0 ]; then
        echo "Ignoring RESUME_FROM. Found hpc_ckpt_*.ckpt files under logs/\${TAG}"
        RESUME_FROM=''
    else
        echo "Passing parameter RESUME_FROM=$RESUME_FROM"
        RESUME_FROM=" --ckpt_path=$RESUME_FROM "
    fi
fi

# Run Model
time srun --output="logs/sbatch_%x_id%j_n%n_t%t.txt" bash -c " \
    \$ENV_COMMAND && \
    export TORCH_NCCL_BLOCKING_WAIT=1 && \
    export NCCL_DEBUG=INFO && \
    export PYTHONFAULTHANDLER=1 && \
    python -u run.py train --slurm \
        --RUN_TRAINING=False \
        --RUN_TESTING=True  \
        --RUN_PREDICT=True  \
        --devices=auto --num_workers=16 \
        --tonemap=MIXED \
        --tag=\${TAG} \
        --max_epochs=3500 \
        --output_dir=logs/\${TAG} \
        --cache_DISK_path=\${SLURM_TMPDIR}/cache/skymangler_envmap_skylatlong \
        \$RESUME_FROM \
    "

# DEFAULT OPTIONS:
# --HDRDB_path=datasets/skymangler/envmap_skylatlong

echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"


### ------------------------------------------------------------------------------ ###
##----------------------------##
#### Data to network storage ####
##----------------------------##
# DANGER! If requeuing, there is not enough time to copy the data to network storage!

# None


### ------------------------------------------------------------------------------ ###
##------------------------##
#### Restart?            ####
##------------------------##
# See SLURMEnvironment plugin for Pytorch Lightning:
# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.plugins.environments.SLURMEnvironment.html

# DANGER! Make sure to include the following sbatch commands:
# #SBATCH --requeue
# #SBATCH --signal=SIGUSR1@90


### ------------------------------------------------------------------------------ ###
##------------------------##
#### Cleanup             ####
##------------------------##

# Remove lock file
FILE_LOCK=".drac/lock_\${SLURM_JOB_ID}_\${SLURM_JOB_NAME}.txt"
if test -f \$FILE_LOCK ; then
    echo "Removing lock file: \$FILE_LOCK"
    rm \$FILE_LOCK
    echo "Removing lock file -- Done!"
fi

EOT
