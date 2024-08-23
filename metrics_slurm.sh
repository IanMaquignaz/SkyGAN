#!/bin/bash -l

# SkyGAN Metrics
JOB_NAME="SkyGAN_Metrics"
ARRAY_FILE="SkyGAN_Metrics.txt"

GLOBAL_COMMAND="\
    --network=output_SkyGAN/00003-stylegan3-t-export_TRAIN-gpus4-batch32-gamma2/network-snapshot-003541.pkl \
    --run_dir=output_metrics \
"

#**********************************#
####################################
#### EVERYTHING BELOW IS GLOBAL ####
####################################
#   PLEASE DONT CHANGE ANYTHING    #
#**********************************#

# Environment
CREATE_ENV_LOCAL=true
ENV_NAME="ENV_SkyGAN"


sbatch << EOT
#!/bin/bash -l

### See: https://slurm.schedmd.com/sbatch.html

### NAME ###
#SBATCH --job-name=$JOB_NAME

### TIME ###
### Time formats:
###     "minutes", "minutes:seconds"
###     "hours:minutes:seconds",
###     "days-hours", "days-hours:minutes"
###     "days-hours:minutes:seconds".
#SBATCH --time="1:0:0"

### ACCOUNT ###
#SBATCH --account=def-jlalonde
#SBATCH --mail-user=ian.maquignaz.1@ulaval.ca
#SBATCH --mail-type=ALL

### OUTPUT x=job-name, j=job-ID, n=node-number ###
#SBATCH --output="output_metrics/sbatch_array_%x_id%j.txt"

### REQUEUING ###
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@720

### HARDWARE ###
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gres="gpu:p100:1"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

### ARRAY ###
### --array=1-7 # jobs 1 to 7 inclusive
### --array=1,3,5 # jobs 1,3 and 5 inclusive
##SBATCH --array=1-$ARRAY_SIZE

### OPTIONAL ###
### --exclusive to get the whole nodes exclusively for this job
##SBATCH --exclusive

### OPTIONAL ###
### --test-only Validate the batch script and return an queue time estimate
##SBATCH --test-only


echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run Starting @@@@@@@@@@@@@@@@@@@@@@@@@"
date
hostname -f
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
##------------------##
#### Env. Vars.   ####
##------------------##

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
FILE_LOCK=".drac_locks/lock_\${SLURM_JOB_ID}_\${SLURM_JOB_NAME}.txt"
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
##-------------------------------##
#### Configuring Run Environment ####
##-------------------------------##
echo ""
echo "#########################################################################"
echo "@@@@@@@@@@@@@@@@@@ Configuring Run Environment @@@@@@@@@@@@@@@@@@@@@@@@@@"
date
hostname -f
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "#########################################################################"

JOB_NAME=$JOB_NAME
TAG="\${JOB_NAME}_\${SLURM_ARRAY_TASK_ID}"

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


### ------------------------------------------------------------------------------ ###
##------------------------##
#### Lauching Run         ####
##------------------------##
echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Lauching Run @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
date
hostname -f
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"

# Run Model
ABLATION_COMMAND=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" $ARRAY_FILE)
CACHE_DIR=\${SLURM_TMPDIR}/cache_StyleGAN_\${SLURM_JOB_ID}
time srun --output="output_metrics/\$TAG/%x_id%j_t%a_n%n_t%t.txt" bash -c " \
    \$ENV_COMMAND && \
    export TORCH_NCCL_BLOCKING_WAIT=1 && \
    export TORCH_DISTRIBUTED_DEBUG=DETAIL && \
    export NCCL_DEBUG=INFO && \
    export NCCL_ASYNC_ERROR_HANDLING=1 && \
    export PYTHONFAULTHANDLER=1 && \
    python -u src/stylegan3/calc_metrics.py \
        --data=datasets/skymangler_skygan_cache/envmap_skylatlong/export_EVALGRID_DEMO.csv \
        --gpus=1 \
        --mirror=0 \
        --metrics=export_full \
        $GLOBAL_COMMAND \
    "


echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@@"
date
hostname -f
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
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
if test -f \$FILE_LOCK ; then
    echo "Removing lock file: \$FILE_LOCK"
    rm \$FILE_LOCK
    echo "Removing lock file -- Done!"
fi


echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Cleanup Completed @@@@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"

EOT
