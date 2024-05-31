#!/bin/bash -l

### See: https://slurm.schedmd.com/sbatch.html

# For Testing
# GLOBAL_COMMAND=" --TEST --limit_train_batches=5"
GLOBAL_COMMAND=""
CREATE_ENV_LOCAL=true
ENV_NAME="ENV_SKYGAN"

# Training
CACHE_DIR=datasets/skymangler_skygan_cache/cache
OUTPUT_DIR=output_SkyGAN


sbatch << EOT
#!/bin/bash -l

### NAME ###
#SBATCH --job-name="Train_SkyGAN"

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
#SBATCH --output="$OUTPUT_DIR/sbatch_%x_id%j.txt"

### REQUEING ###
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@360

### NODES/TASKS ###
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

### HARDWARE ###
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres="gpu:p100:2"

### OPTIONAL ###
## --exclusive to get the whole nodes exclusively for this job
#SBATCH --exclusive

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
TAG="SkyGAN_\${SLURM_JOB_ID}"

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

# Run Model
time srun --output="$OUTPUT_DIR/sbatch_%x_id%j_n%n_t%t.txt" bash -c " \
    \$ENV_COMMAND && \
    export TORCH_NCCL_BLOCKING_WAIT=1 && \
    export NCCL_DEBUG=INFO && \
    export PYTHONFAULTHANDLER=1 && \
    CACHE_DIR=$CACHE_DIR DNNLIB_CACHE_DIR=$CACHE_DIR \
    python -u src/stylegan3/train.py \
    --data=datasets/skymangler_skygan_cache/envmap_skylatlong/export_TRAIN.csv \
    --resolution=256 --gamma=2 \
    --cfg=stylegan3-t --gpus=2 \
    --batch=32 --batch-gpu=4 --tick=1 --snap=1 \
    --outdir=$OUTPUT_DIR \
    --metrics=fid50k_full \
    --mirror=0 \
    --aug-ada-xflip=0 \
    --aug-ada-rotate90=0 \
    --aug-ada-xint=0 \
    --aug-ada-scale=0 \
    --aug-ada-rotate=1 \
    --aug-ada-aniso=0 \
    --aug-ada-xfrac=0 \
    --normalize-azimuth=True \
    --use-encoder=True \
    "

# DEFAULT OPTIONS:
# @click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)


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
