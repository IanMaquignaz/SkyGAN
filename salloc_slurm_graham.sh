#!/bin/bash

### ------------------------------------------------------------------------------ ###
##-------------------------##
#### CONSTANTS (SET ME!) ####
##-------------------------##

DIR_LOGS="logs"
ENV_NAME="ENV_SKYGAN"

### ------------------------------------------------------------------------------ ###
##------------------------##
#### Virtual Environment ####
##------------------------##

# DANGER!
# If an internet connection is required, the
# environment must be built using ./build_env.sh
# on the login node!
# ./.drac/build_env.sh


### For GPU specification, see:
### https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm

###  Time formats: "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".

### ------------------------------------------------------------------------------ ###
##-------------##
#### SALLOC ####
##-------------##

### For multi-node multi-GPU, see:
### https://docs.alliancecan.ca/wiki/PyTorch
### https://github.com/PrincetonUniversity/multi_gpu_training
### https://pytorch-lightning.readthedocs.io/en/1.2.10/clouds/slurm.html

### FOR TESTING, USE INTERACTIVE SESSIONS: ###
### DANGER! TWEAK before copy pasting!
### See: https://slurm.schedmd.com/salloc.html
# salloc \
#     --account def-jlalonde_gpu  `### RUNTIME ###` \
#     --time="3:0:0" `### TIME (3+hrs gets put in queue) ###`  \
#     \
#     `### ACCOUNT ###` \
#     --account=def-jlalonde \
#     --mail-user=ian.maquignaz.1@ulaval.ca \
#     --mail-type=ALL `# [Optional]` \
#     --mail-type=begin `# [Optional]` \
#     \
#     `### NODES/TASKS ###` \
#     --nodes=2 `# Nodes to request` \
#     --ntasks=4 `# Number of instances to be created` \
#     --ntasks-per-node=2 \
#     \
#     `### HARDWARE ###` \
#     --cpus-per-task=8 \
#     --cpus-per-gpu=1 `# CPUs per instance` \
#     --gpus-per-task=1 `# GPUs per instance` \
#     --gpus-per-node="p100:2" `# Number of GPU(s) per node (type:num)` \
#     --exclusive `# Request whole node(s)` \
#     --mem=8G `# CPU memory per node` \
#     --mem=0 `# Request whole node(s) CPU memory` \
#     \
#     `### EXECUTIBLE ###` \
#     srun --output="${DIR_LOGS}/salloc_%x_id%j_n%n_t%t.txt" train.sh
JOB_ID=$(squeue -u iamaq --noheader --format="%A %j" | grep SALLOC_DeepClouds | cut -f1 -d' ')
if [ -z "$JOB_ID" ]; then
    LOG_FILE="${DIR_LOGS}/salloc_id%j_n%n_t%t.txt"
    salloc \
        --job-name="SALLOC_DeepClouds" \
        --account=def-jlalonde_gpu  `### RUNTIME ###` \
        --time="3:0:0" `### TIME (3+hrs gets put in queue) ###`  \
        \
        `### ACCOUNT ###` \
        --account=def-jlalonde \
        --mail-user=ian.maquignaz.1@ulaval.ca \
        --mail-type=begin `# [Optional]` \
        \
        `### NODES/TASKS ###` \
        --nodes=1 `# Nodes to request` \
        --ntasks-per-node=1 `# number of tasks per node` \
        \
        `### HARDWARE ###` \
        --cpus-per-task=16 \
        --mem=64G `# CPU memory per node` \
        --gres="gpu:p100:2" `# GPUs per instance` \

    JOB_ID=$(squeue -u iamaq --noheader --format="%A %j" | grep SALLOC_DeepClouds | cut -f1 -d' ')
fi


### EXECUTIBLE (RUN SEPERATELY) ###

# srun --jobid XXXXX --output=logs/salloc_%x_id%j_n%n_t%t.txt --nodes=1 --ntasks-per-node=2 bash -c " \
#     module restore $ENV_NAME_modules && \
#     source $ENV_NAME/bin/activate && \
#     export TORCH_NCCL_BLOCKING_WAIT=1 && \
#     export NCCL_DEBUG=INFO && \
#     export PYTHONFAULTHANDLER=1 && \
#     python run.py train --slurm \
#     --RUN_TRAINING=True \
#     --RUN_TESTING=True  \
#     --RUN_PREDICT=True  \
#     --devices=auto --num_workers=6 \
#     --enable_progress_bar --tonemap=MIXED  \
#     --num_samples_max=500 --batch_size=4 --max_epochs=10  \
#     --val_every_n_epochs=1 --log_every_n_steps=5 \
#     --tag=__TEST__ \
#     "

# echo "Starting job using JOB_ID=$JOB_ID"
# srun --jobid $JOB_ID \
#     --output=logs/salloc_id%j_n%n_t%t.txt --nodes=1 --ntasks-per-node=1 \
#     bash -c " \
#     module restore ${ENV_NAME}_modules && \
#     source $ENV_NAME/bin/activate && \
#     python run.py train \
#     --RUN_TRAINING=True \
#     --RUN_TESTING=True  \
#     --RUN_PREDICT=True  \
#     --devices=auto --num_workers=16 \
#     --enable_progress_bar --tonemap=MIXED  \
#     --num_samples_max=500 --batch_size=24 --max_epochs=10  \
#     --val_every_n_epochs=1 --log_every_n_steps=5 \
#     --tag=__TEST__ \
#     "