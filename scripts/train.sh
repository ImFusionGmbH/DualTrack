#!/bin/bash
#SBATCH --job-name=tracking_estimation
#SBATCH --output=.slurm/%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=a40
#SBATCH --time=8:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --open-mode=append
#SBATCH --exclude gpu138,gpu156

# send this batch script a SIGUSR1 240 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@240

echo "hostname ${HOSTNAME}"
echo running script...

if [ ! -f "/checkpoint/$USER/$SLURM_JOB_ID/experiment_dir.txt" ]; 
then
    echo "Saving experiment start time"
    experiment_dir="$(pwd)/$(date "+experiments/%Y-%b-%d/%-I:%M:%S%p")"
    echo "Creating experiment dir ${experiment_dir}"
    mkdir -p $experiment_dir
    ln -s -f /checkpoint/$USER/$SLURM_JOB_ID $experiment_dir/checkpoint
    echo $experiment_dir > /checkpoint/$USER/$SLURM_JOB_ID/experiment_dir.txt

    # scontrol requeue $SLURM_JOB_ID
else
    echo "Found experiment state"
    experiment_dir=$(cat /checkpoint/$USER/$SLURM_JOB_ID/experiment_dir.txt)
    echo "Experiment dir: ${experiment_dir}"
fi 

# Kill training process and resubmit job if it receives a SIGUSR1
handle_timeout_or_preemption() {
    echo $(date +"%Y-%m-%d %T") "Caught timeout or preemption signal"
    scontrol requeue $SLURM_JOB_ID
    exit 0
}
trap handle_timeout_or_preemption SIGUSR1

# =====================================================================
# Set environment variables for training - these are useful examples
# export TQDM_MININTERVAL=30
export WANDB_RUN_ID=$SLURM_JOB_ID
export WANDB_RESUME=allow
export PYTHONPATH=$PYTHONPATH:$(pwd)

# =====================================================================
# Run training script - this is where you would put your training script
echo "Running training script"

# srun "$@" --log_dir=$experiment_dir & # <- ampersand is important
srun "$@" --log_dir=$experiment_dir & # <- ampersand is important

# =====================================================================
child_pid=$!
wait $child_pid