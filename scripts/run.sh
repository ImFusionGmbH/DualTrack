#!/bin/bash
#SBATCH --job-name=tracking_estimation
#SBATCH --output=%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=a40
#SBATCH --time=1:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --open-mode=append

export PYTHONPATH=$PYTHONPATH:$(pwd)

srun $@