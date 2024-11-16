#!/bin/bash

# Define the Python script path, virtual environment path, and number of runs
PYTHON_SCRIPT="/home/curtis/baloo/baloo-gym/train.py"
VENV_PATH="/home/curtis/baloo/baloo-env"
NUM_RUNS=5  # Number of times to run the script

for (( i=1; i<=NUM_RUNS; i++ ))
do
  echo "Starting iteration $i"

  source $VENV_PATH/bin/activate

  cd /home/curtis/baloo/baloo-gym

  python3 $PYTHON_SCRIPT --total_timesteps 2000000 --env_name baloo_v4 --num_envs 16 --wandb

  echo "Completed iteration $i"
done
