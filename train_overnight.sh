#!/bin/bash

# Define the Python script path, virtual environment path, and number of runs
PYTHON_SCRIPT="/home/curtis/baloo/baloo-gym/train.py"
VENV_PATH="/home/curtis/baloo/baloo-env"
NUM_RUNS=5  # Number of times to run the script

for (( i=1; i<=NUM_RUNS; i++ ))
do
  echo "Starting iteration $i"

  cd /home/curtis/baloo/baloo-gym

  # uv run train.py --num_envs 16 --randomize_object_size --randomize_object_mass --randomize_object_pos --randomize_object_quat --total_timesteps 40000000 --reward_selection copy_baseline --seed $i --wandb
  uv run train.py --num_envs 16 --randomize_object_size --randomize_object_mass --randomize_object_pos --randomize_object_quat --total_timesteps 40000000 --reward_selection shaped_reward_comparison --seed $i --wandb

  echo "Completed iteration $i"
done