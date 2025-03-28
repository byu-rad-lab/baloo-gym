import json
from baloo_gym.policies.open_loop_hugger import OpenLoopHuggerPolicy
from baloo_gym.utils.helpers import build_env, record_rollout, make_movie
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Lock
from itertools import product
import gc
import argparse
import json
import wandb
from stable_baselines3 import PPO
import os
import shutil
from scipy.stats import qmc
import random
import ast
from pathlib import Path

#parallelize running simulation over each combination ith multiprocessing


def run_simulation(combination):
    config = {
        "total_timesteps":
        1000000,
        "ctrl_timestep":
        .05,
        "env_name":
        "baloo_v9",
        "time_limit_sec":
        60,
        "curriculum_selection": [],
        'reward_selection': [
            'dont_drop', 'chest_proximity', 'touch_ground', 'tactile_nonzero',
            'upward_force'
        ],
        "randomize_initial_height":
        False,
        "randomize_object_size":
        False,
        "randomize_object_mass":
        False,
        "randomize_object_quat":
        True,
        "randomize_object_pos":
        True,
    }
    x, y, z, m = combination
    # print(f"Running trial with object size: {x, y, z} and mass: {m}")

    env = build_env(config,
                    baseline=False if args.runid else True,
                    render_mode="rgb_array",
                    object_size=[x, y, z],
                    object_mass=m)
    successes = []
    tips = []
    slips = []
    episode_lengths = []

    # Get size and weight of object
    xsize, ysize, zsize = env.unwrapped.model.geom("box").size
    mass = env.unwrapped.model.body("box").mass

    trial_data = {
        "xsize": xsize,
        "ysize": ysize,
        "zsize": zsize,
        "mass": mass.item(),
    }

    for _ in range(args.num_trials):
        if args.runid:
            with lock:
                model = PPO.load(model_path)

            frames, rewards, actions, observations, infos = record_rollout(
                env,
                model,
                render=False,
                deterministic=True,
                return_dist=False)
        else:
            model = OpenLoopHuggerPolicy(N=50)
            frames, rewards, actions, observations, infos = record_rollout(
                env,
                model,
                deterministic=True,
                render=False,
                return_dist=False)

        #episodes terminate by success or tip, success = 1 - (tip + slip). Each option is mutually exclusive.
        # print(infos[-1])
        success = infos[-1].get("is_success", False)
        tip = infos[-1].get("box_fell_over", False)
        slip = True - (success or tip)

        # print(f"Success: {success}, Tip: {tip}, Slip: {slip}")

        successes.append(success)
        tips.append(tip)
        slips.append(slip)

        episode_lengths.append(len(rewards))

        frames = None
        rewards = None
        actions = None
        observations = None
        infos = None
        gc.collect()

    trial_data["success_rate"] = sum(successes) / len(successes)
    trial_data["tip_rate"] = sum(tips) / len(tips)
    trial_data["slip_rate"] = sum(slips) / len(slips)
    trial_data["avg_episode_length"] = np.mean(episode_lengths)

    return trial_data


def load_or_download_model(args, local_experiment_folder):
    #this needs to download everythign in the new_experiments folder from wandb
    run_path = None
    try:
        #try loading model from local experiments folder
        #find folder containing runid in /home/curtis/baloo/baloo-gym/new_experiments
        for f in os.listdir(local_experiment_folder):
            if args.runid in f:
                run_path = f"{local_experiment_folder}/{f}/"
                print(f"Found run locally in {run_path}")
                break

        if run_path is None:
            raise FileNotFoundError(
                f"{args.runid} not found in {local_experiment_folder}.")
    except:
        #try downloading model from wandb
        print(
            f"{args.runid} not found in {local_experiment_folder}. Trying download from wandb instead..."
        )

        try:
            run = wandb.Api().run(f"curtiscjohnson/ppo_baloo/{args.runid}")
            print(f"Found run on wandb: curtiscjohnson/ppo_baloo/{args.runid}")
        except Exception as e:
            print(f"Error downloading from wandb: {e}")
            raise FileNotFoundError(
                f"{args.runid} not found in {local_experiment_folder} or on wandb."
            )

        print(f"Downloading run files to {local_experiment_folder}...")
        for file in tqdm(run.files()):
            #download everything in the new_experiments folder
            if file.name.startswith("new_experiments/"):
                file.download(root=os.path.dirname(local_experiment_folder),
                              replace=True)
                run_name = Path(file.name).parts[1]

        run_path = os.path.join(local_experiment_folder, run_name)

    #find model_name in the run_path recursively
    model_path = None
    for root, dirs, files in os.walk(run_path):
        for file in files:
            if file.endswith(".zip") and (args.model_name in file):
                model_path = os.path.join(root, file)
                break
        if model_path:
            break

    if model_path is None:
        raise FileNotFoundError(
            f"Model not found in {run_path}. Please specify the model name.")

    print(f"Loading model from {model_path}")
    return model_path


def sample_lhs(seed, N):
    # Define the parameter ranges (5 points each)
    #needs to match what's in baloo_base.py
    xsize = np.linspace(0.2, 0.6, 5)
    ysize = np.linspace(0.2, 0.6, 5)
    zsize = np.linspace(0.5, 1.25, 5)
    mass = np.linspace(.5, 10, 5)

    # Create the Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=4,
                                 seed=seed)  # 4 dimensions (x, y, z, mass)

    # Generate 5 samples (one for each range)
    num_samples = N
    lhs_samples = sampler.random(n=num_samples)

    # Scale the LHS samples to the respective parameter ranges
    x_samples = np.interp(lhs_samples[:, 0], [0, 1],
                          [xsize.min(), xsize.max()])
    y_samples = np.interp(lhs_samples[:, 1], [0, 1],
                          [ysize.min(), ysize.max()])
    z_samples = np.interp(lhs_samples[:, 2], [0, 1],
                          [zsize.min(), zsize.max()])
    mass_samples = np.interp(lhs_samples[:, 3], [0, 1],
                             [mass.min(), mass.max()])

    # Combine the scaled samples into a list of tuples
    sampled_points = [
        (x, y, z, m)
        for x, y, z, m in zip(x_samples, y_samples, z_samples, mass_samples)
    ]

    return sampled_points


if __name__ == "__main__":

    #ssed numpy random number generator
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument('--runid', type=str, help="Wandb run id")
    parser.add_argument('--model_name', type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--grid', action='store_true')
    group.add_argument('--random', action='store_true')
    group.add_argument('--lhs', action='store_true')

    args = parser.parse_args()
    local_experiment_folder = "/home/curtis/baloo/baloo-gym/new_experiments"
    model_path = None

    if args.runid:
        model_path = load_or_download_model(args, local_experiment_folder)

    if args.grid:
        #create grid of object sizes and weights
        xsize = np.linspace(0.2, 0.6, 5)
        ysize = np.linspace(0.2, 0.6, 5)
        zsize = np.linspace(.5, 1.25, 5)
        mass = np.linspace(5, 20, 5)

        combinations = list(product(xsize, ysize, zsize, mass))
    elif args.random:
        #create random combinations of object sizes and weights
        xsize = np.random.uniform(0.2, 0.6, 5)
        ysize = np.random.uniform(0.2, 0.6, 5)
        zsize = np.random.uniform(.5, 1.25, 5)
        mass = np.random.uniform(5, 20, 5)

        combinations = list(product(xsize, ysize, zsize, mass))

    elif args.lhs:
        N = 1000
        #read from lhs_samples.txt if it exists, otherwise create it
        if os.path.exists(f"{N}_lhs_samples.txt"):
            print(f"Loading LHS samples from {N}_lhs_samples.txt")
            with open(f"{N}_lhs_samples.txt", "r") as f:
                combinations = [
                    ast.literal_eval(line.strip()) for line in f.readlines()
                ]

        else:
            print("Generating LHS samples")
            combinations = sample_lhs(seed=seed, N=N)
            #write these to a file
            with open(f"{len(combinations)}_lhs_samples.txt", "w") as f:
                for combination in combinations:
                    f.write(f"{combination}\n")
    else:
        raise ValueError("Must specify --grid, --random, or --lhs")

    lock = Lock()

    with Pool(processes=os.cpu_count() // 2) as pool:
        results = list(
            tqdm(pool.imap_unordered(run_simulation, combinations),
                 total=len(combinations)))

    # print results to a file
    tag = args.runid if args.runid else "baseline"
    type = "grid" if args.grid else "random" if args.random else "lhs"
    length = len(combinations)
    model = Path(args.model_name).stem if args.model_name else "hugger"

    os.makedirs("data", exist_ok=True)

    with open(f"data/lifting_trials_{tag}_{type}_{length}_{model}.txt",
              "w") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")
