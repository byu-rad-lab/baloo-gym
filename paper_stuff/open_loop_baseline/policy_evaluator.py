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

#parallelize running simulation over each combination ith multiprocessing


def run_simulation(combination):
    config = {
        "total_timesteps": 1000000,
        "ctrl_timestep": .05,
        "env_name": "baloo_v9",
        "time_limit_sec": 30,
        "curriculum_selection": [],
        'reward_selection':
        ["dont_drop", "chest_proximity", "high_contact_forces"],
        "randomize_initial_height": False,
        "randomize_object_size": False,
        "randomize_object_mass": False,
    }
    x, y, z, m = combination
    print(f"Running trial with object size: {x, y, z} and mass: {m}")

    env = build_env(config,
                    baseline=False if args.runid else True,
                    render_mode="rgb_array",
                    object_size=[x, y, z],
                    object_mass=m)
    successes = []

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
        else:
            model = OpenLoopHuggerPolicy(N=50)

        if args.runid:
            frames, rewards, actions, observations, infos = record_rollout(
                env,
                model,
                render=False,
                deterministic=False,
                return_dist=False)
        else:
            frames, rewards, actions, observations, infos, dist = record_rollout(
                env, model, deterministic=False, render=False)

        if infos[-1]["is_success"]:
            successes.append(1)
        else:
            successes.append(0)

        frames = None
        rewards = None
        actions = None
        observations = None
        infos = None
        gc.collect()

    trial_data["success_rate"] = sum(successes) / len(successes)
    return trial_data


if __name__ == "__main__":

    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument('--runid', type=str, help="Wandb run id")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--grid', action='store_true')
    group.add_argument('--random', action='store_true')

    args = parser.parse_args()
    local_experiment_folder = "/home/curtis/baloo/baloo-gym/new_experiments"
    model_path = None

    if args.runid:
        try:
            #try loading model from local experiments folder
            #find folder containing runid in /home/curtis/baloo/baloo-gym/new_experiments
            for f in os.listdir(local_experiment_folder):
                if args.runid in f:
                    model_path = f"{local_experiment_folder}/{f}/best_model/best_model.zip"
                    print(f"Loading model from {model_path}")
                    break

            if model_path is None:
                raise FileNotFoundError(
                    f"{args.runid} not found in experiments folder.")
        except:
            #try downloading model from wandb
            print(
                f"{args.runid} not found in experiments folder. Trying download from wandb instead..."
            )
            run = wandb.Api().run(f"curtiscjohnson/ppo_baloo/{args.runid}")
            for file in run.files():
                if "best_model" in file.name:
                    file.download(replace=True)
                    print(f"Found best model.")
                    old_path = file.name
                    new_path = os.path.join(
                        local_experiment_folder,
                        os.sep.join(file.name.split("/")[1:]))
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    shutil.move(old_path, new_path)
                    print(f"Downloaded best model to {new_path}")
                    model_path = new_path
                    shutil.rmtree(old_path.split("/")[0])
                    break

            if model_path is None:
                raise FileNotFoundError(
                    f"{args.runid} not found in experiments folder or on wandb."
                )

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

    lock = Lock()

    with Pool(processes=16) as pool:
        results = list(
            tqdm(pool.imap(run_simulation, combinations),
                 total=len(combinations)))

    # print results to a file
    tag = args.runid if args.runid else "baseline"
    type = "grid" if args.grid else "random"
    with open(f"data/lifting_trials_{tag}_{type}.txt", "w") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")

# X, Y, Z, M = np.meshgrid(xsize, ysize, zsize, mass)

# X = X.flatten()
# Y = Y.flatten()
# Z = Z.flatten()
# M = M.flatten()

# #iterate over all combinations
# for x, y, z, m in tqdm(zip(X, Y, Z, M)):
#     print(f"Running trial with object size: {x, y, z} and mass: {m}")
#     env = build_env(config,
#                     baseline=True,
#                     render_mode="rgb_array",
#                     object_size=[x, y, z],
#                     object_mass=m)
#     successes = []

#     #get size and weight of object
#     xsize, ysize, zsize = env.unwrapped.model.geom("box").size
#     mass = env.unwrapped.model.body("box").mass

#     trial_data = {
#         "xsize": xsize,
#         "ysize": ysize,
#         "zsize": zsize,
#         "mass": mass.item(),
#     }

#     for _ in range(10):
#         model = OpenLoopHuggerPolicy(N=50)
#         frames, rewards, actions, observations, infos = record_rollout(
#             env, model, render=False)
#         if infos[-1]["is_success"]:
#             successes.append(1)
#         else:
#             successes.append(0)

#         frames = None
#         rewards = None
#         actions = None
#         observations = None
#         infos = None
#         gc.collect()

#     trial_data["success_rate"] = np.mean(successes)

#     print(trial_data)

#     env.close()

#     #save data to file
#     with open("lifting_trials.json", "a") as f:
#         json.dump(trial_data, f)
#         f.write("\n")
#         f.close()

#     env = None
#     gc.collect()

# # N = int(config["time_limit_sec"] / config["ctrl_timestep"])

# maybe group objects into weight classes, then height v aspect ratio (xsize/ysize) can be 3 3d plots.

# make_movie(frames, "test.mp4", fps=1 / config["ctrl_timestep"])

# plt.plot(rewards)
# plt.xlabel("Timesteps")
# plt.ylabel("Rewards")
# plt.savefig("rewards.png", dpi=300)

# fig, axs = plt.subplots(
#     env.action_space.shape[0],
#     1,
#     sharex=True,
#     figsize=(10, 30),
# )
# for i in range(env.action_space.shape[0]):
#     axs[i].plot(np.array(actions)[:, i])
#     axs[i].set_ylabel(f"a{i}")

# axs[-1].set_xlabel("Timesteps")
# plt.savefig("actions.png", dpi=300, bbox_inches='tight')

# # make histogram of actions
# fig, axs = plt.subplots(
#     env.action_space.shape[0],
#     1,
#     figsize=(10, 30),
# )
# for i in range(env.action_space.shape[0]):
#     axs[i].hist(np.array(actions)[:, i], bins=100)
#     axs[i].set_ylabel(f"a{i}")

# axs[-1].set_xlabel("Action Value")
# plt.savefig("actions_hist.png", dpi=300, bbox_inches='tight')

# fig, axs = plt.subplots(env.observation_space.shape[0],
#                         1,
#                         sharex=True,
#                         figsize=(10, 40))
# for i in range(env.observation_space.shape[0]):
#     axs[i].plot(np.array(observations)[:, i])
#     axs[i].set_ylabel(f"o{i}")

# axs[-1].set_xlabel("Timesteps")
# plt.savefig("observations.png", dpi=300, bbox_inches='tight')
