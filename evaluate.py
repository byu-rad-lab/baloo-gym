#take in policy from recent_model folder of a given experiment.and
# load it with sb3
# build an environment
# run the environment with the policy
# record the rewards, actions, and observations

from stable_baselines3 import PPO
import argparse
from baloo_gym.utils.helpers import build_env, record_rollout, make_movie
from baloo_gym.policies.open_loop_hugger import OpenLoopHuggerPolicy
import wandb
import os
from baloo_mujoco_sim.utils.baloo_mj_api import (set_box_size,
                                                 set_box_position)
import shutil
import numpy as np

np.set_printoptions(precision=3, suppress=True)


def extract_reward_trajectories(infos):
    reward_history = {
        "dont_drop": [],
        "dropped": [],
        "chest_proximity": [],
        "touch_ground": [],
        'copy_baseline': [],
        "tactile_nonzero": [],
        "minimize_torques": [],
        "upward_force": [],
        "success": [],
        "box_fell_over": [],
    }

    for info in infos:
        for k in reward_history.keys():
            reward_history[k].append(info["reward_terms"].get(k, 0))

    return reward_history


def load_or_download_model(args, local_experiment_folder):
    run_path = None
    try:
        #try loading model from local experiments folder
        #find folder containing runid in /home/curtis/baloo/baloo-gym/new_experiments
        for f in os.listdir(local_experiment_folder):
            if args.runid in f:
                run_path = f"{local_experiment_folder}/{f}/best_model/best_model.zip"
                print(f"Loading model from {run_path}")
                break

        if run_path is None:
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
                new_path = os.path.join(local_experiment_folder,
                                        os.sep.join(file.name.split("/")[1:]))
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.move(old_path, new_path)
                print(f"Downloaded best model to {new_path}")
                run_path = new_path
                # shutil.rmtree(old_path.split("/")[0])
                break

        if run_path is None:
            raise FileNotFoundError(
                f"{args.runid} not found in experiments folder or on wandb.")

    return run_path


parser = argparse.ArgumentParser()
parser.add_argument('--runid', type=str, help="Wandb run id")
parser.add_argument('--num_rollouts',
                    type=int,
                    default=1,
                    help="Number of rollouts to average over")

parser.add_argument(
    '--resolution',
    type=int,
    default=1,
    help="Resolution multiplier of video (160x120) * resolution")

args = parser.parse_args()

#load the config from the wandb synced run.
if args.runid is None:
    folder_name = "open_loop_hugger"
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
            'dont_drop',
            'copy_baseline',
            'chest_proximity',
            'touch_ground',
            'tactile_nonzero',
            'upward_force',
            'minimize_torques',
        ],
        "randomize_initial_height":
        False,
        "randomize_object_size":
        True,
        "randomize_object_mass":
        True,
        "randomize_object_quat":
        True,
        "randomize_object_pos":
        True,
    }

    env = build_env(config,
                    baseline=True,
                    render_mode="rgb_array",
                    resolution=args.resolution)
else:
    run = wandb.Api().run(f"curtiscjohnson/ppo_baloo/{args.runid}")
    folder_name = f"{run.name}-{run.id}"
    config = {
        "total_timesteps": run.config["total_timesteps"],
        "ctrl_timestep": run.config["ctrl_timestep"],
        "env_name": run.config["env_name"],
        "time_limit_sec": run.config["time_limit_sec"],
        "curriculum_selection": run.config["curriculum_selection"],
        'reward_selection': run.config['reward_selection'],
        "randomize_initial_height": False,
        "randomize_object_size": True,
        "randomize_object_mass": True,
    }

    model_path = load_or_download_model(args, f"./new_experiments")
    model = PPO.load(model_path)

    env = build_env(config,
                    baseline=False,
                    render_mode="rgb_array",
                    resolution=args.resolution)

successes = []

for j in range(args.num_rollouts):
    if args.runid is None:
        model = OpenLoopHuggerPolicy(N=50)
        frames, rewards, actions, observations, infos = record_rollout(
            env, model, deterministic=True, return_dist=False)
    else:
        frames, rewards, actions, observations, infos, dists = record_rollout(
            env, model, deterministic=True)

    if "is_success" in infos[-1]:
        successes.append(infos[-1]["is_success"])

    reward_history = extract_reward_trajectories(infos)

    #plot rewards, actions, and observations over time, make video with moviepy

    print(len(frames), len(rewards), len(actions), len(observations))

    # plot all lines in reward_history as subplots
    # run_path = os.path.dirname(args.model_file)
    run_path = f"./evaluation_results/{folder_name}/rollout_{j}"
    os.makedirs(run_path, exist_ok=True)

    import matplotlib.pyplot as plt
    import numpy as np
    fig, axs = plt.subplots(10, 1, figsize=(10, 40), sharex=True)

    for i, k in enumerate(reward_history.keys()):
        axs[i].plot(reward_history[k], label=k)
        axs[i].legend()
        axs[i].grid()
        axs[i].set_ylabel("Rewards")

    axs[-1].set_xlabel("Timesteps")
    plt.savefig(
        f"./evaluation_results/{folder_name}/rollout_{j}/reward_terms.png",
        dpi=300,
        bbox_inches='tight')
    plt.close(fig)

    make_movie(frames,
               run_path + f"/evaluation_rollout.mp4",
               fps=1 / config["ctrl_timestep"])

    fig = plt.figure()

    plt.plot(rewards)
    plt.grid()
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.savefig(run_path + f"/rewards.png", dpi=300)

    fig, axs = plt.subplots(
        env.action_space.shape[0],
        1,
        sharex=True,
        figsize=(10, 30),
    )
    for i in range(env.action_space.shape[0]):
        axs[i].plot(np.array(actions)[:, i])
        axs[i].set_ylabel(f"a{i}")
        axs[i].grid()

    axs[-1].set_xlabel("Timesteps")
    plt.savefig(run_path + f"/actions.png", dpi=300, bbox_inches='tight')

    # make histogram of actions
    fig, axs = plt.subplots(
        env.action_space.shape[0],
        1,
        figsize=(10, 30),
    )
    for i in range(env.action_space.shape[0]):
        axs[i].hist(np.array(actions)[:, i], bins=100)
        axs[i].set_ylabel(f"a{i}")
        axs[i].grid()

    axs[-1].set_xlabel("Action Value")
    plt.savefig(run_path + f"/actions_hist.png", dpi=300, bbox_inches='tight')

    fig, axs = plt.subplots(env.observation_space.shape[0],
                            1,
                            sharex=True,
                            figsize=(10, 60))
    for i in range(env.observation_space.shape[0]):
        axs[i].plot(np.array(observations)[:, i])
        axs[i].set_ylabel(f"o{i}")
        axs[i].grid()

    axs[-1].set_xlabel("Timesteps")
    plt.savefig(run_path + f"/observations.png", dpi=300, bbox_inches='tight')

    #plot the action distribution for each action, 13 x 2
    if args.runid is not None:
        means = [dists[i][0] for i in range(len(dists))]
        stds = [dists[i][1] for i in range(len(dists))]

        means = np.array(means)
        stds = np.array(stds)

        fig, axs = plt.subplots(env.action_space.shape[0], 1, figsize=(10, 30))

        for i in range(env.action_space.shape[0]):
            axs[i].plot(means[:, i], label="mean")
            axs[i].plot(np.array(actions)[:, i], 'r--', label="action")
            axs[i].plot(np.ones(len(means)) * env.action_space.low[i], 'k--')
            axs[i].plot(np.ones(len(means)) * env.action_space.high[i], 'k--')
            axs[i].fill_between(
                np.arange(len(means)),
                means[:, i] - 3 * stds[:, i],
                means[:, i] + 3 * stds[:, i],
                alpha=0.5,
                label="3 std",
            )
            axs[i].legend()
            axs[i].set_ylabel(f"a{i}")
            axs[i].grid()

        axs[-1].set_xlabel("Timesteps")
        plt.savefig(run_path + f"/action_dist.png",
                    dpi=300,
                    bbox_inches='tight')
        plt.close('all')

print(f"Success rate: {np.mean(successes)}")
