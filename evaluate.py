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

parser = argparse.ArgumentParser()
parser.add_argument('--runid', type=str, help="Wandb run id")
parser.add_argument('--num_rollouts',
                    type=int,
                    default=1,
                    help="Number of rollouts to average over")

args = parser.parse_args()

#load the config from the wandb synced run.

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

saved_model = run.file("model.zip").download(replace=True)
model = PPO.load(saved_model.name)

env = build_env(config, baseline=False, render_mode="rgb_array")

successes = []

for j in range(args.num_rollouts):
    frames, rewards, actions, observations, infos = record_rollout(env, model)

    if "is_success" in infos[-1]:
        successes.append(infos[-1]["is_success"])

    print(len(frames), len(rewards), len(actions), len(observations))

    #plot rewards, actions, and observations over time, make video with moviepy
    import matplotlib.pyplot as plt
    import numpy as np

    # model_path = os.path.dirname(args.model_file)
    model_path = f"./evaluation_results/rollout_{j}"
    os.makedirs(model_path, exist_ok=True)

    make_movie(frames,
               model_path + f"/evaluation_rollout.mp4",
               fps=1 / config["ctrl_timestep"])

    fig = plt.figure()

    plt.plot(rewards)
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.savefig(model_path + f"/rewards.png", dpi=300)

    fig, axs = plt.subplots(
        env.action_space.shape[0],
        1,
        sharex=True,
        figsize=(10, 30),
    )
    for i in range(env.action_space.shape[0]):
        axs[i].plot(np.array(actions)[:, i])
        axs[i].set_ylabel(f"a{i}")

    axs[-1].set_xlabel("Timesteps")
    plt.savefig(model_path + f"/actions.png", dpi=300, bbox_inches='tight')

    # make histogram of actions
    fig, axs = plt.subplots(
        env.action_space.shape[0],
        1,
        figsize=(10, 30),
    )
    for i in range(env.action_space.shape[0]):
        axs[i].hist(np.array(actions)[:, i], bins=100)
        axs[i].set_ylabel(f"a{i}")

    axs[-1].set_xlabel("Action Value")
    plt.savefig(model_path + f"/actions_hist.png",
                dpi=300,
                bbox_inches='tight')

    fig, axs = plt.subplots(env.observation_space.shape[0],
                            1,
                            sharex=True,
                            figsize=(10, 40))
    for i in range(env.observation_space.shape[0]):
        axs[i].plot(np.array(observations)[:, i])
        axs[i].set_ylabel(f"o{i}")

    axs[-1].set_xlabel("Timesteps")
    plt.savefig(model_path + f"/observations.png",
                dpi=300,
                bbox_inches='tight')

print(f"Success rate: {np.mean(successes)}")
