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

parser = argparse.ArgumentParser()
parser.add_argument('--runid', type=str, help="Wandb run id")
parser.add_argument('--model_file', type=str, help="Path to model file")
parser.add_argument('--num_rollouts',
                    type=int,
                    default=1,
                    help="Number of rollouts to average over")

args = parser.parse_args()

#load the config from the wandb synced run.

with wandb.init(id=args.runid, resume="allow") as run:
    folder_name = f"{run.name}-{run.id}"
    config = {
        "total_timesteps": run.config["total_timesteps"],
        "ctrl_timestep": run.config["ctrl_timestep"],
        "env_name": run.config["env_name"],
        "time_limit_sec": run.config["time_limit_sec"],
        "time_aware_obs": run.config["time_aware_obs"],
        "curriculum_selection": run.config["curriculum_selection"],
        'reward_selection': run.config['reward_selection'],
        "randomize_initial_height": run.config['randomize_initial_height'],
    }

    model = PPO.load(args.model_file)

    env = build_env(config,
                    folder_name,
                    baseline=False,
                    monitor=False,
                    render_mode="rgb_array")

    #todo: rollout several times and average results.
    for i in range(args.num_rollouts):
        frames, rewards, actions, observations = record_rollout(env, model)

    print(len(frames), len(rewards), len(actions), len(observations))

    #plot rewards, actions, and observations over time, make video with moviepy
    import matplotlib.pyplot as plt
    import numpy as np

    model_path = os.path.dirname(args.model_file)

    make_movie(frames,
               model_path + "/evaluation_rollout.mp4",
               fps=1 / config["ctrl_timestep"])

    plt.plot(rewards)
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.savefig(model_path + "/rewards.png", dpi=300)

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
    plt.savefig(model_path + "/actions.png", dpi=300, bbox_inches='tight')

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
    plt.savefig(model_path + "/actions_hist.png", dpi=300, bbox_inches='tight')

    fig, axs = plt.subplots(env.observation_space.shape[0],
                            1,
                            sharex=True,
                            figsize=(10, 40))
    for i in range(env.observation_space.shape[0]):
        axs[i].plot(np.array(observations)[:, i])
        axs[i].set_ylabel(f"o{i}")

    axs[-1].set_xlabel("Timesteps")
    plt.savefig(model_path + "/observations.png", dpi=300, bbox_inches='tight')

    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
