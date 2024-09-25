#take in policy from recent_model folder of a given experiment.and
# load it with sb3
# build an environment
# run the environment with the policy
# record the rewards, actions, and observations

from stable_baselines3 import PPO
import argparse
from utils.helpers import build_env, record_rollout, make_movie

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, help='Experiment name')
args = parser.parse_args()

args.experiment = "crimson-water-375"

model = PPO.load(f"./experiments/{args.experiment}/recent_model/model.zip")

config = {
    "total_timesteps": 1000000,
    "ctrl_timestep": .1,
    "env_name": "baloo_v3",
    "class_name": "BalooV3",
    "time_limit_sec": 30,
    "time_aware_obs": True,
}

env = build_env(config)

frames, rewards, actions, observations = record_rollout(env, model)

print(len(frames), len(rewards), len(actions), len(observations))

#plot rewards, actions, and observations over time, make video with moviepy
import matplotlib.pyplot as plt
import numpy as np

make_movie(
    frames,
    f"./experiments/{args.experiment}/recent_model/evaluation_rollout.mp4",
    fps=1 / config["ctrl_timestep"])

plt.plot(rewards)
plt.xlabel("Timesteps")
plt.ylabel("Rewards")
plt.savefig(f"./experiments/{args.experiment}/recent_model/rewards.png",
            dpi=300)

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
plt.savefig(f"./experiments/{args.experiment}/recent_model/actions.png",
            dpi=300,
            bbox_inches='tight')

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
plt.savefig(f"./experiments/{args.experiment}/recent_model/actions_hist.png",
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
plt.savefig(f"./experiments/{args.experiment}/recent_model/observations.png",
            dpi=300,
            bbox_inches='tight')
