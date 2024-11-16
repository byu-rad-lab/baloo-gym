from baloo_gym.policies.open_loop_hugger import OpenLoopHuggerPolicy
from baloo_gym.utils.helpers import build_env, record_rollout, make_movie
import matplotlib.pyplot as plt
import numpy as np

#just pick a config right now. The config is only really used for the environment name.

config = {
    "total_timesteps": 1000000,
    "ctrl_timestep": .1,
    "env_name": "baloo_v4",
    "time_limit_sec": 30,
    "time_aware_obs": True,
    "curriculum_selection": [],
}

env = build_env(config,
                folder_name="test",
                baseline=True,
                monitor=False,
                render_mode="rgb_array")

# N = int(config["time_limit_sec"] / config["ctrl_timestep"])
model = OpenLoopHuggerPolicy(N=100)

frames, rewards, actions, observations = record_rollout(env, model)

make_movie(frames, "test.mp4", fps=1 / config["ctrl_timestep"])

plt.plot(rewards)
plt.xlabel("Timesteps")
plt.ylabel("Rewards")
plt.savefig("rewards.png", dpi=300)

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
plt.savefig("actions.png", dpi=300, bbox_inches='tight')

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
plt.savefig("actions_hist.png", dpi=300, bbox_inches='tight')

fig, axs = plt.subplots(env.observation_space.shape[0],
                        1,
                        sharex=True,
                        figsize=(10, 40))
for i in range(env.observation_space.shape[0]):
    axs[i].plot(np.array(observations)[:, i])
    axs[i].set_ylabel(f"o{i}")

axs[-1].set_xlabel("Timesteps")
plt.savefig("observations.png", dpi=300, bbox_inches='tight')
