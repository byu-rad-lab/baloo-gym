import json
from baloo_gym.policies.open_loop_hugger import OpenLoopHuggerPolicy
from baloo_gym.utils.helpers import build_env, record_rollout, make_movie
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc

#just pick a config right now. The config is only really used for the environment name.

config = {
    "total_timesteps": 1000000,
    "ctrl_timestep": .1,
    "env_name": "baloo_v8",
    "time_limit_sec": 30,
    "curriculum_selection": [],
    'reward_selection': ["dont_drop", "chest_proximity"],
    "randomize_initial_height": False,
    "randomize_object_size": False,
    "randomize_object_mass": False,
}

#for 100 samples of box weight, size
#for 5 times each environment - get average success rate over 5
lifting_trials = []
#create grid of object sizes and weights
xsize = np.arange(0.1, .6, .1)
ysize = np.arange(0.1, .6, .1)
zsize = np.arange(0.5, 1.25, .1)
mass = np.arange(5, 20, 1)

X, Y, Z, M = np.meshgrid(xsize, ysize, zsize, mass)

X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()
M = M.flatten()

#iterate over all combinations
for x, y, z, m in tqdm(zip(X, Y, Z, M)):
    print(f"Running trial with object size: {x, y, z} and mass: {m}")
    env = build_env(config,
                    baseline=True,
                    render_mode="rgb_array",
                    object_size=[x, y, z],
                    object_mass=m)
    successes = []

    #get size and weight of object
    xsize, ysize, zsize = env.unwrapped.model.geom("box").size
    mass = env.unwrapped.model.body("box").mass

    trial_data = {
        "xsize": xsize,
        "ysize": ysize,
        "zsize": zsize,
        "mass": mass.item(),
    }

    for _ in range(10):
        model = OpenLoopHuggerPolicy(N=50)
        frames, rewards, actions, observations, infos = record_rollout(
            env, model, render=False)
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

    trial_data["success_rate"] = np.mean(successes)

    print(trial_data)

    env.close()

    #save data to file
    with open("lifting_trials.json", "a") as f:
        json.dump(trial_data, f)
        f.write("\n")
        f.close()

    env = None
    gc.collect()

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
