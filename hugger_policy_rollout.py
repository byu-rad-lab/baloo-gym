from baloo_gym.policies.open_loop_hugger import OpenLoopHuggerPolicy
from baloo_gym.utils.helpers import build_env, record_rollout, make_movie

#just pick a config right now. The config is only really used for the environment name.

config = {
    "total_timesteps": 1000000,
    "ctrl_timestep": .1,
    "env_name": "baloo_v4",
    "time_limit_sec": 30,
    "time_aware_obs": True,
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
