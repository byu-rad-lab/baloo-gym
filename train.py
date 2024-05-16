from envs.baloo_v0 import BalooV0
from gymnasium.wrappers import RecordVideo, TimeLimit, TimeAwareObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# deprecation warning https://github.com/pytorch/pytorch/issues/84712

config = {
    "total_timesteps": 500000,
    "env_name": "BalooGymEnv",
    "time_limit_sec": 5,
    "time_aware_obs": True,
}

# run = wandb.init(
#     project="ppo_baloo",
#     config=config,
#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#     monitor_gym=True,  # auto-upload the videos of agents playing the game
#     save_code=True,  # optional
# )


def make_env():
    env = BalooV0("human",
                  camera_name="fixedcam",
                  xml_path="/home/curtis/baloo_gym/assets/baloo.xml")
    check_env(env)

    env = TimeLimit(
        env, max_episode_steps=config["time_limit_sec"] /
        env.control_timestep)  # must come before TimeAwareObservationV0

    if config["time_aware_obs"]:
        env = TimeAwareObservation(env)  #! causes chagne to float64

    # from wrappers.force_reward_wrapper import ForceRewardWrapper
    # env = ForceRewardWrapper(env)

    # env = RecordVideo(
    #     env, f"./rollout_videos/{5}", episode_trigger=lambda x: x % 100 == 0
    # ) #!causes change to float64
    return env


env = make_env()

obs, info = env.reset()
env.render()

while True:
    # action = env.action_space.sample()
    action = [1, -1] * int(env.action_space.shape[0] / 2)
    action.insert(1, 0)
    print(action)

    obs, reward, terminated, truncated, info = env.step(action)
    print(env.unwrapped.data.ctrl)
    env.render()

# rl_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
# rl_model.learn(
#     total_timesteps=config["total_timesteps"],
#     progress_bar=True,
#     callback=WandbCallback(
#         gradient_save_freq=100,
#         model_save_path=f"models/{run.id}",
#         verbose=2,
#     ),
# )
# rl_model.save("ppo_baloo_first_try")
# run.finish()
