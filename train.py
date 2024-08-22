from envs.baloo_v0 import BalooV0
from gymnasium.wrappers import RecordVideo, TimeLimit, TimeAwareObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import importlib
import wandb
from wandb.integration.sb3 import WandbCallback

from dataclasses import dataclass


@dataclass
class Run:
    id: str


if __name__ == "__main__":

    USE_WANDB = False

    config = {
        "total_timesteps": 100000,
        "ctrl_timestep": .1,
        "env_name": "baloo_v1",
        "class_name": "BalooV1",
        "time_limit_sec": 15,
        "time_aware_obs": True,
        "reward_signal": "-distance approach, subproc env test",
    }

    if USE_WANDB:

        run = wandb.init(
            project="ppo_baloo",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=
            True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            tags=["carlo"],
        )

    else:

        run = Run("test")

    def make_env():
        EnvClass = getattr(
            importlib.import_module(f"envs.{config['env_name']}"),
            config['class_name'])

        env = EnvClass(render_mode="rgb_array",
                       camera_name="fixedcam",
                       ctrl_timestep=config["ctrl_timestep"])

        check_env(env)
        '''
        When using time aware observation, I should use terminated instead of truncated. 
        #! requires float 32, but then np.appends self.t on line 51, which numpy casts as float64. 
        #! Looks like this wrapper will change alot with new release of gymnasium (not on pip yet). this is v0.29.1.
        '''
        if config["time_aware_obs"]:
            env = TimeAwareObservation(env)

        from wrappers.three_part_reward_wrapper import ThreePartRewardWrapper
        env = ThreePartRewardWrapper(env)

        env = RecordVideo(env,
                          f"./experiments/rollout_videos/{run.id}",
                          episode_trigger=lambda x: x % 20 == 0)

        return env

    env = make_env()

    # env = SubprocVecEnv([make_env for _ in range(4)])

    # while True:
    #     action = env.action_space.sample()
    #     # action = [1, -1] * int(env.action_space.shape[0] / 2)
    #     # print(action)

    #     obs, reward, terminated, truncated, info = env.step(action)
    #     print(
    #         f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    #     env.render()

    rl_model = PPO("MlpPolicy",
                   env,
                   verbose=1,
                   tensorboard_log=f"./experiments/runs/{run.id}")
    rl_model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=WandbCallback(
            # gradient_save_freq=100,
            model_save_path=f"./experiments/models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()
