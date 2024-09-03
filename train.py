import argparse
from gymnasium.wrappers import TimeAwareObservation
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import importlib
import wandb
from wandb.integration.sb3 import WandbCallback

from dataclasses import dataclass


@dataclass
class Run:
    name: str


if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Train a reinforcement learning model.")
    parser.add_argument('--total_timesteps',
                        type=int,
                        default=1000000,
                        help='Total timesteps for training')
    parser.add_argument('--num_envs',
                        type=int,
                        default=16,
                        help='Number of environments for SubprocVecEnv')
    parser.add_argument('--use_wandb',
                        type=bool,
                        default=True,
                        help='Use Weights and Biases for logging')
    args = parser.parse_args()

    USE_WANDB = args.use_wandb

    config = {
        "total_timesteps": args.total_timesteps,
        "ctrl_timestep": .1,
        "env_name": "baloo_v1",
        "class_name": "BalooV1",
        "time_limit_sec": 30,
        "time_aware_obs": True,
        "reward_signal": "approach, sensor, lift",
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

        wandb.run.log_code("./wrappers/")

        callback = WandbCallback(
            # gradient_save_freq=100,
            model_save_path=f"./experiments/{run.name}/model",
            verbose=2,
        )

    else:

        run = Run("test")
        callback = None

    def make_env():
        EnvClass = getattr(
            importlib.import_module(f"envs.{config['env_name']}"),
            config['class_name'])

        env = EnvClass(render_mode="rgb_array",
                       camera_name="fixedcam",
                       ctrl_timestep=config["ctrl_timestep"],
                       render_width=320,
                       render_height=240)

        check_env(env)
        '''
        When using time aware observation, I should use terminated instead of truncated. 
        #! requires float 32, but then np.appends self.t on line 51, which numpy casts as float64. 
        #! Looks like this wrapper will change alot with new release of gymnasium (not on pip yet). this is v0.29.1.
        '''
        if config["time_aware_obs"]:
            env = TimeAwareObservation(env)

        from wrappers.time_limit_termination_wrapper import TimeLimitTerminationWrapper
        env = TimeLimitTerminationWrapper(env, config["time_limit_sec"])

        from wrappers.three_part_reward_wrapper import ThreePartRewardWrapper
        env = ThreePartRewardWrapper(env)

        env = Monitor(env, f"./experiments/{run.name}/monitor_logs")

        return env

    env = SubprocVecEnv([make_env for _ in range(args.num_envs)])

    from wrappers.vec_env_record_video_wrapper import VecVideoRecorder
    env = VecVideoRecorder(env,
                           f"./experiments/{run.name}/rollout_videos",
                           record_video_trigger=lambda x: x % 50 == 0,
                           video_length=config["time_limit_sec"] /
                           config["ctrl_timestep"],
                           name_prefix="rollout",
                           wandb=USE_WANDB)

    # env = make_env()
    # time = 0
    # env.reset()
    # while True:
    #     action = env.action_space.sample()
    #     # action = [1, -1] * int(env.action_space.shape[0] / 2)
    #     # print(action)

    #     action[0] = -1
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     print(
    #         f"Time: {time}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
    #     )
    #     env.render()
    #     time += config["ctrl_timestep"]

    rl_model = PPO("MlpPolicy",
                   env,
                   verbose=2,
                   tensorboard_log=f"./experiments/{run.name}/runs")
    rl_model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=callback,
    )

    if USE_WANDB:

        run.finish()
