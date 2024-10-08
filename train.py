import argparse

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CallbackList, EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from dataclasses import dataclass
from baloo_gym.utils.helpers import build_env
from stable_baselines3.common.monitor import Monitor


@dataclass
class Run:
    name: str


def train():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Train a reinforcement learning model.")
    parser.add_argument('--total_timesteps',
                        type=int,
                        default=1000000,
                        help='Total timesteps for training')
    parser.add_argument('--num_envs',
                        type=int,
                        default=9,
                        help='Number of environments for SubprocVecEnv')
    parser.add_argument('--wandb',
                        action='store_true',
                        help='Use Weights and Biases for logging')
    parser.add_argument('--env_name',
                        type=str,
                        default='baloo_v3',
                        help='Name of the environment')
    parser.add_argument(
        '--remote_train',
        action='store_true',
        help=
        'Run training on remote server. Need to change mujoco graphics backend to egl.'
    )

    args = parser.parse_args()
    if args.remote_train:
        import os
        os.environ["MUJOCO_GL"] = "egl"

    print(args)

    config = {
        "total_timesteps": args.total_timesteps,
        "ctrl_timestep": .1,
        "env_name": args.env_name,
        "time_limit_sec": 30,
        "time_aware_obs": True,
    }

    def build_monitor_env():
        env = build_env(config)
        env = Monitor(env, f"./experiments/{run.name}/monitor_logs")
        return env

    def make_parallel_env():
        env = SubprocVecEnv([build_monitor_env for _ in range(args.num_envs)])

        from baloo_gym.wrappers.vec_env_record_video_wrapper import VecVideoRecorder
        total_episodes = config["total_timesteps"] / (
            (config["time_limit_sec"] / config["ctrl_timestep"]) *
            args.num_envs)

        ten_every_run = int(total_episodes / 10)

        env = VecVideoRecorder(
            env,
            f"./experiments/{run.name}/rollout_videos",
            record_video_trigger=lambda x: int(x % ten_every_run) == 0,
            video_length=config["time_limit_sec"] / config["ctrl_timestep"],
            name_prefix="rollout",
            wandb=args.wandb)

        return env

    if args.wandb:

        run = wandb.init(
            project="ppo_baloo",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=
            True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            tags=["carlo", "post-bug"],
            notes="")

        wandb.run.log_code("./wrappers/")

        wandb_callback = WandbCallback(
            model_save_path=f"./experiments/{run.name}/recent_model",
            model_save_freq=int(config["total_timesteps"] / 10 /
                                args.num_envs),
            verbose=1,
        )

        #!throws a warning, but the env IS the same, just not vectorized to save RAM.
        eval_env = build_monitor_env()

        eval_callback = EvalCallback(
            eval_env=eval_env,
            n_eval_episodes=5,
            eval_freq=int(
                config["total_timesteps"] / 10 /
                args.num_envs),  #eval_num_timesteps = eval_freq * num_envs
            deterministic=True,
            best_model_save_path=f"./experiments/{run.name}/best_model",
            render=False,
            verbose=1,
        )

        callback = CallbackList([eval_callback, wandb_callback])

    else:

        run = Run("test")
        callback = None

    env = make_parallel_env()

    rl_model = PPO(
        "MlpPolicy",
        env,
        ent_coef=0.01,
        #    use_sde=True, #usually for continuous action spaces.
        verbose=1,
        tensorboard_log=f"./experiments/{run.name}/runs",
    )

    rl_model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=callback,
    )

    if args.wandb:

        run.finish()


if __name__ == "__main__":
    train()
