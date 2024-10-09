import argparse
import importlib

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CallbackList, EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from dataclasses import dataclass
from baloo_gym.utils.helpers import make_parallel_env


#just a dataclass to hold the run name and id for testing purposes only.
@dataclass
class Run:
    name: str
    id: str = "0000"


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
    print(args)

    if args.remote_train:
        import os
        os.environ["MUJOCO_GL"] = "egl"

    name2class = {
        'baloo_v0': 'BalooV0',
        'baloo_v1': 'BalooV1',
        'baloo_v2': 'BalooV2',
        'baloo_v3': 'BalooV3',
        'baloo_v4': 'BalooV4',
    }

    class_name = name2class[args.env_name]
    EnvClass = getattr(
        importlib.import_module(f"baloo_gym.envs.{args.env_name}"), class_name)

    config = {
        "total_timesteps": args.total_timesteps,
        "ctrl_timestep": .1,
        "EnvClass": EnvClass,
        "time_limit_sec": 30,
        "time_aware_obs": True,
    }

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

        folder_name = f"{run.name}-{run.id}"
        wandb_callback = WandbCallback(
            model_save_path=f"./experiments/{folder_name}/recent_model",
            model_save_freq=int(config["total_timesteps"] / 10 /
                                args.num_envs),
            verbose=1,
        )

        #!throws a warning, but the env IS the same, just not vectorized to save RAM.
        eval_env = build_monitor_env(config, run)

        eval_callback = EvalCallback(
            eval_env=eval_env,
            n_eval_episodes=5,
            eval_freq=int(
                config["total_timesteps"] / 10 /
                args.num_envs),  #eval_num_timesteps = eval_freq * num_envs
            deterministic=True,
            best_model_save_path=f"./experiments/{folder_name}/best_model",
            render=False,
            verbose=1,
        )

        callback = CallbackList([eval_callback, wandb_callback])

    else:

        run = Run("test")
        folder_name = f"{run.name}-{run.id}"
        callback = None

    env = make_parallel_env(config=config,
                            run=run,
                            baseline=False,
                            monitor=True,
                            num_envs=args.num_envs,
                            wandb=args.wandb)

    rl_model = PPO(
        "MlpPolicy",
        env,
        ent_coef=0.01,
        #    use_sde=True, #usually for continuous action spaces.
        verbose=1,
        tensorboard_log=f"./experiments/{folder_name}/runs",
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
