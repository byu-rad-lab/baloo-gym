import argparse

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CallbackList, EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from dataclasses import dataclass
from baloo_gym.utils.helpers import make_parallel_env, build_env


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
                        default=1,
                        help='Number of environments for SubprocVecEnv')
    parser.add_argument('--wandb',
                        action='store_true',
                        help='Use Weights and Biases for logging')
    parser.add_argument('--env_name',
                        type=str,
                        default='baloo_v4',
                        help='Name of the environment')

    parser.add_argument(
        '--remote_train',
        action='store_true',
        help=
        'Run training on remote server. Need to change mujoco graphics backend to egl.'
    )

    parser.add_argument(
        '--reward_selection',
        nargs='+',
        type=str,
        default=[],
        help='List of rewards to use for training. Options: "tactile_nonzero"',
    )

    parser.add_argument(
        "--curriculum_selection",
        nargs="+",
        type=str,
        default=[],
        help=
        "List of curriculums to use for training. Options: 'manipuland_initial_position'",
    )

    args = parser.parse_args()
    print(args)

    if args.remote_train:
        import os
        os.environ["MUJOCO_GL"] = "egl"

    config = {
        "total_timesteps": args.total_timesteps,
        "ctrl_timestep": .1,
        "env_name": args.env_name,
        "time_limit_sec": 30,
        "time_aware_obs": True,
        "curriculum_selection": args.curriculum_selection,
        'reward_selection': args.reward_selection,
    }

    if args.wandb:

        run = wandb.init(
            project="ppo_baloo",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=
            True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            tags=["carlo", "post-bug", "all-in-one-reward"],
        )

        wandb.run.log_code("./baloo_gym/wrappers/")

        folder_name = f"{run.name}-{run.id}"
        wandb_callback = WandbCallback(
            model_save_path=f"./experiments/{folder_name}/recent_model",
            model_save_freq=int(config["total_timesteps"] / 10 /
                                args.num_envs),
            verbose=1,
        )

        #!throws a warning, but the env IS the same, just not vectorized to save RAM.
        eval_env = build_env(config,
                             run,
                             baseline=False,
                             monitor=True,
                             render_mode="rgb_array")

        eval_callback = EvalCallback(
            eval_env=eval_env,
            n_eval_episodes=10,
            eval_freq=int(
                config["total_timesteps"] / 5 /
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

    env = make_parallel_env(config,
                            folder_name,
                            baseline=False,
                            monitor=True,
                            num_envs=args.num_envs,
                            wandb=args.wandb,
                            record_video=True)

    policy_kwargs = dict(net_arch=[128, 128, 128])
    rl_model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        # use_sde=True, #!rails outputs for some reason...
        batch_size=256,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log=f"./experiments/{folder_name}/runs",
    )

    print("TRAINING MODEL")
    rl_model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=callback,
    )

    if args.wandb:
        run.finish()


if __name__ == "__main__":
    train()
