import argparse

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CallbackList, EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from dataclasses import dataclass
from baloo_gym.utils.helpers import make_parallel_env, build_env
from stable_baselines3.common.utils import set_random_seed


#just a dataclass to hold the run name and id for testing purposes only.
@dataclass
class Run:
    name: str
    id: str = "0000"


def train(args):

    set_random_seed(args.seed)

    config = {
        "total_timesteps": args.total_timesteps,
        "ctrl_timestep": .1,
        "env_name": args.env_name,
        "time_limit_sec": 30,
        "time_aware_obs": True,
        "curriculum_selection": args.curriculum_selection,
        'reward_selection': args.reward_selection,
        "randomize_initial_height": args.randomize_initial_height,
        "randomize_object_size": args.randomize_object_size,
        "randomize_object_mass": args.randomize_object_mass,
    }

    if args.wandb:

        if args.resume_training_runid:
            #restart saved run from wandb to log more metrics to
            run = wandb.init(project="ppo_baloo",
                             id=args.resume_training_runid,
                             resume="must",
                             config=config,
                             sync_tensorboard=True,
                             monitor_gym=True,
                             save_code=True,
                             tags=["success"])
        else:
            run = wandb.init(
                project="ppo_baloo",
                config=config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=
                True,  # auto-upload the videos of agents playing the game
                save_code=True,  # optional
                tags=["success"],
            )

        wandb.run.log_code("./baloo_gym/wrappers/")

        folder_name = f"{run.name}-{run.id}"
        wandb_callback = WandbCallback(
            model_save_path=f"./experiments/{folder_name}/recent_model",
            model_save_freq=100000,
            verbose=2,
        )

        #!throws a warning, but the env IS the same, just not vectorized to save RAM.
        eval_env = build_env(config,
                             run,
                             baseline=False,
                             monitor=True,
                             render_mode="rgb_array")

        # eval_callback = EvalCallback(
        #     eval_env=eval_env,
        #     n_eval_episodes=10,
        #     eval_freq=int(
        #         config["total_timesteps"] / 5 /
        #         args.num_envs),  #eval_num_timesteps = eval_freq * num_envs
        #     deterministic=True,
        #     best_model_save_path=f"./experiments/{folder_name}/best_model",
        #     render=False,
        #     verbose=1,
        # )

        # callback = CallbackList([eval_callback, wandb_callback])
        callback = CallbackList([wandb_callback])

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

    def linear_schedule(initial_value: float):

        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func

    policy_kwargs = dict(net_arch=[128, 128])

    if args.resume_training_runid:
        #download most recent saved model from wandb server
        api = wandb.Api()
        saved_run = api.run(
            f"curtiscjohnson/ppo_baloo/{args.resume_training_runid}")
        print(f"Downloading model...")

        saved_model = saved_run.file("model.zip").download(replace=True)
        rl_model = PPO.load(saved_model.name, env=env)
    else:
        rl_model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            # use_sde=True, #!rails outputs for some reason...
            batch_size=256,
            learning_rate=3e-4,
            ent_coef=.005,
            verbose=2,
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
        help=
        'List of rewards to use for training. Options: "tactile_nonzero", "action_smoothness", "arm_convex_hull", "rms_robot_dist", "chest_proximity',
    )

    parser.add_argument(
        "--curriculum_selection",
        nargs="+",
        type=str,
        default=[],
        help=
        "List of curriculums to use for training. Options: 'manipuland_initial_position'",
    )

    parser.add_argument('--randomize_initial_height',
                        action='store_true',
                        help='Randomize initial height of the elevator')

    parser.add_argument('--randomize_object_size',
                        action='store_true',
                        help='Randomize object size')

    parser.add_argument('--randomize_object_mass',
                        action='store_true',
                        help='Randomize object mass')

    parser.add_argument('--resume_training_runid',
                        type=str,
                        default=None,
                        help='wandb run id to resume training')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    print(args)

    if args.remote_train:
        import os
        os.environ["MUJOCO_GL"] = "egl"

    train(args)
