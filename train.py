import argparse

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback

from baloo_gym.utils.helpers import build_env
import copy


# Parallel environments
def train(args):
    set_random_seed(args.seed, using_cuda=True)

    run_folder = "test-0000"

    save_freq = 500000 // args.num_envs

    config = {
        "total_timesteps": args.total_timesteps,
        "ctrl_timestep": .05,
        "env_name": args.env_name,
        "time_limit_sec": 60,
        "curriculum_selection": args.curriculum_selection,
        'reward_selection': args.reward_selection,
        "randomize_initial_height": args.randomize_initial_height,
        "randomize_object_size": args.randomize_object_size,
        "randomize_object_mass": args.randomize_object_mass,
        "randomize_object_quat": args.randomize_object_quat,
    }

    callbacks = []
    if args.wandb:
        run = wandb.init(
            project="ppo_baloo",
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            tags=["success", "sde_working"],
            dir="new_experiments",
            group="potential_based_reward"
            if args.potential_based_reward else None,
        )

        run_folder = f"{run.name}-{run.id}"

        wandb.run.log_code("./src/baloo_gym/wrappers/")

        callbacks.append(
            WandbCallback(
                gradient_save_freq=save_freq,
                model_save_freq=save_freq,
                model_save_path=f"new_experiments/{run_folder}/recent_model",
                verbose=2,
            ))

    #automatically wraps each environment in a monitor
    vec_env = make_vec_env(
        build_env,
        env_kwargs={
            "config": config,
            "baseline": False,
            "render_mode": "rgb_array",
            "potential_based_reward": args.potential_based_reward,
        },
        n_envs=args.num_envs,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=f"new_experiments/{run_folder}/monitor",
        monitor_kwargs={
            'info_keywords': ('is_success', ),
        },
        seed=args.seed,
    )

    print("Building evaluation environment...")
    eval_config = copy.deepcopy(config)
    eval_config["randomize_initial_height"] = False
    eval_env = build_env(eval_config,
                         baseline=False,
                         render_mode="rgb_array",
                         monitor=True)

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=10,
        eval_freq=save_freq,
        log_path=f"new_experiments/{run_folder}/eval_logs",
        best_model_save_path=f"new_experiments/{run_folder}/best_model",
        deterministic=True,
        verbose=1,
    )
    callbacks.append(eval_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // 2,
        save_path=f"new_experiments/{run_folder}/checkpoints",
        verbose=2,
    )

    callbacks.append(checkpoint_callback)

    vec_env = VecVideoRecorder(
        vec_env,
        f"new_experiments/{run_folder}/videos",
        record_video_trigger=lambda x: x % save_freq == 0,
        video_length=30 / config["ctrl_timestep"],
        name_prefix=run_folder,
    )

    def linear_schedule(initial_value: float, final_value: float):
        """
        Linear learning rate schedule from initial to final value.

        :param initial_value: Initial learning rate.
        :param final_value: Final learning rate at the end of training.
        :return: schedule that computes current learning rate depending on remaining progress
        """

        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining: Remaining progress as a fraction from 1 to 0.
            :return: current learning rate
            """
            return progress_remaining * (initial_value -
                                         final_value) + final_value

        return func

    policy_kwargs = dict(
        net_arch=[512, 512],
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=8192,
        gamma=0.995,
        policy_kwargs=policy_kwargs,
        batch_size=512,
        learning_rate=linear_schedule(5e-4, 1e-5),
        ent_coef=0.0,
        verbose=2,
        tensorboard_log=f"new_experiments/{run_folder}/tensorboard_logs",
        device="auto",
    )

    if args.resume_training_runid:
        #download most recent saved model from wandb server to override default model
        api = wandb.Api()
        saved_run = api.run(
            f"curtiscjohnson/ppo_baloo/{args.resume_training_runid}")
        print(f"Downloading model...")

        saved_model = saved_run.file("model.zip").download(replace=True)

        model = PPO.load(
            saved_model.name,
            env=vec_env,
        )

    print("BEGINNING TRAINING")
    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=callbacks,
    )

    if args.wandb:
        #log best model to wandb too
        wandb.save(f"new_experiments/{run_folder}/best_model/best_model.zip")
        wandb.save(f"new_experiments/{run_folder}/checkpoints/*.zip")
        run.finish()

    vec_env.close()


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Train a reinforcement learning model.")

    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=1000000,
        help='Total timesteps for training',
    )
    parser.add_argument(
        '--num_envs',
        type=int,
        default=1,
        help='Number of environments for SubprocVecEnv',
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Use Weights and Biases for logging',
    )
    parser.add_argument(
        '--env_name',
        type=str,
        default='baloo_v9',
        help='Name of the environment',
    )

    parser.add_argument(
        '--remote_train',
        action='store_true',
        help=
        'Run training on remote server. Need to change mujoco graphics backend to egl.',
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

    parser.add_argument(
        '--randomize_initial_height',
        action='store_true',
        help='Randomize initial height of the elevator',
    )

    parser.add_argument(
        '--randomize_object_size',
        action='store_true',
        help='Randomize object size',
    )

    parser.add_argument(
        '--randomize_object_mass',
        action='store_true',
        help='Randomize object mass',
    )

    parser.add_argument(
        '--randomize_object_quat',
        action='store_true',
        help='Randomize object quaternion',
    )

    parser.add_argument(
        '--resume_training_runid',
        type=str,
        default=None,
        help='wandb run id to resume training',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )

    parser.add_argument(
        '--potential_based_reward',
        action='store_true',
        help='Use potential based rewards',
    )

    args = parser.parse_args()
    print(args)

    if args.remote_train:
        import os
        os.environ["MUJOCO_GL"] = "egl"

    train(args)


if __name__ == "__main__":
    main()
