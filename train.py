import argparse
from gymnasium.wrappers import TimeAwareObservation
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
import importlib
import wandb
from wandb.integration.sb3 import WandbCallback
from baloo_mujoco_sim.utils.baloo_mj_api import get_elevator_height

from dataclasses import dataclass
from wrappers.time_limit_termination_wrapper import TimeLimitTerminationWrapper
from wrappers.three_part_reward_wrapper import ThreePartRewardWrapper


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
    parser.add_argument('--wandb',
                        action='store_true',
                        help='Use Weights and Biases for logging')
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
        "env_name": "baloo_v2",
        "class_name": "BalooV2",
        "time_limit_sec": 30,
        "time_aware_obs": True,
    }

    def build_env():
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

        env = TimeLimitTerminationWrapper(env, config["time_limit_sec"])

        env = ThreePartRewardWrapper(env)

        env = Monitor(env, f"./experiments/{run.name}/monitor_logs")
        return env

    def make_parallel_env():
        env = SubprocVecEnv([build_env for _ in range(args.num_envs)])

        from wrappers.vec_env_record_video_wrapper import VecVideoRecorder
        env = VecVideoRecorder(env,
                               f"./experiments/{run.name}/rollout_videos",
                               record_video_trigger=lambda x: x % 50 == 0,
                               video_length=config["time_limit_sec"] /
                               config["ctrl_timestep"],
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
            tags=["carlo"],
        )

        wandb.run.log_code("./wrappers/")

        wandb_callback = WandbCallback(
            # gradient_save_freq=100,
            model_save_path=f"./experiments/{run.name}/model",
            verbose=2,
        )

        #make separate evaluation environment for evaluation, not parallelized
        eval_env = build_env()

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

        callback = CallbackList([wandb_callback, eval_callback])

    else:

        run = Run("test")
        callback = None

    env = make_parallel_env()

    rl_model = PPO(
        "MlpPolicy",
        env,
        ent_coef=0.01,
        #    use_sde=True, #usually for continuous action spaces.
        verbose=2,
        tensorboard_log=f"./experiments/{run.name}/runs")

    rl_model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=callback,
    )

    if args.wandb:

        run.finish()
