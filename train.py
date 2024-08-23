from envs.baloo_v0 import BalooV0
from gymnasium.wrappers import RecordVideo, TimeLimit, TimeAwareObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import importlib
import wandb
from wandb.integration.sb3 import WandbCallback

from dataclasses import dataclass

from utils.helpers import record_rollout


@dataclass
class Run:
    id: str


if __name__ == "__main__":

    USE_WANDB = False

    config = {
        "total_timesteps": 2000,
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

        callback = WandbCallback(
            # gradient_save_freq=100,
            model_save_path=f"./experiments/{run.id}/model",
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
                       ctrl_timestep=config["ctrl_timestep"])

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

        #! this eats up a lot of RAM when parallelized
        # env = RecordVideo(env,
        #                   f"./experiments/{run.id}/rollout_videos",
        #                   episode_trigger=lambda x: x % 20 == 0)

        env = Monitor(env, f"./experiments/{run.id}/monitor_logs")

        return env

    env = make_env()

    # env = SubprocVecEnv([make_env for _ in range(10)])

    # time = 0
    # env.reset()
    # while True:
    #     action = env.action_space.sample()
    #     # action = [1, -1] * int(env.action_space.shape[0] / 2)
    #     # print(action)

    #     obs, reward, terminated, truncated, info = env.step(action)
    #     print(
    #         f"Time: {time}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
    #     )
    # env.render()
    #     time += config["ctrl_timestep"]

    rl_model = PPO("MlpPolicy",
                   env,
                   verbose=2,
                   tensorboard_log=f"./experiments/{run.id}/runs")
    rl_model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=callback,
    )

    #save a video and upload it to wandb
    frames, rewards = record_rollout(env, rl_model)

    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel("Time Step")
    plt.ylabel("Reward")

    if USE_WANDB:
        wandb.log({"rollout_video": wandb.Video(frames, fps=30, format="mp4")})

        wandb.log({"rewards": plt})
        run.finish()
    else:
        #save video to disk
        import imageio
        imageio.mimsave(f"./experiments/{run.id}/rollout_video.mp4",
                        frames,
                        fps=1 / config["ctrl_timestep"])
        plt.savefig(f"./experiments/{run.id}/rollout_rewards.png")
