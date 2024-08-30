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
    name: str


if __name__ == "__main__":

    USE_WANDB = False

    config = {
        "total_timesteps": 1000000,
        "ctrl_timestep": .1,
        "env_name": "baloo_v1",
        "class_name": "BalooV1",
        "time_limit_sec": 30,
        "time_aware_obs": True,
        "reward_signal": "only lift reward",
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

        # #Ithink all of the subprocesses try to write to the same file...
        # env = RecordVideo(
        #     env,
        #     episode_trigger=lambda x: x % 10 == 0,
        #     video_folder=f"./experiments/{run.name}/rollout_videos",
        #     name_prefix="recorded")

        env = Monitor(env, f"./experiments/{run.name}/monitor_logs")

        return env

    env = SubprocVecEnv([make_env for _ in range(8)])

    # #drops to 60 it/s from almost 200 it/s with 8 envs. ouch.
    #doesn't have episode trigger, only step trigger. meh.
    from wrappers.vec_env_record_video_wrapper import VecVideoRecorder
    env = VecVideoRecorder(env,
                           f"./experiments/{run.name}/rollout_videos",
                           record_video_trigger=lambda x: x % 2 == 0,
                           video_length=config["time_limit_sec"] /
                           config["ctrl_timestep"],
                           name_prefix="rollout")

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

    # #save a video and upload it to wandb
    # env = make_env()
    # frames, rewards = record_rollout(env, rl_model)

    # import matplotlib.pyplot as plt
    # plt.plot(rewards)
    # plt.xlabel("Time Step")
    # plt.ylabel("Reward")

    # #save video to disk
    # import imageio
    # video_path = f"./experiments/{run.name}/rollout_video.mp4"
    # imageio.mimsave(video_path, frames, fps=1 / config["ctrl_timestep"])
    # plt.savefig(f"./experiments/{run.name}/rollout_rewards.png")

    # if USE_WANDB:
    #     wandb.log({
    #         "rollout_video":
    #         wandb.Video(video_path,
    #                     fps=1 / config['ctrl_timestep'],
    #                     format="mp4")
    #     })

    #     wandb.log({"rewards": plt})

    #     run.finish()
