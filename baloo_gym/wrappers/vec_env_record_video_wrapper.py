import os
from typing import Callable

from gymnasium.wrappers.monitoring import video_recorder

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from stable_baselines3.common.vec_env.base_vec_env import tile_images
import numpy as np
import matplotlib.pyplot as plt
import wandb


def plot2img(plt):
    fig = plt.gcf()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img


class VecVideoRecorder(VecEnvWrapper):
    """
    Wraps a VecEnv or VecEnvWrapper object to record rendered image as mp4 video.
    It requires ffmpeg or avconv to be installed on the machine.

    :param venv:
    :param video_folder: Where to save videos
    :param record_video_trigger: Function that defines when to start recording.
                                        The function takes the current number of episodes,
                                        and returns whether we should start recording or not.
    :param video_length:  Length of recorded videos
    :param name_prefix: Prefix to the video name
    """
    def __init__(
        self,
        venv: VecEnv,
        video_folder: str,
        record_video_trigger: Callable[[int], bool],
        video_length: int = 200,
        name_prefix: str = "rl-video",
        wandb=False,
    ):
        VecEnvWrapper.__init__(self, venv)

        self.env = venv
        # Temp variable to retrieve metadata
        temp_env = venv

        # Unwrap to retrieve metadata dict
        # that will be used by gym recorder
        while isinstance(temp_env, VecEnvWrapper):
            temp_env = temp_env.venv

        if isinstance(temp_env, DummyVecEnv) or isinstance(
                temp_env, SubprocVecEnv):
            metadata = temp_env.get_attr("metadata")[0]
        else:
            metadata = temp_env.metadata

        self.env.metadata = metadata
        assert self.env.render_mode == "rgb_array", f"The render_mode must be 'rgb_array', not {self.env.render_mode}"

        self.record_video_trigger = record_video_trigger
        self.video_recorder = None

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.episode_id = 0
        self.step_id = 0
        self.video_length = int(video_length)

        self.recording = False
        self.recorded_frames = 0

        self.recorded_rewards = []
        self.USEWANBD = wandb

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs

    def start_video_recorder(self) -> None:
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-episode-{self.episode_id}"
        print(f"Recording {video_name}.mp4...")
        base_path = os.path.join(self.video_folder, video_name)

        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={
                "episode_id": self.episode_id,
                "video_length": self.video_length
            },
        )
        self.recording = True

    def _video_enabled(self) -> bool:
        return self.record_video_trigger(self.episode_id)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()

        # capture list of reward trajectories, one for each env.
        #convert those into plots, then images, then tile_images to save as big_image that matches video.

        #!doesn't work if any envs truncate--restarts and rewards are out of sync but will be a lot of work to fix.
        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            self.recorded_rewards.append(rews)
            if any(dones):
                self.recorded_rewards = np.array(self.recorded_rewards)
                rew_imgs = []
                for i in range(self.recorded_rewards.shape[1]):
                    plt.plot(np.arange(0, self.recorded_rewards.shape[0]),
                             self.recorded_rewards[:, i])
                    plt.xlabel("Time Step")
                    plt.ylabel("Reward")
                    rew_imgs.append(plot2img(plt))
                    plt.clf()

                big_rew_img = tile_images(rew_imgs)
                filename = f"{self.video_recorder.path}.png"
                plt.imsave(filename, big_rew_img)

                if self.USEWANBD:
                    wandb.log({f"rollout_rewards": wandb.Image(filename)},
                              commit=False)
                #commit=false to log png and video on same wandb step

                print(f"Saving video to {self.video_recorder.path}")
                self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        self.step_id += 1
        # print(self.step_id)
        if any(dones):
            self.episode_id += 1
            print(f"Episode {self.episode_id}")

        return obs, rews, dones, infos

    def close_video_recorder(self) -> None:
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 0
        self.recorded_rewards = []

    def close(self) -> None:
        VecEnvWrapper.close(self)
        self.close_video_recorder()

    def __del__(self):
        self.close_video_recorder()
