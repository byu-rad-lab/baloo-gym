import importlib
from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_box_position,
    get_box_vel,
    get_elevator_height,
    get_elevator_vel,
    get_joint_angles,
    get_joint_vel,
)

from stable_baselines3.common.env_checker import check_env
from wrappers.time_limit_termination_wrapper import TimeLimitTerminationWrapper
from wrappers.three_part_reward_wrapper import ThreePartRewardWrapper
from gymnasium.wrappers import TimeAwareObservation

import numpy as np


def get_sensor_data(model, data):
    left_pos = []
    left_vel = []
    right_pos = []
    right_vel = []
    for i in range(3):
        left_pos.append(get_joint_angles(model, data, "left", i))
        left_vel.append(get_joint_vel(model, data, "left", i))
        right_pos.append(get_joint_angles(model, data, "right", i))
        right_vel.append(get_joint_vel(model, data, "right", i))

    object_pos = get_box_position(model, data)
    object_vel = get_box_vel(model, data)
    elevator_pos = get_elevator_height(model, data)
    elevator_vel = get_elevator_vel(model, data)

    return {
        "object_pos": object_pos,
        "object_vel": object_vel,
        "elevator_pos": elevator_pos,
        "elevator_vel": elevator_vel,
        "left_pos": np.hstack(left_pos),
        "right_pos": np.hstack(right_pos),
        "left_vel": np.hstack(left_vel),
        "right_vel": np.hstack(right_vel),
    }


def record_rollout(env, policy):
    obs, info = env.reset()
    done = False

    frames = []
    rewards = []
    actions = []
    observations = []

    while not done:
        action, _states = policy.predict(obs)
        observations.append(obs)
        actions.append(action)
        frames.append(env.render())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(reward)

    env.close()

    return frames, rewards, actions, observations


def make_movie(frames: list, filename: str, fps=30):
    import moviepy.editor as mpy

    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filename)


def build_env(config: dict, render_mode: str = "rgb_array"):
    """Builds a gym environment with the given configuration.

    Args:
        config (dict):
            total_timesteps (int): The total number of timesteps to train the model.
            ctrl_timestep (float): The duration of a single control timestep.
            env_name (str): The name of the environment to build.
            class_name (str): The name of the class to instantiate.
            time_limit_sec (float): The time limit for each episode.
            time_aware_obs (bool): Whether to use time-aware observations.
    """

    EnvClass = getattr(importlib.import_module(f"envs.{config['env_name']}"),
                       config['class_name'])

    env = EnvClass(render_mode=render_mode,
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

    return env
