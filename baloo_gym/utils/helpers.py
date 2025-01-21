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
from baloo_gym.wrappers import OpenLoopBaselineWrapper, ThreePartRewardWrapper

from gymnasium.wrappers import TimeLimit

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder


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


def record_rollout(env, policy, render=True):
    obs, info = env.reset()
    # print(f"In rollout, obs shape: {obs.shape}")
    done = False

    frames = []
    rewards = []
    actions = []
    observations = []
    infos = []

    while not done:
        action, _states = policy.predict(obs, deterministic=False)
        observations.append(obs)
        actions.append(action)
        if render:
            frames.append(env.render())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        infos.append(info)

    env.close()

    return frames, rewards, actions, observations, infos


def make_movie(frames: list, filename: str, fps=30):
    import moviepy.editor as mpy

    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filename)


def build_env(config: dict, baseline: bool, render_mode, **kwargs):
    """Builds a gym environment with the given configuration.

    Args:
        config (dict):
            total_timesteps (int): The total number of timesteps to train the model.
            ctrl_timestep (float): The duration of a single control timestep.
            env_name (str): The name of the environment to build.
            time_limit_sec (float): The time limit for each episode.
            time_aware_obs (bool): Whether to use time-aware observations.
    """

    name2class = {
        "baloo_v0": "BalooV0",
        "baloo_v1": "BalooV1",
        "baloo_v2": "BalooV2",
        "baloo_v3": "BalooV3",
        "baloo_v4": "BalooV4",
        "baloo_v5": "BalooV5",
        "baloo_v6": "BalooV6",
        "baloo_v7": "BalooV7",
        "baloo_v8": "BalooV8",
    }

    EnvClass = getattr(
        importlib.import_module(f"baloo_gym.envs.{config['env_name']}"),
        name2class[config["env_name"]])

    env = EnvClass(
        render_mode=render_mode,
        camera_name="fixedcam",
        ctrl_timestep=config["ctrl_timestep"],
        render_width=320,
        render_height=240,
        randomize_initial_height=config["randomize_initial_height"],
        randomize_object_size=config["randomize_object_size"],
        randomize_object_mass=config["randomize_object_mass"],
        object_size=kwargs.get("object_size", None),
        object_mass=kwargs.get("object_mass", None),
    )

    check_env(env)

    #emit truncation signal at the end of the episode.
    env = TimeLimit(env,
                    max_episode_steps=int(
                        config["time_limit_sec"] / config["ctrl_timestep"], ))

    env = ThreePartRewardWrapper(env, config["reward_selection"])
    # env = CurriculumEnv(env, config["curriculum_selection"])

    if baseline:
        #overwrite to be compatible with open-loop baseline policy.
        env = OpenLoopBaselineWrapper(env)
        print("Using open-loop baseline policy.")

    return env


def parallelize_env(args, config, run_folder, save_freq):
    #automatically wraps each environment in a monitor
    vec_env = make_vec_env(
        build_env,
        env_kwargs={
            "config": config,
            "baseline": False,
            "render_mode": "rgb_array",
        },
        n_envs=args.num_envs,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=f"new_experiments/{run_folder}/monitor",
        monitor_kwargs={
            'info_keywords': ('is_success', ),
        },
    )

    vec_env = VecVideoRecorder(
        vec_env,
        f"new_experiments/{run_folder}/videos",
        record_video_trigger=lambda x: x % save_freq == 0,
        video_length=30 / config["ctrl_timestep"],
        name_prefix=run_folder,
    )

    return vec_env
