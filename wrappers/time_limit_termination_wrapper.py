import gymnasium as gym
from baloo_mujoco_sim.utils.baloo_mj_api import get_contact_forces_on_body, get_tactile_image, get_box_position
import numpy as np
import mujoco


class TimeLimitTerminationWrapper(gym.Wrapper):
    """
    Now this class can just overrides the calculate_reward() method to return a force based reward. 
    """
    def __init__(self, env, time_limit_sec, control_timestep):
        """Constructor for the Reward wrapper."""
        super().__init__(env)
        self.time_limit_sec = time_limit_sec
        self.step_count = 0
        self.control_timestep = control_timestep

    def step(self, action):
        """Step function that calls the parent step function and then calculates the reward."""

        # call baloo_base step function
        observation, reward, terminated, truncated, info = self.env.step(
            action)

        self.step_count += 1
        print(self.step_count)

        if self.step_count == int(self.time_limit_sec / self.control_timestep):
            terminated = True

        return observation, reward, terminated, truncated, info

    def reset(self):
        self.step_count = 0
        return self.env.reset()
