import gymnasium as gym
from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_contact_forces_on_body, get_tactile_image, get_box_position,
    get_box_vel, detect_box_touch, get_joint_angles, get_box_quat,
    get_elevator_vel)
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


class ThreePartRewardWrapper(gym.Wrapper):
    """
    Now this class can just overrides the calculate_reward() method to return a force based reward. 
    """
    def __init__(self, env):
        """Constructor for the Reward wrapper."""
        super().__init__(env)

        #get nominal box position
        self.box_pos = get_box_position(self.env.unwrapped.model,
                                        self.env.unwrapped.data)

        self.desired_box_pos = self.box_pos.copy()
        self.desired_box_pos[2] += 0.5
        self.prev_box_error = np.linalg.norm(self.desired_box_pos -
                                             self.box_pos)

        self.box_quat = R.from_quat(
            get_box_quat(self.env.unwrapped.model,
                         self.env.unwrapped.data,
                         scalar_first=False))

        self.state = 'approach'

    def step(self, action):
        """Step function that calls the parent step function and then calculates the reward."""
        # call baloo_base step function
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        reward = self.calculate_reward()

        # #truncate if box is lifted, box is far away, or if box tipped over. This makes it really hard to record videos...
        # if np.linalg.norm(self.desired_box_pos - get_box_position(
        #         self.env.unwrapped.model, self.env.unwrapped.data)) < 0.1:
        #     terminated = True

        # if np.linalg.norm(self.box_pos - get_box_position(
        #         self.env.unwrapped.model, self.env.unwrapped.data)) > 0.5:
        #     truncated = True

        # if (self.box_quat.inv() * R.from_quat(
        #         get_box_quat(self.env.unwrapped.model,
        #                      self.env.unwrapped.data,
        #                      scalar_first=False))).magnitude() > 1:
        #     truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset function that calls the parent reset function and then calculates the reward."""
        # call baloo_base reset function
        self.state = 'approach'
        return self.env.reset()

    def _cosine_similarity(self, v1, v2):
        if np.linalg.norm(v1) <= 1e-6 or np.linalg.norm(v2) <= 1e-6:
            return 0

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def calculate_reward(self):
        """
        Calculates the reward to return. Used with Carlo Alessi at SSSA
        """

        reward = 0

        chest_pos = self.env.unwrapped.data.geom('chest').xpos
        box_pos = get_box_position(self.env.unwrapped.model,
                                   self.env.unwrapped.data)

        #shaped reward to approach box
        reward += np.exp(-1 * np.linalg.norm(chest_pos - box_pos))

        box_error = np.linalg.norm(self.desired_box_pos - box_pos)

        # current_box_quat = R.from_quat(
        #     get_box_quat(self.env.unwrapped.model, self.env.unwrapped.data,scalar_first=False)
        # )

        # #orientation error
        # orientation_error = (current_box_quat.inv() * self.box_quat).magnitude() #always [0,pi]

        # #if chest is close to box, reward for lifting object
        if (self.prev_box_error - box_error) > 0:
            #if box is getting closer, reward
            reward += 1
        # if np.linalg.norm(chest_pos - box_pos) < 0.5:

    #         self.env.unwrapped.model.geom('box').rgba = [0, 1, 0, 1]
        #     box_vel = get_box_vel(self.env.unwrapped.model,
        #                           self.env.unwrapped.data)

        #     if box_vel[2] > 1e-2:
        #         #turn box green and +1
        #         reward += 1

        else:
            #box back to red, no reward if not lifted
            self.env.unwrapped.model.geom('box').rgba = [1, 0, 0, 1]

        #update previous box error
        self.prev_box_error = box_error

        self.env.unwrapped.model.geom('box').rgba = [1, 0, 0, 1]

        return reward
