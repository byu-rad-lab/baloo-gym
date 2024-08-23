import gymnasium as gym
from baloo_mujoco_sim.utils.baloo_mj_api import get_contact_forces_on_body, get_tactile_image, get_box_position
import numpy as np
import mujoco


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

    def step(self, action):
        """Step function that calls the parent step function and then calculates the reward."""
        # call baloo_base step function
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        reward = self.calculate_reward()

        #todo: need to truncate if the box tips over?
        return observation, reward, terminated, truncated, info

    def calculate_reward(self):
        """
        Calculates the reward to return. Used with Carlo Alessi at SSSA

        Three part reward:

        reward = -a if there are numerical errors
        reward = R_approach + R_sensor + R_grasp + R_body
        """

        #penalize not moving
        #todo: change r_approach to be distance from chest to box. -distance basically?
        chest_pos = self.env.unwrapped.data.geom('chest').xpos
        box_pos = get_box_position(self.env.unwrapped.model,
                                   self.env.unwrapped.data)
        r_approach = -np.linalg.norm(chest_pos - box_pos)
        # r_approach = -1

        # L_T0 = get_tactile_image(self.env.unwrapped.model,
        #                          self.env.unwrapped.data, 'left', 0)
        # L_T1 = get_tactile_image(self.env.unwrapped.model,
        #                          self.env.unwrapped.data, 'left', 1)
        # R_T0 = get_tactile_image(self.env.unwrapped.model,
        #                          self.env.unwrapped.data, 'right', 0)
        # R_T1 = get_tactile_image(self.env.unwrapped.model,
        #                          self.env.unwrapped.data, 'right', 1)
        # C_T = get_tactile_image(self.env.unwrapped.model,
        #                         self.env.unwrapped.data, 'chest', None)

        # #contact forces on body, or reward making box move up
        # #encourage grasping with sensor areas
        # r_sensor = (np.linalg.norm(L_T0, 'fro') + np.linalg.norm(L_T1, 'fro') +
        #             np.linalg.norm(R_T0, 'fro') + np.linalg.norm(R_T1, 'fro') +
        #             np.linalg.norm(C_T, 'fro'))

        # # #penalize large differences between the two arms
        # # r_grasp = -np.linalg.norm(L_T0 - R_T0, 'fro') + np.linalg.norm(
        # #     L_T1 - R_T1, 'fro')

        # #reward for box moving upwards, implying that it was somehow lifted up.
        # box_pos = get_box_position(self.env.unwrapped.model, self.env.unwrapped.data)
        # r_box = box_pos[2] - self.box_pos[2]

        # a = 1
        # b = 1
        # c = .1
        # d = 1

        # # reward = a * r_approach + b * r_sensor + c * r_grasp + d * r_box
        # reward = a * r_approach + b * r_sensor + c * r_box

        reward = r_approach
        return reward
