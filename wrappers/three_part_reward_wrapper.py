import gymnasium as gym
from baloo_mujoco_sim.utils.baloo_mj_api import (get_contact_forces_on_body,
                                                 get_tactile_image,
                                                 get_box_position,
                                                 detect_box_touch,
                                                 get_joint_angles)
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

        self.state = 'approach'

    def step(self, action):
        """Step function that calls the parent step function and then calculates the reward."""
        # call baloo_base step function
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        reward = self.calculate_reward()

        #todo: need to truncate if the box tips over?
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset function that calls the parent reset function and then calculates the reward."""
        # call baloo_base reset function
        self.state = 'approach'
        return self.env.reset()

    def calculate_reward(self):
        """
        Calculates the reward to return. Used with Carlo Alessi at SSSA

        Three part reward:

        reward = -a if there are numerical errors
        reward = R_approach + R_sensor + R_grasp + R_body
        """

        #penalize being far from the box
        chest_pos = self.env.unwrapped.data.geom('chest').xpos
        box_pos = get_box_position(self.env.unwrapped.model,
                                   self.env.unwrapped.data)

        #add joint centering penalty on q^2
        right_j0 = get_joint_angles(self.env.unwrapped.model,
                                    self.env.unwrapped.data, 'right', 0)
        right_j1 = get_joint_angles(self.env.unwrapped.model,
                                    self.env.unwrapped.data, 'right', 1)
        right_j2 = get_joint_angles(self.env.unwrapped.model,
                                    self.env.unwrapped.data, 'right', 2)
        left_j0 = get_joint_angles(self.env.unwrapped.model,
                                   self.env.unwrapped.data, 'left', 0)
        left_j1 = get_joint_angles(self.env.unwrapped.model,
                                   self.env.unwrapped.data, 'left', 1)
        left_j2 = get_joint_angles(self.env.unwrapped.model,
                                   self.env.unwrapped.data, 'left', 2)

        joint_penalty = -np.linalg.norm(right_j0)**2 - np.linalg.norm(
            right_j1)**2 - np.linalg.norm(right_j2)**2 - np.linalg.norm(
                left_j0)**2 - np.linalg.norm(left_j1)**2 - np.linalg.norm(
                    left_j2)**2

        if self.state == 'approach':
            r_approach = -1 * np.linalg.norm(chest_pos - box_pos)

            #penalize touching object until we are ready to grasp
            box_touched = detect_box_touch(self.env.unwrapped.model,
                                           self.env.unwrapped.data)

            if box_touched:
                r_approach += -1

            reward = r_approach

            #if height of chest is within a certain range of manipuland centroid, then we are ready to grasp
            if abs(chest_pos[2] - box_pos[2]) < 0.25:
                self.state = 'grasp'

                #change box color to green
                self.env.unwrapped.model.geom('box').rgba = [0, 1, 0, 1]

        elif self.state == 'grasp':
            #reward = r_sensor + r_grasp
            L_T0 = get_tactile_image(self.env.unwrapped.model,
                                     self.env.unwrapped.data, 'left', 0)
            L_T1 = get_tactile_image(self.env.unwrapped.model,
                                     self.env.unwrapped.data, 'left', 1)
            R_T0 = get_tactile_image(self.env.unwrapped.model,
                                     self.env.unwrapped.data, 'right', 0)
            R_T1 = get_tactile_image(self.env.unwrapped.model,
                                     self.env.unwrapped.data, 'right', 1)
            C_T = get_tactile_image(self.env.unwrapped.model,
                                    self.env.unwrapped.data, 'chest', None)

            #contact forces on body, or reward making box move up
            #encourage grasping with sensor areas
            r_sensor = (np.log(1 + np.linalg.norm(L_T0, 'fro')) +
                        np.log(1 + np.linalg.norm(L_T1, 'fro')) +
                        np.log(1 + np.linalg.norm(R_T0, 'fro')) +
                        np.log(1 + np.linalg.norm(R_T1, 'fro')) +
                        np.log(1 + np.linalg.norm(C_T, 'fro')))

            # #penalize large differences between the two arms.
            # r_grasp = -np.linalg.norm(L_T0 - R_T0, 'fro') + np.linalg.norm(
            #     L_T1 - R_T1, 'fro')

            reward = r_sensor

            # elif self.state == 'lift':
            #reward for box moving upwards, implying that it was somehow lifted up.
            # box_pos = get_box_position(self.env.unwrapped.model,
            #                         self.env.unwrapped.data)
            # r_box = box_pos[2] - self.box_pos[2]

            # reward = 100 * r_box

        return reward + .1 * joint_penalty
