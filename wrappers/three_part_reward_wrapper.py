import gymnasium as gym
from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_box_position,
    get_box_vel,
    get_box_quat,
    get_link_position,
    get_chest_position,
    set_mocap_pose,
    set_mocap_size,
)
import numpy as np
from scipy.spatial.transform import Rotation as R


class ThreePartRewardWrapper(gym.Wrapper):
    """
    Now this class can just overrides the calculate_reward() method to return a force based reward. 
    """
    def __init__(self, env):
        """Constructor for the Reward wrapper."""
        super().__init__(env)

        self.sphere_radius = 0.5

    def step(self, action):
        """Step function that calls the parent step function and then calculates the reward."""
        # call baloo_base step function
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        reward = self.calculate_reward()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset function that calls the parent reset function and then calculates the reward."""
        # call baloo_base reset function
        return self.env.reset()

    def _cosine_similarity(self, v1, v2):
        if np.linalg.norm(v1) <= 1e-6 or np.linalg.norm(v2) <= 1e-6:
            return 0

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def distance_to_sphere(self, point, center, radius):
        return np.linalg.norm(point - center) - radius

    def calculate_reward(self):
        """
        Calculates the reward to return. Used with Carlo Alessi at SSSA
        """

        reward = 0
        box_xpos = get_box_position(self.env.unwrapped.model,
                                    self.env.unwrapped.data)
        chest_xpos = get_chest_position(self.env.unwrapped.model,
                                        self.env.unwrapped.data)

        if np.linalg.norm(chest_xpos - box_xpos) > .5:
            #### APPROACH PHASE ####
            self.env.unwrapped.model.geom('box').rgba = [1, 0, 0, 1]
            self.env.unwrapped.model.geom("object_force_field").rgba = [
                1, 0, 0, .3
            ]

            left_link0_xpos = get_link_position(self.env.unwrapped.model,
                                                self.env.unwrapped.data,
                                                'left', 0)
            left_link1_xpos = get_link_position(self.env.unwrapped.model,
                                                self.env.unwrapped.data,
                                                'left', 1)
            right_link0_xpos = get_link_position(self.env.unwrapped.model,
                                                 self.env.unwrapped.data,
                                                 'right', 0)
            right_link1_xpos = get_link_position(self.env.unwrapped.model,
                                                 self.env.unwrapped.data,
                                                 'right', 1)

            sphere_center = box_xpos

            #update mujoco model to show sphere. this isn't working rn. the sphere isn't
            set_mocap_size(self.env.unwrapped.model, self.env.unwrapped.data,
                           "object_force_field", [self.sphere_radius])

            set_mocap_pose(self.env.unwrapped.model, self.env.unwrapped.data,
                           "object_force_field", sphere_center)

            chest_dist = self.distance_to_sphere(chest_xpos, sphere_center,
                                                 self.sphere_radius)
            left_link0_dist = self.distance_to_sphere(left_link0_xpos,
                                                      sphere_center,
                                                      self.sphere_radius)
            left_link1_dist = self.distance_to_sphere(left_link1_xpos,
                                                      sphere_center,
                                                      self.sphere_radius)
            right_link0_dist = self.distance_to_sphere(right_link0_xpos,
                                                       sphere_center,
                                                       self.sphere_radius)
            right_link1_dist = self.distance_to_sphere(right_link1_xpos,
                                                       sphere_center,
                                                       self.sphere_radius)

            rms_dist = np.sqrt(
                np.mean(
                    np.array([
                        chest_dist, left_link0_dist, left_link1_dist,
                        right_link0_dist, right_link1_dist
                    ])**2))

            reward -= rms_dist

        else:
            ### Grasp phase ###
            self.env.unwrapped.model.geom("object_force_field").rgba = [
                0, 0, 0, 0
            ]

            #get box velocity
            box_xvel = get_box_vel(self.env.unwrapped.model,
                                   self.env.unwrapped.data)

            desired_box_pos = self.get_wrapper_attr("desired_box_pos")
            box_error_pos = np.linalg.norm(desired_box_pos - box_xpos)

            if box_error_pos < 0.1:
                # or if box is close, stay close to desired position
                self.env.unwrapped.model.geom('box').rgba = [0, 1, 0,
                                                             1]  #green
                reward += 1
            elif box_xvel[2] > 1e-2:
                #reward if box is lifted towards desired position since we are not close to position yet.
                self.env.unwrapped.model.geom('box').rgba = [1, 1, 0,
                                                             1]  #yellow
                reward += 1

        return reward
