import gymnasium as gym
from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_contact_forces_on_body, get_tactile_image, get_box_position,
    get_box_vel, detect_box_touch, get_joint_angles, get_box_quat)
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
        self.desired_box_pos[2] += 0.3

        self.box_quat = R.from_quat(
            get_box_quat(self.env.unwrapped.model,
                         self.env.unwrapped.data,
                         scalar_first=False))

        self.state = 'approach'

        self.prev_intrinsic_reward = 0

    def step(self, action):
        """Step function that calls the parent step function and then calculates the reward."""
        # call baloo_base step function
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        reward = self.calculate_reward()

        # #truncate if box is lifted, box is far away, or if box tipped over
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
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def calculate_reward(self):
        """
        Calculates the reward to return. Used with Carlo Alessi at SSSA

        Three part reward:

        reward = -a if there are numerical errors
        reward = R_approach + R_sensor + R_grasp + R_body
        """

        reward = 0

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

        #reward being close to straight to avoid self collisions.
        joint_centering = (np.exp(-np.linalg.norm(right_j0)**2) +
                           np.exp(-np.linalg.norm(right_j1)**2) +
                           np.exp(-np.linalg.norm(right_j2)**2) +
                           np.exp(-np.linalg.norm(left_j0)**2) +
                           np.exp(-np.linalg.norm(left_j1)**2) +
                           np.exp(-np.linalg.norm(left_j2)**2))

        reward += joint_centering / 6

        #penalize chest being far from the box, exp means reward ranges from 0 to 1, increasing as norm gets lower.
        chest_pos = self.env.unwrapped.data.geom('chest').xpos
        box_pos = get_box_position(self.env.unwrapped.model,
                                   self.env.unwrapped.data)

        #if within a certain distance of the box, then keep previous reward
        if np.linalg.norm(chest_pos - box_pos) < .25:
            intrinsic_reward = self.prev_intrinsic_reward
        else:
            intrinsic_reward = np.exp(-np.linalg.norm(chest_pos - box_pos))
            self.prev_intrinsic_reward = intrinsic_reward

        # #cosine similarity between error and velocity
        # box_vel = get_box_vel(self.env.unwrapped.model, self.env.unwrapped.data)
        # reward += self._cosine_similarity(error, box_vel)

        # reward contact forces on box in appropriate direction, not including ground
        error = self.desired_box_pos - box_pos
        contact_forces = get_contact_forces_on_body(self.env.unwrapped.model,
                                                    self.env.unwrapped.data,
                                                    'box')

        # #reward contacts of any kind to encourage exploration
        # reward += np.log(1 + contact_forces.shape[0])

        #but really emphasize the contact forces in a useful direction.
        net_contact_force = np.sum(contact_forces, axis=0)
        reward += self._cosine_similarity(error, net_contact_force)

        # #penalize touching object until we are ready to grasp
        # if self.state == 'approach':
        #     #penalize touching object until we are ready to grasp
        #     box_touched = detect_box_touch(self.env.unwrapped.model,
        #                                    self.env.unwrapped.data)
        #     if box_touched:
        #         reward += -1

        #     #if height of chest is within a certain range of manipuland centroid, then we are ready to grasp
        #     if abs(chest_pos[2] - box_pos[2]) < 0.25:
        #         self.state = 'grasp'

        #         #change box color to green
        #         self.env.unwrapped.model.geom('box').rgba = [0, 1, 0, 1]

        # # elif self.state == 'grasp':
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

        #encourage grasping with sensor areas with average frobenius norm
        r_sensor = (np.log(1 + np.linalg.norm(L_T0, 'fro')) +
                    np.log(1 + np.linalg.norm(L_T1, 'fro')) +
                    np.log(1 + np.linalg.norm(R_T0, 'fro')) +
                    np.log(1 + np.linalg.norm(R_T1, 'fro')) +
                    np.log(1 + np.linalg.norm(C_T, 'fro')))
        reward += r_sensor / 5

        # #penalize large differences between the two arms.
        # r_grasp = -np.linalg.norm(L_T0 - R_T0, 'fro') + np.linalg.norm(
        #     L_T1 - R_T1, 'fro')

        # #reward for box moving upwards, implying that it was somehow lifted up. This also penalizes box falling down.
        # box_pos = get_box_position(self.env.unwrapped.model,
        #                            self.env.unwrapped.data)
        # r_box = box_pos[2] - self.box_pos[2]
        # reward += 10 * r_box

        # #penalize equally for large changes in the box orientation
        # box_quat = R.from_quat(
        #     get_box_quat(self.env.unwrapped.model,
        #                  self.env.unwrapped.data,
        #                  scalar_first=False))

        # diff = box_quat.inv() * self.box_quat
        # reward += -1 * np.linalg.norm(diff.magnitude())

        return reward + intrinsic_reward
