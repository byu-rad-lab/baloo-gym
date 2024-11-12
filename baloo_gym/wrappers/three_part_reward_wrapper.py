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
import mujoco
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
        self.sphere_visual_initialized = False
        self.desired_box_visual_initialized = False

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
        self.desired_box_visual_initialized = False
        self.sphere_visual_initialized = False
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
            # self.env.unwrapped.model.geom("object_force_field").rgba = [
            #     1, 0, 0, .3
            # ]

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
            #get box velocity
            box_xvel = get_box_vel(self.env.unwrapped.model,
                                   self.env.unwrapped.data)

            desired_box_pos = self.get_wrapper_attr("desired_box_pos")
            box_error_pos = np.linalg.norm(desired_box_pos - box_xpos)

            if box_error_pos < 0.05:
                # or if box is close, stay close to desired position
                self.env.unwrapped.model.geom('box').rgba = [0, 1, 0,
                                                             1]  #green
                reward += 2
            elif box_xvel[2] > 1e-2:
                #reward if box is lifted towards desired position since we are not close to position yet.
                self.env.unwrapped.model.geom('box').rgba = [1, 1, 0,
                                                             1]  #yellow
                reward += 1
            else:
                self.env.unwrapped.model.geom('box').rgba = [1, 0, 0, 1]

        return reward

    def render(self):
        super().render()  #to initialize viewer.
        if not self.sphere_visual_initialized:
            self.sphere_visual_initialized = True

            #this will add the marker the next time the scene is rendered.
            # everything in the marker list will be re-rendered each time.
            # so to change postion, just change the position of the marker in the list.
            #this is a bit hacky, since the renderer doesn't really expose this functionality.
            self.sphereid = len(self.unwrapped.mujoco_renderer.viewer._markers)
            self.unwrapped.mujoco_renderer.viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=self.sphere_radius,
                pos=np.array([0, 0, 0]),
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 0.3]),
            )

        if not self.desired_box_visual_initialized:
            self.desired_box_visual_initialized = True

            self.desired_boxid = len(
                self.unwrapped.mujoco_renderer.viewer._markers)

            self.unwrapped.mujoco_renderer.viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([0.1, 0.1, 0.1]),
                pos=self.get_wrapper_attr("desired_box_pos"),
                mat=np.eye(3).flatten(),
                rgba=np.array([0, 1, 0, 0.3]),
            )

        box_pos = self.unwrapped.data.body('box').xipos
        self.unwrapped.mujoco_renderer.viewer._markers[
            self.sphereid]['pos'] = box_pos
        self.unwrapped.mujoco_renderer.viewer._markers[
            self.sphereid]['rgba'] = np.array([1, 0, 0, .1])

        return super().render()
