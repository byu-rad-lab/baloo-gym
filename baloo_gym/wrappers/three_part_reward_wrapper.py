import gymnasium as gym
from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_box_position,
    get_box_vel,
    get_box_quat,
    get_link_position,
    get_chest_position,
    set_mocap_pose,
    set_mocap_size,
    get_tactile_image,
    get_contact_forces_on_body,
    detect_box_touch,
)
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull


class ThreePartRewardWrapper(gym.Wrapper):
    """
    Now this class can just overrides the calculate_reward() method to return a force based reward. 
    """

    def __init__(self, env, reward_selection=None):
        """Constructor for the Reward wrapper."""
        super().__init__(env)

        self.sphere_radius = 0.5
        self.sphere_visual_initialized = False
        self.desired_box_visual_initialized = False
        self.previous_action = np.zeros(self.unwrapped.action_space.shape)
        self.reward_selection = reward_selection

    def step(self, action):
        """Step function that calls the parent step function and then calculates the reward."""
        # call baloo_base step function
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        reward = self.calculate_reward(action)
        self.previous_action = action

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset function that calls the parent reset function and then calculates the reward."""
        # call baloo_base reset function
        self.desired_box_visual_initialized = False
        self.sphere_visual_initialized = False
        self.box_error_prev = 0
        self.previous_action = np.zeros(self.unwrapped.action_space.shape)
        return self.env.reset()

    def _decay_sphere_radius(self):
        start = 0.5
        stop = 0

        self.sphere_radius = start - (
            start - stop
        ) * self.unwrapped.data.time / self.get_wrapper_attr("time_limit_sec")

    def _cosine_similarity(self, v1, v2):
        if np.linalg.norm(v1) <= 1e-6 or np.linalg.norm(v2) <= 1e-6:
            return 0

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _distance_to_sphere(self, point, center, radius):
        return np.linalg.norm(point - center) - radius

    def calculate_reward(self, action):
        """
        Calculates the reward to return. Used with Carlo Alessi at SSSA
        """

        box_xpos = get_box_position(self.unwrapped.model, self.unwrapped.data)
        chest_xpos = get_chest_position(self.unwrapped.model,
                                        self.unwrapped.data)

        desired_box_pos = self.get_wrapper_attr("desired_box_pos")
        box_xerror = desired_box_pos - box_xpos
        box_error = np.linalg.norm(box_xerror)

        numerical_threshold = 1e-3

        reward = 0
        #red
        self.unwrapped.model.geom('box').rgba = [1, 0, 0, .5]

        if np.abs(box_xerror[2]) < 0.05:
            #green
            self.unwrapped.model.geom('box').rgba = [0, 1, 0, .5]
            reward += 1
            return reward

        #reward moving towards desired position, penalize moving away
        if box_error < self.box_error_prev - numerical_threshold:
            #yellow
            self.unwrapped.model.geom('box').rgba = [1, 1, 0, .5]
            reward += .5
        elif box_error > self.box_error_prev + numerical_threshold:
            reward -= .1

        #penalize large changes in action
        if "action_smoothness" in self.reward_selection:
            smoothness_weight = 0.1
            action_diff = np.linalg.norm(action - self.previous_action)
            reward -= smoothness_weight * action_diff

        if "tactile_nonzero" in self.reward_selection:
            taxel_reward = self._count_nonzero_taxels()
            reward += taxel_reward

        if "robot_convex_hull" in self.reward_selection:
            reward -= self._get_convex_hull_distance(box_xpos)

        if "rms_robot_dist" in self.reward_selection:
            reward -= self._get_rms_robot_dist(box_xpos, chest_xpos)

        self.box_error_prev = box_error
        return reward

    def _count_nonzero_taxels(self):
        left_link0_taxels = get_tactile_image(self.unwrapped.model,
                                              self.unwrapped.data, "left", 0)
        left_link1_taxels = get_tactile_image(self.unwrapped.model,
                                              self.unwrapped.data, "left", 1)
        right_link0_taxels = get_tactile_image(self.unwrapped.model,
                                               self.unwrapped.data, "right", 0)
        right_link1_taxels = get_tactile_image(self.unwrapped.model,
                                               self.unwrapped.data, "right", 1)
        chest_taxels = get_tactile_image(self.unwrapped.model,
                                         self.unwrapped.data, "chest", -1)

        left_link0_percent = np.count_nonzero(
            left_link0_taxels) / left_link0_taxels.size
        left_link1_percent = np.count_nonzero(
            left_link1_taxels) / left_link1_taxels.size
        right_link0_percent = np.count_nonzero(
            right_link0_taxels) / right_link0_taxels.size
        right_link1_percent = np.count_nonzero(
            right_link1_taxels) / right_link1_taxels.size
        chest_percent = np.count_nonzero(chest_taxels) / chest_taxels.size

        total = (left_link0_percent + left_link1_percent +
                 right_link0_percent + right_link1_percent + chest_percent) / 5

        return total

    def _get_rms_robot_dist(self, box_xpos, chest_xpos):
        left_link0_xpos = get_link_position(self.unwrapped.model,
                                            self.unwrapped.data, 'left', 0)
        left_link1_xpos = get_link_position(self.unwrapped.model,
                                            self.unwrapped.data, 'left', 1)
        right_link0_xpos = get_link_position(self.unwrapped.model,
                                             self.unwrapped.data, 'right', 0)
        right_link1_xpos = get_link_position(self.unwrapped.model,
                                             self.unwrapped.data, 'right', 1)

        sphere_center = box_xpos

        chest_dist = self._distance_to_sphere(chest_xpos, sphere_center,
                                              self.sphere_radius)
        left_link0_dist = self._distance_to_sphere(left_link0_xpos,
                                                   sphere_center,
                                                   self.sphere_radius)
        left_link1_dist = self._distance_to_sphere(left_link1_xpos,
                                                   sphere_center,
                                                   self.sphere_radius)
        right_link0_dist = self._distance_to_sphere(right_link0_xpos,
                                                    sphere_center,
                                                    self.sphere_radius)
        right_link1_dist = self._distance_to_sphere(right_link1_xpos,
                                                    sphere_center,
                                                    self.sphere_radius)

        rms_dist = np.sqrt(
            np.mean(
                np.array([
                    chest_dist, left_link0_dist, left_link1_dist,
                    right_link0_dist, right_link1_dist
                ])**2))

        return rms_dist

    def _get_robot_convex_hull(self, chest_xpos):
        left_link0_xpos = get_link_position(self.unwrapped.model,
                                            self.unwrapped.data, 'left', 0)
        left_link1_xpos = get_link_position(self.unwrapped.model,
                                            self.unwrapped.data, 'left', 1)
        right_link0_xpos = get_link_position(self.unwrapped.model,
                                             self.unwrapped.data, 'right', 0)
        right_link1_xpos = get_link_position(self.unwrapped.model,
                                             self.unwrapped.data, 'right', 1)

        pts = np.vstack([
            chest_xpos, left_link0_xpos, left_link1_xpos, right_link0_xpos,
            right_link1_xpos
        ])

        return ConvexHull(pts)

    def _get_convex_hull_distance(self, box_xpos):
        hull = self._get_robot_convex_hull(box_xpos)
        hull_centroid = np.mean(hull.points[hull.vertices], axis=0)
        return np.linalg.norm(hull_centroid - box_xpos)

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
                rgba=np.array([1, 0, 0, .1]),
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
                rgba=np.array([0, 1, 0, 1]),
            )

        box_pos = self.unwrapped.data.body('box').xipos
        self.unwrapped.mujoco_renderer.viewer._markers[
            self.sphereid]['pos'] = box_pos
        self.unwrapped.mujoco_renderer.viewer._markers[
            self.sphereid]['rgba'] = np.array([1, 0, 0, .1])

        return super().render()
