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
    get_joint_angles,
    detect_box_on_ground,
)
import mujoco
import numpy as np
import scipy.spatial
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import scipy


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
        self.convex_hull_visual_initialized = False
        self.previous_action = np.zeros(self.unwrapped.action_space.shape)
        self.reward_selection = reward_selection
        self.max_zerror = 1
        self.box_lifted_already = False
        self.initial_box_zerror = 0
        self.object_off_floor_consecutive_steps = 0

    def step(self, action):
        """Step function that calls the parent step function and then calculates the reward."""
        # call baloo_base step function
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        reward = self.calculate_reward(action)

        info["is_success"] = False
        if self.object_off_floor_consecutive_steps >= 5 / self.unwrapped.control_timestep:
            terminated = True
            info["is_success"] = True
            # reward += 50
        self.previous_action = action

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset function that calls the parent reset function and then calculates the reward."""
        # call baloo_base reset function
        self.desired_box_visual_initialized = False
        self.sphere_visual_initialized = False
        self.convex_hull_visual_initialized = False
        self.box_zerror_prev = 0
        self.previous_action = np.zeros(self.unwrapped.action_space.shape)
        self.box_lifted_already = False
        self.object_off_floor_consecutive_steps = 0

        box_xpos = get_box_position(self.unwrapped.model, self.unwrapped.data)
        chest_xpos = get_chest_position(self.unwrapped.model,
                                        self.unwrapped.data)

        desired_box_pos = self.get_wrapper_attr("desired_box_pos")
        box_xerror = desired_box_pos - box_xpos
        self.initial_box_zerror = np.abs(box_xerror[2])
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
        box_zerror = np.abs(box_xerror[2])

        numerical_threshold = 1e-3

        ##### PRIMARY REWARD #####
        reward = 0

        # scaling_factor = 1 / self.max_zerror**2
        # reward -= scaling_factor * box_zerror**2
        # self.unwrapped.model.geom('box').rgba = [
        #     box_zerror / self.initial_box_zerror,
        #     1 - box_zerror / self.initial_box_zerror, 0, .5
        # ]

        # #goal bonuses
        # threshold = 0.05
        # if box_zerror < threshold:
        #     #green
        #     reward += 1
        #     self.unwrapped.model.geom('box').rgba = [1, 1, 1, .5]

        if "dont_drop" in self.reward_selection:
            if not detect_box_on_ground(self.unwrapped.model,
                                        self.unwrapped.data):
                self.box_lifted_already = True
                self.object_off_floor_consecutive_steps += 1
                reward += .1 * self.object_off_floor_consecutive_steps
                self.unwrapped.model.geom('box').rgba = [0, 1, 0, 1]
            else:
                if "chest_proximity" in self.reward_selection:
                    reward -= self._calc_chest_proximity_reward(box_xpos)

                self.object_off_floor_consecutive_steps = 0

                # if self.unwrapped.object is not None:
                #     self.unwrapped.model.geom(
                #         'box').rgba = self.unwrapped.object.color
                # else:

                #redness as an indicator of mass. dark red = heavy, light red = light
                redness = 1 - self.unwrapped.model.body('box').mass.item() / 20
                self.unwrapped.model.geom('box').rgba = [1, redness, redness, 1]

        ##### SECONDARY REWARDS #####
        #this just feels really unnatural, but its effective at avoiding finger crushing.
        # but Im not sure occasionally getting into bad states is bad. Its a natural part of learning.
        if "joint_centering" in self.reward_selection:
            centering_weight = 0.1
            reward -= centering_weight * self.get_joint_centering_reward()

        if "action_smoothness" in self.reward_selection:
            # action_diff = 1 is max change corresponding to -1 to 1 actions.
            normalizer = np.linalg.norm([2] * len(action))
            action_diff = np.linalg.norm(action - self.previous_action)
            reward -= action_diff / normalizer

        if "high_contact_forces" in self.reward_selection:
            #if contact forces anywhere are high enough to cause damages, penalize.
            pass

        if "tactile_nonzero" in self.reward_selection:
            taxel_reward = self._count_nonzero_percentage()
            reward += taxel_reward

        if "arm_convex_hull" in self.reward_selection:
            reward -= self._get_convex_hull_distance(box_xpos)

        if "rms_robot_dist" in self.reward_selection:
            reward -= self._get_rms_robot_dist(box_xpos, chest_xpos)

        # #reward moving towards desired position, penalize moving away
        # if box_zerror < self.box_zerror_prev - numerical_threshold:
        #     #yellow
        #     self.unwrapped.model.geom('box').rgba = [1, 1, 0, .5]
        #     reward += .5
        # elif box_zerror > self.box_zerror_prev + numerical_threshold:
        #     reward -= .1

        self.box_zerror_prev = box_zerror
        return reward

    def _calc_chest_proximity_reward(self, box_xpos):
        '''
        Thought here is that when reaching for something, my arm is in charge of moving my hand to the object
        My fingers are in charge of getting into a good pose to grasp the object.

        Up till now, I've been using all three, so the finger try to approach the object which gets 
        the arms into bad configurations. 

        The absolute distance has a time pressure component.
        Since agent wants error to go away, it will try to get object as close to chest as fast as possible

        So then, if I reward changes in position, then nothing is the matter
        '''

        # this assumes that the robot has already "aligned" the object in the x and y directions (if it were mobile).
        chest_xpos = get_chest_position(self.unwrapped.model,
                                        self.unwrapped.data)

        #-.13 to get to geoetric center of tactile sensor on chest.
        z_error = np.abs((chest_xpos[2] - .13) - box_xpos[2])
        reward = 1 * z_error**2
        return reward

    def _count_nonzero_percentage(self):
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

    def _get_arm_convex_hull(self):
        chest_xpos = get_chest_position(self.unwrapped.model,
                                        self.unwrapped.data)
        left_link0_xpos = get_link_position(self.unwrapped.model,
                                            self.unwrapped.data, 'left', 0)
        left_link1_xpos = get_link_position(self.unwrapped.model,
                                            self.unwrapped.data, 'left', 1)
        right_link0_xpos = get_link_position(self.unwrapped.model,
                                             self.unwrapped.data, 'right', 0)
        right_link1_xpos = get_link_position(self.unwrapped.model,
                                             self.unwrapped.data, 'right', 1)

        #arms are in charge of centering side to side.
        #elevator is in charge of centering vertically.
        pts = np.vstack([
            left_link0_xpos, left_link1_xpos, right_link0_xpos,
            right_link1_xpos
        ])

        return ConvexHull(pts)

    def _get_convex_hull_distance(self, box_xpos):
        try:
            hull = self._get_arm_convex_hull()
            self.hull_centroid = np.mean(hull.points[hull.vertices], axis=0)
            self.sphericity = self.calc_sphericity(hull)

        except scipy.spatial._qhull.QhullError:
            # convex hull fails due to planar points, fall back to point cloud centroid
            left_link0_xpos = get_link_position(self.unwrapped.model,
                                                self.unwrapped.data, 'left', 0)
            left_link1_xpos = get_link_position(self.unwrapped.model,
                                                self.unwrapped.data, 'left', 1)
            right_link0_xpos = get_link_position(self.unwrapped.model,
                                                 self.unwrapped.data, 'right',
                                                 0)
            right_link1_xpos = get_link_position(self.unwrapped.model,
                                                 self.unwrapped.data, 'right',
                                                 1)
            pts = np.vstack([
                left_link0_xpos, left_link1_xpos, right_link0_xpos,
                right_link1_xpos
            ])

            self.hull_centroid = np.mean(pts, axis=0)

        #don't really care about the y or z error, we assume object is in front of us and elevator
        # will try to center it vertically.
        return np.sqrt((self.hull_centroid[0] - box_xpos[0])**2)

    def calc_sphericity(self, hull):
        # https://en.wikipedia.org/wiki/Sphericity. 1 = perfect sphere, 0 = planar
        # think about 3d implications for this, not sure I want this or volume/surface ratio instead.
        phi = np.pi**(1 / 3) * (6 * hull.volume)**(2 / 3) / hull.area
        return phi

    def get_joint_centering_reward(self):
        left_j0_q = get_joint_angles(self.unwrapped.model, self.unwrapped.data,
                                     'left', 0)
        right_j0_q = get_joint_angles(self.unwrapped.model,
                                      self.unwrapped.data, 'right', 0)
        left_j1_q = get_joint_angles(self.unwrapped.model, self.unwrapped.data,
                                     'left', 1)
        right_j1_q = get_joint_angles(self.unwrapped.model,
                                      self.unwrapped.data, 'right', 1)
        left_j2_q = get_joint_angles(self.unwrapped.model, self.unwrapped.data,
                                     'left', 2)
        right_j2_q = get_joint_angles(self.unwrapped.model,
                                      self.unwrapped.data, 'right', 2)

        joint_angles = np.hstack([
            left_j0_q, right_j0_q, left_j1_q, right_j1_q, left_j2_q, right_j2_q
        ])

        q_max = np.array([np.pi] * 12)
        normalizer = np.linalg.norm(q_max)
        centering = np.linalg.norm(joint_angles) / normalizer

        return centering

    def render(self):
        if not self.sphere_visual_initialized and 'rms_robot_dist' in self.reward_selection:
            super().render()  #to initialize viewer.
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

        if not self.convex_hull_visual_initialized and "arm_convex_hull" in self.reward_selection:
            super().render()
            self.convex_hull_visual_initialized = True

            self.convex_hullid = len(
                self.unwrapped.mujoco_renderer.viewer._markers)

            self.unwrapped.mujoco_renderer.viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=0.1,
                pos=self.hull_centroid,
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 1, 1, .25]),
            )

        if "rms_robot_dist" in self.reward_selection:
            box_pos = self.unwrapped.data.body('box').xipos
            self.unwrapped.mujoco_renderer.viewer._markers[
                self.sphereid]['pos'] = box_pos
            self.unwrapped.mujoco_renderer.viewer._markers[
                self.sphereid]['rgba'] = np.array([1, 0, 0, .1])

        if "arm_convex_hull" in self.reward_selection:
            self.unwrapped.mujoco_renderer.viewer._markers[
                self.convex_hullid]['pos'] = self.hull_centroid
            # self.unwrapped.mujoco_renderer.viewer._markers[
            #     self.convex_hullid]['size'] = self.sphericity / 2

        return super().render()
