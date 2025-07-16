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
    get_all_contact_wrenches,
    check_arms_touching_ground,
    check_arm_arm_collision,
    check_arm_base_collision,
)
import mujoco
import numpy as np
import scipy.spatial
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import scipy

from baloo_gym.policies.open_loop_hugger import OpenLoopHuggerPolicy
from baloo_gym.utils import action_spaces


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
        self.prev_box_proximity = np.zeros(3)
        self.baseline_policy = OpenLoopHuggerPolicy(N=50)
        self.previous_obs = None
        self.collision_counter = 0
        self.initial_height = None
        self.imitation_decay_rate = 1 / 10000000
        self.step_count = 0

    def _box_fell_over(self):
        # if the angle between box z axis and the world z axis is more than 80 degrees, box probably fell over.
        box_quat = get_box_quat(self.unwrapped.model, self.unwrapped.data)
        box_quat = R.from_quat(np.roll(box_quat, -1))
        box_z_axis = box_quat.apply([0, 0, 1])
        world_z_axis = np.array([0, 0, 1])
        angle = np.arccos(self._cosine_similarity(box_z_axis, world_z_axis))
        angle = np.degrees(angle)
        if angle > 80:
            return True
        return False

    def step(self, action):
        """Step function that calls the parent step function and then calculates the reward."""
        # call baloo_base step function
        observation, reward, terminated, truncated, info = self.env.step(
            action)

        reward = self.calculate_reward(action, info)

        #terminate if box is off the ground for 5 steps or if it falls over.
        terminated = info.get("is_success", False)
        terminated = info.get("box_fell_over", terminated)

        self.previous_obs = observation.copy()

        self.step_count += 1

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset function that calls the parent reset function and then calculates the reward."""

        obs, info = self.env.reset(seed=seed, options=options)
        # call baloo_base reset function
        self.desired_box_visual_initialized = False
        self.sphere_visual_initialized = False
        self.convex_hull_visual_initialized = False
        self.box_zerror_prev = 0
        self.previous_action = np.zeros(self.unwrapped.action_space.shape)
        self.box_lifted_already = False
        self.object_off_floor_consecutive_steps = 0
        self.previous_obs = obs.copy()
        self.baseline_policy.restart()
        self.collision_counter = 0
        self.initial_height = get_box_position(self.unwrapped.model,
                                               self.unwrapped.data)[2]

        return obs, info

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

    def calculate_reward(self, action, info):
        """
        Calculates the reward to return. Used with Carlo Alessi at SSSA
        """

        box_xpos = get_box_position(self.unwrapped.model, self.unwrapped.data)
        chest_xpos = get_chest_position(self.unwrapped.model,
                                        self.unwrapped.data)
        z_error = (chest_xpos[2] + .13) - box_xpos[2]

        ##### PRIMARY REWARD #####
        reward = 0
        info["reward_terms"] = {}

        if not detect_box_on_ground(self.unwrapped.model, self.unwrapped.data):
            self.box_lifted_already = True
            self.object_off_floor_consecutive_steps += 1
            self.unwrapped.model.geom('box').rgba = [0, 1, 0, 1]
        else:
            self.object_off_floor_consecutive_steps = 0
            #redness as an indicator of mass. dark red = heavy, light red = light
            redness = 1 - self.unwrapped.model.body('box').mass.item() / 20
            self.unwrapped.model.geom('box').rgba = [1, redness, redness, 1]

        info["is_success"] = False

        box_height = box_xpos[2]
        if box_height > self.initial_height + 0.5:
            info["is_success"] = True
            success_reward = 10
            info["reward_terms"]["success"] = success_reward
            reward += success_reward

        if self._box_fell_over():
            info["is_success"] = False
            info["box_fell_over"] = True
            box_fell_over_reward = -2
            info["reward_terms"]["box_fell_over"] = box_fell_over_reward
            reward -= box_fell_over_reward

        if "shaped_reward_comparison" in self.reward_selection:
            # this is the shaped reward that we compare against. Will be combined with the success reward above.
            r_approach = .1 * self._calc_chest_proximity_reward(box_xpos)

            if np.abs(z_error) < .35:
                box_contact_forces = get_contact_forces_on_body(
                    self.unwrapped.model, self.unwrapped.data, 'box')
                num_contacts = box_contact_forces.shape[0]
                r_grasp = 0.1 * num_contacts
            else:
                r_grasp = 0

            change_in_box_height = box_height - self.initial_height
            if change_in_box_height > 0:
                r_height = change_in_box_height * 1
            else:
                r_height = 0

            reward += r_approach + r_grasp + r_height

        if "copy_baseline" in self.reward_selection:
            #copy the reward from the baseline.
            baseline_actions, _ = self.baseline_policy.predict(
                self.previous_obs)

            #difference between baseline actions and the actions this policy chose
            action_diff = np.linalg.norm(action - baseline_actions)
            decaying_imitation_weight = 0.1 * max(
                0, 1 - self.imitation_decay_rate * self.step_count)
            action_prior_reward = decaying_imitation_weight * np.exp(
                -0.5 * action_diff**2)

            info["reward_terms"]["copy_baseline"] = action_prior_reward
            # print("Action reward from baseline policy: ", action_prior_reward)
            reward += action_prior_reward

        if "dont_drop" in self.reward_selection:
            dont_drop_reward = .01 * self.object_off_floor_consecutive_steps
            reward += dont_drop_reward
            info["reward_terms"]["dont_drop"] = dont_drop_reward

            #penalize if the box WAS off the ground, but is now on the ground.
            if self.object_off_floor_consecutive_steps == 0 and self.box_lifted_already:
                dropped_reward = -.001 * self.object_off_floor_consecutive_steps
                info["reward_terms"]["dropped"] = dropped_reward
                reward += dropped_reward
                self.box_lifted_already = False

            #penalize if the box WAS off the ground, but is now on the ground.
            if self.object_off_floor_consecutive_steps == 0 and self.box_lifted_already:
                dropped_reward = 0
                info["reward_terms"]["dropped"] = dropped_reward
                reward += dropped_reward
                self.box_lifted_already = False

        ##### SECONDARY REWARDS #####
        if "chest_proximity" in self.reward_selection:
            chest_proximity_reward = .1 * self._calc_chest_proximity_reward(
                box_xpos)
            reward += chest_proximity_reward
            info["reward_terms"]["chest_proximity"] = chest_proximity_reward

        if "joint_centering" in self.reward_selection:
            centering_weight = 0.005
            reward -= centering_weight * self.get_joint_centering_reward()

        if "minimize_torques" in self.reward_selection:
            #penalize torques to minimize energy use. pressure torques are encoded in the actions[1:]. 1 corresponds to maximum torque.
            joint_torques = action[1:]

            norm = np.linalg.norm(joint_torques)
            min_torque_reward = -0.0005 * norm

            info["reward_terms"]["minimize_torques"] = min_torque_reward
            reward += min_torque_reward

        if "action_smoothness" in self.reward_selection:
            # action_diff = 1 is max change corresponding to -1 to 1 actions.
            action_diff = np.linalg.norm(action - self.previous_action)**2
            reward -= .1 * action_diff

        if "high_contact_forces" in self.reward_selection:
            #if contact forces anywhere are high enough to cause damages, penalize.
            wrenches = get_all_contact_wrenches(self.unwrapped.model,
                                                self.unwrapped.data)
            #wrenches is ncon x 6
            # if there's contact forces, check if they're too high.
            if wrenches.size > 0:
                force_norms = np.linalg.norm(wrenches[:, :3], axis=1)
                # torque_norms = np.linalg.norm(wrenches[:, 3:], axis=1)
                # wrench_norms = np.sqrt(force_norms**2 + torque_norms**2)

                # print(
                #     f"min/max wrench norms: {np.min(force_norms), np.max(force_norms)}"
                # )

                max_object_mass = 20
                max_force_threshold = max_object_mass * 9.81

                if np.max(force_norms) / max_force_threshold > 1:
                    normalized = np.max(force_norms) / max_force_threshold
                    reward -= 0.05 * (normalized - 1)**2
                    #make box yellow to indicate high forces.
                    self.unwrapped.model.geom('box').rgba = [1, 1, 0, 1]

        #if the chest is close enough, then start rewards for touching box
        if np.abs(z_error) < .5:
            if "tactile_nonzero" in self.reward_selection:
                taxel_reward = 20 * self._count_nonzero_percentage()
                reward += taxel_reward
                info["reward_terms"]["tactile_nonzero"] = taxel_reward

            if "upward_force" in self.reward_selection:
                #get the net force on the box in the world frame
                box_contact_forces = get_contact_forces_on_body(
                    self.unwrapped.model, self.unwrapped.data, 'box')

                net_force = np.sum(box_contact_forces, axis=0)
                box_mass = self.unwrapped.model.body('box').mass.item()

                if self.unwrapped.model.opt.gravity[2] != 0:
                    normalized_force = net_force / np.abs(
                        (box_mass * self.unwrapped.model.opt.gravity[2]))
                else:
                    raise RuntimeError(
                        f"Upward force reward requires non-zero gravity. Gravity is set to {self.unwrapped.model.opt.gravity[2]}"
                    )

                if normalized_force[2] > 0:
                    upward_force_reward = .01 * normalized_force[2]
                    reward += upward_force_reward
                    info["reward_terms"]["upward_force"] = upward_force_reward

            if "inward_force" in self.reward_selection:
                #get the net force on the box in the world frame
                box_contact_forces = get_contact_forces_on_body(
                    self.unwrapped.model, self.unwrapped.data, 'box')

                net_force = np.sum(box_contact_forces, axis=0)
                box_mass = self.unwrapped.model.body('box').mass.item()

                if self.unwrapped.model.opt.gravity[2] != 0:
                    normalized_force = net_force / np.abs(
                        (box_mass * self.unwrapped.model.opt.gravity[2]))
                else:
                    raise RuntimeError(
                        f"Upward force reward requires non-zero gravity. Gravity is set to {self.unwrapped.model.opt.gravity[2]}"
                    )

                if normalized_force[1] < 0:
                    inward_force_reward = 1 * normalized_force[1]
                    reward += inward_force_reward
                    info["reward_terms"]["inward_force"] = inward_force_reward

        if "arm_convex_hull" in self.reward_selection:
            reward -= self._get_convex_hull_distance(box_xpos)

        if "rms_robot_dist" in self.reward_selection:
            reward -= self._get_rms_robot_dist(box_xpos, chest_xpos)

        if "touch_ground" in self.reward_selection:
            #penalize any part of arms touching the ground.
            touching_ground = check_arms_touching_ground(
                self.unwrapped.model, self.unwrapped.data)
            touch_base = check_arm_base_collision(self.unwrapped.model,
                                                  self.unwrapped.data)

            touching_arms = check_arm_arm_collision(self.unwrapped.model,
                                                    self.unwrapped.data)

            if touching_ground or touch_base or touching_arms:
                self.collision_counter += 1

                touch_ground_reward = -.0001 * self.collision_counter
                info["reward_terms"]["touch_ground"] = touch_ground_reward
                reward += touch_ground_reward
            else:
                self.collision_counter = 0

        self.previous_action = action.copy()
        return reward  #hard coded for a whole episode length

    def _calc_chest_proximity_reward(self, box_xpos):
        # this assumes that the robot has already "aligned" the object in the x and y directions (if it were mobile).
        chest_xpos = get_chest_position(self.unwrapped.model,
                                        self.unwrapped.data)

        #-.13 to get to geoetric center of tactile sensor on chest.
        x_error = chest_xpos[0] - box_xpos[0]
        y_error = chest_xpos[1] - box_xpos[1]
        z_error = (chest_xpos[2] + .13) - box_xpos[2]

        error = np.array([x_error, y_error, z_error])

        box_xerror = np.linalg.norm(error, ord=2)

        # reward = 1 * z_error**2
        # reward = np.exp(-a*box_xerror**2) - np.exp(-a*np.linalg.norm(self.prev_box_proximity)**2)

        # reward = self._phi(box_xerror) - self._phi(
        #     np.linalg.norm(self.prev_box_proximity))

        reward = self._phi(box_xerror)

        # print(f"chest proximity error: {box_xerror}")

        self.prev_box_proximity = error.copy()
        return reward

    def _phi(self, x):
        a = 4  #tune to some small signal from anywhere in state space.
        return np.exp(-a * x**2)

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

