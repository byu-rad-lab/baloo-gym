import numpy as np
from gymnasium import spaces

from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_box_position,
    get_box_vel,
    get_elevator_height,
    get_elevator_vel,
    get_joint_angles,
    get_joint_vel,
    get_tactile_image,
)
from envs.baloo_base import BalooBase

from utils.observation import Observation


class NormalizedAction:
    def __init__(self, normalized_action_vector):
        self.elevator_height = normalized_action_vector[0]
        self.left_j0_pressure = normalized_action_vector[1:5]
        self.left_j1_pressure = normalized_action_vector[5:9]
        self.left_j2_pressure = normalized_action_vector[9:13]
        self.right_j0_pressure = normalized_action_vector[13:17]
        self.right_j1_pressure = normalized_action_vector[17:21]
        self.right_j2_pressure = normalized_action_vector[21:25]

        self.action_lower_bound = np.asarray([-1000] + [0] * 24)
        self.action_upper_bound = np.asarray([0] + [300] * 24)

    def __repr__(self):
        return f"Action: {self._to_array()}"

    def _to_array(self):
        return np.hstack([
            self.elevator_height,
            self.left_j0_pressure,
            self.left_j1_pressure,
            self.left_j2_pressure,
            self.right_j0_pressure,
            self.right_j1_pressure,
            self.right_j2_pressure,
        ])

    def unnormalize(self):
        unnormalized_actions = (self._to_array() + 1) * (
            self.action_upper_bound -
            self.action_lower_bound) / 2 + self.action_lower_bound

        self.elevator_height = unnormalized_actions[0]
        self.left_j0_pressure = unnormalized_actions[1:5]
        self.left_j1_pressure = unnormalized_actions[5:9]
        self.left_j2_pressure = unnormalized_actions[9:13]
        self.right_j0_pressure = unnormalized_actions[13:17]
        self.right_j1_pressure = unnormalized_actions[17:21]
        self.right_j2_pressure = unnormalized_actions[21:25]

        return self


class BalooV0(BalooBase):
    '''
    BalooV0 implements an environment where 
    '''
    def __init__(
        self,
        render_mode=None,
        camera_id=None,
        camera_name=None,
        ctrl_timestep=0.01,
    ):
        super().__init__(render_mode=render_mode,
                         camera_id=camera_id,
                         camera_name=camera_name,
                         ctrl_timestep=ctrl_timestep)

        # action space is elevator height, pressure commands for each joint (h, left [0,1,2,3], right [0,1,2,3])
        self.action_space = spaces.Box(-1,
                                       1,
                                       shape=(1 + 24, ),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(-1,
                                            1,
                                            shape=(6 + 6 + 6 + 6 + 3 + 3 +
                                                   2, ),
                                            dtype=np.float32)

    def get_observation_from_mujoco(self):
        rawObs = Observation(**self._get_sensor_data(self.model, self.data))

        return rawObs.normalize_and_center().astype(
            self.observation_space.dtype)

    def map_action_to_commands(self, action):
        unnormalized_actions = NormalizedAction(action).unnormalize()

        return unnormalized_actions._to_array()

    def calculate_reward(self):
        # calculate reward based on number of active taxels
        taxel_left_l0 = get_tactile_image(self.model, self.data, "left", 0)
        taxel_left_l1 = get_tactile_image(self.model, self.data, "left", 1)
        taxel_right_l0 = get_tactile_image(self.model, self.data, "right", 0)
        taxel_right_l1 = get_tactile_image(self.model, self.data, "right", 1)
        taxel_chest = get_tactile_image(self.model, self.data, "chest", None)

        reward_left_l0 = np.count_nonzero(taxel_left_l0)
        reward_left_l1 = np.count_nonzero(taxel_left_l1)
        reward_right_l0 = np.count_nonzero(taxel_right_l0)
        reward_right_l1 = np.count_nonzero(taxel_right_l1)
        reward_chest = np.count_nonzero(taxel_chest)

        total_reward = (reward_left_l0 + reward_left_l1 + reward_right_l0 +
                        reward_right_l1 + reward_chest)

        return (
            total_reward - 1
        )  # penalize if total_reward is 0, hopefully to push arms to move

    def _get_sensor_data(self, model, data):
        left_pos = []
        left_vel = []
        right_pos = []
        right_vel = []
        for i in range(3):
            left_pos.append(get_joint_angles(model, data, "left", i))
            left_vel.append(get_joint_vel(model, data, "left", i))
            right_pos.append(get_joint_angles(model, data, "right", i))
            right_vel.append(get_joint_vel(model, data, "right", i))

        object_pos = get_box_position(model, data)
        object_vel = get_box_vel(model, data)
        elevator_pos = get_elevator_height(model, data)
        elevator_vel = get_elevator_vel(model, data)

        return {
            "object_pos": object_pos,
            "object_vel": object_vel,
            "elevator_pos": elevator_pos,
            "elevator_vel": elevator_vel,
            "left_pos": np.hstack(left_pos),
            "right_pos": np.hstack(right_pos),
            "left_vel": np.hstack(left_vel),
            "right_vel": np.hstack(right_vel),
        }
