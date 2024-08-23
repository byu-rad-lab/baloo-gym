from gymnasium import spaces
import numpy as np

from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_tactile_image, )

from envs.baloo_base import BalooBase

from utils.observation import Observation
from utils.helpers import get_sensor_data


class IncrementalAction:
    """
    This class is used to store the action vector.
    """
    def __init__(self, normalized_action_vector):
        self.elevator_height = np.asarray(normalized_action_vector[0])
        self.left_j0_pressure = np.asarray(normalized_action_vector[1:5])
        self.left_j1_pressure = np.asarray(normalized_action_vector[5:9])
        self.left_j2_pressure = np.asarray(normalized_action_vector[9:13])
        self.right_j0_pressure = np.asarray(normalized_action_vector[13:17])
        self.right_j1_pressure = np.asarray(normalized_action_vector[17:21])
        self.right_j2_pressure = np.asarray(normalized_action_vector[21:25])

        self.action_lower_bound = np.asarray([-1000] + [0] * 24)
        self.action_upper_bound = np.asarray([0] + [300] * 24)

        # add flag to declare if action is normalized or not.
        self.is_normalized = True

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

    def _saturate(self):
        np.clip(
            self.elevator_height,
            self.action_lower_bound[0],
            self.action_upper_bound[0],
            out=self.elevator_height,
        )
        np.clip(
            self.left_j0_pressure,
            self.action_lower_bound[1:5],
            self.action_upper_bound[1:5],
            out=self.left_j0_pressure,
        )
        np.clip(
            self.left_j1_pressure,
            self.action_lower_bound[5:9],
            self.action_upper_bound[5:9],
            out=self.left_j1_pressure,
        )
        np.clip(
            self.left_j2_pressure,
            self.action_lower_bound[9:13],
            self.action_upper_bound[9:13],
            out=self.left_j2_pressure,
        )
        np.clip(
            self.right_j0_pressure,
            self.action_lower_bound[13:17],
            self.action_upper_bound[13:17],
            out=self.right_j0_pressure,
        )
        np.clip(
            self.right_j1_pressure,
            self.action_lower_bound[17:21],
            self.action_upper_bound[17:21],
            out=self.right_j1_pressure,
        )
        np.clip(
            self.right_j2_pressure,
            self.action_lower_bound[21:25],
            self.action_upper_bound[21:25],
            out=self.right_j2_pressure,
        )

    def increment(self, increment_directions):
        """
        increment_directions is a 25 element vector of +1, 0, or -1.
        Each on is scaled by 20kpa for pressures and .1m for height.
        """
        self.elevator_height += increment_directions[0] * 50
        self.left_j0_pressure += increment_directions[1:5] * 10
        self.left_j1_pressure += increment_directions[5:9] * 10
        self.left_j2_pressure += increment_directions[9:13] * 10
        self.right_j0_pressure += increment_directions[13:17] * 10
        self.right_j1_pressure += increment_directions[17:21] * 10
        self.right_j2_pressure += increment_directions[21:25] * 10

        self._saturate()

        return self


class BalooV1(BalooBase):
    def __init__(
        self,
        render_mode=None,
        camera_name=None,
        ctrl_timestep=0.01,
    ):
        super().__init__(
            render_mode=render_mode,
            camera_name=camera_name,
            ctrl_timestep=ctrl_timestep,
        )

        # action space is elevator height, pressure commands for each joint (h, left [0,1,2,3], right [0,1,2,3])
        self.action_space = spaces.MultiDiscrete([3] * 25)

        self.observation_space = spaces.Box(-1,
                                            1,
                                            shape=(6 + 6 + 6 + 6 + 3 + 3 +
                                                   2, ),
                                            dtype=np.float32)

        self.current_actions = IncrementalAction(np.zeros(25))

    def get_observation_from_mujoco(self):
        rawObs = Observation(**get_sensor_data(self.model, self.data))

        return rawObs.normalize_and_center().astype(
            self.observation_space.dtype)

    def map_action_to_commands(self, action):
        # ! gym 0.29.0 added start parameter for multidiscrete action space [-1,1] and sb3 2.1.0 I don't think follows it. [0,2]
        # ! downgrading to 0.28.1 and 2.0.0 eliminates the start param issue, but now actions can only be 0,1,or2.

        action = np.asarray(action) - 1  # shift to be between -1 and 1.

        # action is +1,0,or 1 for each command.
        self.current_actions = self.current_actions.increment(action)

        return self.current_actions._to_array()

    def calculate_reward(self):
        # calculate reward based on number of active taxels
        taxel_left_l0 = get_tactile_image(self.model, self.data, "left", 0)
        taxel_left_l1 = get_tactile_image(self.model, self.data, "left", 1)
        taxel_right_l0 = get_tactile_image(self.model, self.data, "right", 0)
        taxel_right_l1 = get_tactile_image(self.model, self.data, "right", 1)
        taxel_chest = get_tactile_image(self.model, self.data, "chest", None)

        reward_left_l0 = np.count_nonzero(taxel_left_l0) / 1024
        reward_left_l1 = np.count_nonzero(taxel_left_l1) / 1024
        reward_right_l0 = np.count_nonzero(taxel_right_l0) / 1024
        reward_right_l1 = np.count_nonzero(taxel_right_l1) / 1024
        reward_chest = np.count_nonzero(taxel_chest) / 1024

        total_reward = (reward_left_l0 + reward_left_l1 + reward_right_l0 +
                        reward_right_l1 + reward_chest)

        return (
            total_reward - 1
        )  # penalize if total_reward is 0, hopefully to push arms to move
