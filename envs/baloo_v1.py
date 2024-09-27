from gymnasium import spaces
import numpy as np

from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_tactile_image, )

from envs.baloo_base import BalooBase

from utils.observation_spaces import Observation
from utils.helpers import get_sensor_data
from utils.action_spaces import IncrementalAction


class BalooV1(BalooBase):
    def __init__(
        self,
        render_mode=None,
        camera_name=None,
        ctrl_timestep=0.01,
        render_width=320,
        render_height=240,
    ):
        super().__init__(
            render_mode=render_mode,
            camera_name=camera_name,
            ctrl_timestep=ctrl_timestep,
            render_width=render_width,
            render_height=render_height,
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
