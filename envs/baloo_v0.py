import numpy as np
from gymnasium import spaces

from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_tactile_image, )
from envs.baloo_base import BalooBase

from utils.observation_spaces import StateObservation
from utils.helpers import get_sensor_data
from utils.action_spaces import NormalizedAction


class BalooV0(BalooBase):
    '''
    BalooV0 implements an environment where 
    '''
    def __init__(
        self,
        render_mode=None,
        camera_name=None,
        ctrl_timestep=0.01,
    ):
        super().__init__(render_mode=render_mode,
                         camera_name=camera_name,
                         ctrl_timestep=ctrl_timestep)

        # action space is elevator height, pressure commands for each joint (h, left [0,1,2,3], right [0,1,2,3])
        self.action_space = spaces.Box(-1,
                                       1,
                                       shape=NormalizedAction.shape,
                                       dtype=np.float32)

        self.observation_space = spaces.Box(-1,
                                            1,
                                            shape=StateObservation.shape,
                                            dtype=np.float32)

    def get_observation_from_mujoco(self):
        rawObs = StateObservation(**get_sensor_data(self.model, self.data))

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
