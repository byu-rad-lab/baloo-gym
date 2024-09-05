from envs.baloo_base import BalooBase
import numpy as np
from numpy.typing import ArrayLike


class BalooVTemplate(BalooBase):
    '''
    BalooV0 implements an environment where 
    '''

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

        #TODO: define action space
        self.action_space = None

        #TODO: define observation space
        self.observation_space = None

    def get_observation_from_mujoco(self) -> ArrayLike:
        #TODO: implement observation space using mujoco data
        pass

    def map_action_to_commands(self, action: ArrayLike) -> ArrayLike:
        #TODO: map actions to commands that mujoco accepts
        pass

    def calculate_reward(self) -> float:
        #TODO: implement reward function
        pass
