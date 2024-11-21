from baloo_gym.envs.baloo_base import BalooBase
import numpy as np

from numpy.typing import ArrayLike

from gymnasium.spaces import Box
from baloo_gym.utils.observation_spaces import StateObservationPressure
from baloo_gym.utils.action_spaces import NormalizedAction
from baloo_mujoco_sim.utils.baloo_mj_api import get_joint_pressures
from baloo_gym.utils.helpers import get_sensor_data


class BalooV5(BalooBase):
    
    def __init__(
        self,
        render_mode=None,
        camera_name=None,
        ctrl_timestep=0.01,
        render_width=320,
        render_height=240,
        desired_box_pos=None,
        randomize_initial_height=False
    ):
        super().__init__(
            render_mode=render_mode,
            camera_name=camera_name,
            ctrl_timestep=ctrl_timestep,
            render_width=render_width,
            render_height=render_height,
            randomize_initial_height=randomize_initial_height
        )

        self.action_space = Box(low=-1,
                                high=1,
                                shape=NormalizedAction.shape,
                                dtype=np.float32)

        self.observation_space = Box(low=-1,
                                     high=1,
                                     shape=StateObservationPressure.shape,
                                     dtype=np.float32)
        if desired_box_pos is None:
            self.desired_box_pos = np.array([0, 0.5, .75])
            print("No desired box position given, defaulting to ",
                  self.desired_box_pos)
        else:
            self.desired_box_pos = desired_box_pos

    def get_observation_from_mujoco(self):
        sensor_data = get_sensor_data(self.model, self.data)
        sensor_data["left_j0_pressures"] = get_joint_pressures(
            self.model, self.data, 'left', 0)
        sensor_data["left_j1_pressures"] = get_joint_pressures(
            self.model, self.data, 'left', 1)
        sensor_data["left_j2_pressures"] = get_joint_pressures(
            self.model, self.data, 'left', 2)

        sensor_data["right_j0_pressures"] = get_joint_pressures(
            self.model, self.data, 'right', 0)
        sensor_data["right_j1_pressures"] = get_joint_pressures(
            self.model, self.data, 'right', 1)
        sensor_data["right_j2_pressures"] = get_joint_pressures(
            self.model, self.data, 'right', 2)

        sensor_data["object_pos_error"] = self.desired_box_pos - sensor_data[
            "object_pos"]
        sensor_data.pop("object_pos")

        rawObs = StateObservationPressure(**sensor_data)

        return rawObs.normalize_and_center().astype(
            self.observation_space.dtype)

    def map_action_to_commands(self, action: ArrayLike) -> ArrayLike:
        #actions are now continuous [-1,1] * 25, so they need to scale up to the actual bounds.
        commands = NormalizedAction(action).unnormalize()
        # print(f"Policy Actions:\n", action)
        # print(commands)

        return commands._to_array()

    def calculate_reward(self) -> float:
        return 0
