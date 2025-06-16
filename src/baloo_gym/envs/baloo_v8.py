from baloo_gym.envs.baloo_base import BalooBase
from gymnasium import spaces
import numpy as np
from baloo_gym.utils.observation_spaces import StateObservationPressurePrevActionsNoError
from baloo_gym.utils.helpers import get_sensor_data
from baloo_mujoco_sim.utils.baloo_mj_api import get_joint_pressures, get_elevator_cmd, get_contact_forces_on_body, get_box_position, get_chest_position
from baloo_gym.utils.action_spaces import IncrementalTorques
import mujoco


class BalooV8(BalooBase):

    def __init__(
        self,
        render_mode=None,
        camera_name=None,
        ctrl_timestep=0.01,
        render_width=320,
        render_height=240,
        desired_box_pos=None,
        randomize_initial_height=False,
        randomize_object_size=False,
        randomize_object_mass=False,
        object_size=None,
        object_mass=None,
    ):
        super().__init__(render_mode=render_mode,
                         camera_name=camera_name,
                         ctrl_timestep=ctrl_timestep,
                         render_width=render_width,
                         render_height=render_height,
                         randomize_initial_height=randomize_initial_height,
                         randomize_object_size=randomize_object_size,
                         randomize_object_mass=randomize_object_mass,
                         object_size=object_size,
                         object_mass=object_mass)

        action_size = IncrementalTorques.shape[0]
        self.action_space = self.action_space = spaces.MultiDiscrete(
            [3] * action_size)

        self.observation_space = spaces.Box(
            -1,
            1,
            shape=StateObservationPressurePrevActionsNoError.shape,
            dtype=np.float32)

        elevator_cmd = get_elevator_cmd(self.model, self.data)
        self.torque_cmds = IncrementalTorques(
            np.hstack([elevator_cmd, np.zeros(12)]))

        if desired_box_pos is None:
            self.desired_box_pos = np.array([0, 0.5, .75])

        else:
            self.desired_box_pos = desired_box_pos

        self.first_render_call = True

    def get_observation_from_mujoco(self):
        #implement something that includes previous actions as well as state observations
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

        sensor_data[
            "prev_height_cmd"] = self.torque_cmds.prev_elevator_height_cmd
        sensor_data["prev_left_j0_tau"] = self.torque_cmds.prev_left_j0_tau_cmd
        sensor_data["prev_left_j1_tau"] = self.torque_cmds.prev_left_j1_tau_cmd
        sensor_data["prev_left_j2_tau"] = self.torque_cmds.prev_left_j2_tau_cmd
        sensor_data[
            "prev_right_j0_tau"] = self.torque_cmds.prev_right_j0_tau_cmd
        sensor_data[
            "prev_right_j1_tau"] = self.torque_cmds.prev_right_j1_tau_cmd
        sensor_data[
            "prev_right_j2_tau"] = self.torque_cmds.prev_right_j2_tau_cmd

        box_xpos = get_box_position(self.unwrapped.model, self.unwrapped.data)
        chest_xpos = get_chest_position(self.unwrapped.model,
                                        self.unwrapped.data)

        sensor_data["chest_proximity"] = chest_xpos - box_xpos

        rawObs = StateObservationPressurePrevActionsNoError(**sensor_data)

        tmp = rawObs.normalize_and_center().astype(
            self.observation_space.dtype)

        return tmp

    def map_action_to_commands(self, action):
        # ! gym 0.29.0 added start parameter for multidiscrete action space [-1,1] and sb3 2.1.0 I don't think follows it. [0,2]
        # ! downgrading to 0.28.1 and 2.0.0 eliminates the start param issue, but now actions can only be 0,1,or2.

        action = np.asarray(action) - 1  # shift to be between -1 and 1.

        self.torque_cmds.update(action)
        #torque commands for this time step

        #make this more concise later with a loop
        commands = np.zeros(self.len_command)
        commands[0] = self.torque_cmds.elevator_height_cmd
        commands[1] = 150 + self.torque_cmds.left_j0_tau_cmd[0] / 2
        commands[2] = 150 - self.torque_cmds.left_j0_tau_cmd[0] / 2

        commands[3] = 150 + self.torque_cmds.left_j0_tau_cmd[1] / 2
        commands[4] = 150 - self.torque_cmds.left_j0_tau_cmd[1] / 2

        commands[5] = 150 + self.torque_cmds.left_j1_tau_cmd[0] / 2
        commands[6] = 150 - self.torque_cmds.left_j1_tau_cmd[0] / 2

        commands[7] = 150 + self.torque_cmds.left_j1_tau_cmd[1] / 2
        commands[8] = 150 - self.torque_cmds.left_j1_tau_cmd[1] / 2

        commands[9] = 150 + self.torque_cmds.left_j2_tau_cmd[0] / 2
        commands[10] = 150 - self.torque_cmds.left_j2_tau_cmd[0] / 2

        commands[11] = 150 + self.torque_cmds.left_j2_tau_cmd[1] / 2
        commands[12] = 150 - self.torque_cmds.left_j2_tau_cmd[1] / 2

        commands[13] = 150 + self.torque_cmds.right_j0_tau_cmd[0] / 2
        commands[14] = 150 - self.torque_cmds.right_j0_tau_cmd[0] / 2

        commands[15] = 150 + self.torque_cmds.right_j0_tau_cmd[1] / 2
        commands[16] = 150 - self.torque_cmds.right_j0_tau_cmd[1] / 2

        commands[17] = 150 + self.torque_cmds.right_j1_tau_cmd[0] / 2
        commands[18] = 150 - self.torque_cmds.right_j1_tau_cmd[0] / 2

        commands[19] = 150 + self.torque_cmds.right_j1_tau_cmd[1] / 2
        commands[20] = 150 - self.torque_cmds.right_j1_tau_cmd[1] / 2

        commands[21] = 150 + self.torque_cmds.right_j2_tau_cmd[0] / 2
        commands[22] = 150 - self.torque_cmds.right_j2_tau_cmd[0] / 2

        commands[23] = 150 + self.torque_cmds.right_j2_tau_cmd[1] / 2
        commands[24] = 150 - self.torque_cmds.right_j2_tau_cmd[1] / 2

        return commands

    def reset(self, seed=None, options=None):
        #this will reload the model from xml and reset to some state,
        # but return incorrect previous commands since torque_cmds hasn't been reset.
        _, info = super().reset(seed=seed, options=options)

        #set to correct commands now that model has been reset to some state
        elevator_cmd = get_elevator_cmd(self.model, self.data)
        self.torque_cmds = IncrementalTorques(
            np.hstack([elevator_cmd, np.zeros(12)]))

        #override observation to give correct previous commands
        obs = self.get_observation_from_mujoco()

        return obs, info

    def calculate_reward(self) -> float:
        return 0
