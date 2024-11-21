import numpy as np


class NormalizedAction:
    shape = (25, )

    def __init__(self, current_command):
        self.elevator_height_cmd = current_command[0]
        self.left_j0_pressure = current_command[1:5]
        self.left_j1_pressure = current_command[5:9]
        self.left_j2_pressure = current_command[9:13]
        self.right_j0_pressure = current_command[13:17]
        self.right_j1_pressure = current_command[17:21]
        self.right_j2_pressure = current_command[21:25]

        self.cmd_lower_bound = np.asarray([-1000] + [0] * 24)
        self.cmd_upper_bound = np.asarray([0] + [300] * 24)

    def __repr__(self):
        return f"Action: {self._to_array()}"

    def _to_array(self):
        return np.hstack([
            self.elevator_height_cmd,
            self.left_j0_pressure,
            self.left_j1_pressure,
            self.left_j2_pressure,
            self.right_j0_pressure,
            self.right_j1_pressure,
            self.right_j2_pressure,
        ])

    def unnormalize(self):
        unnormalized_actions = (self._to_array() + 1) * (
            self.cmd_upper_bound -
            self.cmd_lower_bound) / 2 + self.cmd_lower_bound

        self.elevator_height_cmd = unnormalized_actions[0]
        self.left_j0_pressure = unnormalized_actions[1:5]
        self.left_j1_pressure = unnormalized_actions[5:9]
        self.left_j2_pressure = unnormalized_actions[9:13]
        self.right_j0_pressure = unnormalized_actions[13:17]
        self.right_j1_pressure = unnormalized_actions[17:21]
        self.right_j2_pressure = unnormalized_actions[21:25]

        return self


class NormalizedDifferentialPressure:
    shape = (13, )

    def __init__(self, current_command):
        self.elevator_height_cmd = current_command[0]
        self.left_j0_delta_pressure = current_command[1:3]
        self.left_j1_delta_pressure = current_command[3:5]
        self.left_j2_delta_pressure = current_command[5:7]
        self.right_j0_delta_pressure = current_command[7:9]
        self.right_j1_delta_pressure = current_command[9:11]
        self.right_j2_delta_pressure = current_command[11:13]

        self.average_pressure = 150
        self.cmd_lower_bound = np.asarray([-1000] + [-150] * 12)
        self.cmd_upper_bound = np.asarray([0] + [150] * 12)

    def __repr__(self):
        return f"Action: {self._to_array()}"

    def _to_array(self):
        return np.hstack([
            self.elevator_height_cmd,
            self.left_j0_delta_pressure,
            self.left_j1_delta_pressure,
            self.left_j2_delta_pressure,
            self.right_j0_delta_pressure,
            self.right_j1_delta_pressure,
            self.right_j2_delta_pressure,
        ])

    def unnormalize(self):
        unnormalized_actions = (self._to_array() + 1) * (
            self.cmd_upper_bound -
            self.cmd_lower_bound) / 2 + self.cmd_lower_bound

        self.elevator_height_cmd = unnormalized_actions[0]
        self.left_j0_delta_pressure = unnormalized_actions[1:3]
        self.left_j1_delta_pressure = unnormalized_actions[3:5]
        self.left_j2_delta_pressure = unnormalized_actions[5:7]
        self.right_j0_delta_pressure = unnormalized_actions[7:9]
        self.right_j1_delta_pressure = unnormalized_actions[9:11]
        self.right_j2_delta_pressure = unnormalized_actions[11:13]

        return self


class IncrementalAction:
    """
    This class is used to store the action vector.
    """
    shape = (25, )

    def __init__(self, current_command):
        self.elevator_height_cmd = np.asarray(current_command[0])
        self.left_j0_pressure = np.asarray(current_command[1:5])
        self.left_j1_pressure = np.asarray(current_command[5:9])
        self.left_j2_pressure = np.asarray(current_command[9:13])
        self.right_j0_pressure = np.asarray(current_command[13:17])
        self.right_j1_pressure = np.asarray(current_command[17:21])
        self.right_j2_pressure = np.asarray(current_command[21:25])

        self.cmd_lower_bound = np.asarray([-1000] + [0] * 24)
        self.cmd_upper_bound = np.asarray([0] + [300] * 24)

        # add flag to declare if action is normalized or not.
        self.is_normalized = True

    def __repr__(self):
        return f"Action: {self._to_array()}"

    def _to_array(self):
        return np.hstack([
            self.elevator_height_cmd,
            self.left_j0_pressure,
            self.left_j1_pressure,
            self.left_j2_pressure,
            self.right_j0_pressure,
            self.right_j1_pressure,
            self.right_j2_pressure,
        ])

    def _saturate(self):
        np.clip(
            self.elevator_height_cmd,
            self.cmd_lower_bound[0],
            self.cmd_upper_bound[0],
            out=self.elevator_height_cmd,
        )
        np.clip(
            self.left_j0_pressure,
            self.cmd_lower_bound[1:5],
            self.cmd_upper_bound[1:5],
            out=self.left_j0_pressure,
        )
        np.clip(
            self.left_j1_pressure,
            self.cmd_lower_bound[5:9],
            self.cmd_upper_bound[5:9],
            out=self.left_j1_pressure,
        )
        np.clip(
            self.left_j2_pressure,
            self.cmd_lower_bound[9:13],
            self.cmd_upper_bound[9:13],
            out=self.left_j2_pressure,
        )
        np.clip(
            self.right_j0_pressure,
            self.cmd_lower_bound[13:17],
            self.cmd_upper_bound[13:17],
            out=self.right_j0_pressure,
        )
        np.clip(
            self.right_j1_pressure,
            self.cmd_lower_bound[17:21],
            self.cmd_upper_bound[17:21],
            out=self.right_j1_pressure,
        )
        np.clip(
            self.right_j2_pressure,
            self.cmd_lower_bound[21:25],
            self.cmd_upper_bound[21:25],
            out=self.right_j2_pressure,
        )

    def update(self, actions):
        """
        actions is a 25 element vector of +1, 0, or -1.
        """
        self.elevator_height_cmd += actions[0] * 50
        self.left_j0_pressure += actions[1:5] * 10
        self.left_j1_pressure += actions[5:9] * 10
        self.left_j2_pressure += actions[9:13] * 10
        self.right_j0_pressure += actions[13:17] * 10
        self.right_j1_pressure += actions[17:21] * 10
        self.right_j2_pressure += actions[21:25] * 10

        self._saturate()

        return self


class IncrementalTorques:
    """
    This class is used to store the command vector.
    """
    shape = (13, )

    def __init__(self, current_command):
        """
        Actions dictate whether to increase, decrease, or maintain the current command to the system.

        The current command is what is being commanded to the system. 

        taus represent the delta pressures between opposing pressure chambers, with x and y

        Example:
        left_j0_pressures = [avg + tau[0]/2, avg - tau[0]/2, avg + tau[1]/2, avg - tau[1]/2]

        Note: divide by 2 because adding/subtracting tau to both chambers results in overall difference of tau

        Args:
            current_command (list): [elevator_height, left_j0_tau, left_j1_tau, left_j2_tau, right_j0_tau, right_j1_tau, right_j2_tau]
        """
        self.elevator_height_cmd = np.asarray(current_command[0])
        self.left_j0_tau_cmd = np.asarray(current_command[1:3])
        self.left_j1_tau_cmd = np.asarray(current_command[3:5])
        self.left_j2_tau_cmd = np.asarray(current_command[5:7])
        self.right_j0_tau_cmd = np.asarray(current_command[7:9])
        self.right_j1_tau_cmd = np.asarray(current_command[9:11])
        self.right_j2_tau_cmd = np.asarray(current_command[11:])

        self.cmd_lower_bound = np.asarray([-1000] + [-300] * 12)
        self.cmd_upper_bound = np.asarray([0] + [300] * 12)

        self.prev_elevator_height_cmd = 0
        self.prev_left_j0_tau_cmd = np.zeros(2)
        self.prev_left_j1_tau_cmd = np.zeros(2)
        self.prev_left_j2_tau_cmd = np.zeros(2)
        self.prev_right_j0_tau_cmd = np.zeros(2)
        self.prev_right_j1_tau_cmd = np.zeros(2)
        self.prev_right_j2_tau_cmd = np.zeros(2)

    def __repr__(self):
        return f"Action: {self._to_array()}"

    def _to_array(self):
        return np.hstack([
            self.elevator_height_cmd,
            self.left_j0_tau_cmd,
            self.left_j1_tau_cmd,
            self.left_j2_tau_cmd,
            self.right_j0_tau_cmd,
            self.right_j1_tau_cmd,
            self.right_j2_tau_cmd,
        ])

    def _saturate(self):
        np.clip(
            self.elevator_height_cmd,
            self.cmd_lower_bound[0],
            self.cmd_upper_bound[0],
            out=self.elevator_height_cmd,
        )
        np.clip(
            self.left_j0_tau_cmd,
            self.cmd_lower_bound[1:3],
            self.cmd_upper_bound[1:3],
            out=self.left_j0_tau_cmd,
        )
        np.clip(
            self.left_j1_tau_cmd,
            self.cmd_lower_bound[3:5],
            self.cmd_upper_bound[3:5],
            out=self.left_j1_tau_cmd,
        )
        np.clip(
            self.left_j2_tau_cmd,
            self.cmd_lower_bound[5:7],
            self.cmd_upper_bound[5:7],
            out=self.left_j2_tau_cmd,
        )
        np.clip(
            self.right_j0_tau_cmd,
            self.cmd_lower_bound[7:9],
            self.cmd_upper_bound[7:9],
            out=self.right_j0_tau_cmd,
        )
        np.clip(
            self.right_j1_tau_cmd,
            self.cmd_lower_bound[9:11],
            self.cmd_upper_bound[9:11],
            out=self.right_j1_tau_cmd,
        )
        np.clip(
            self.right_j2_tau_cmd,
            self.cmd_lower_bound[11:],
            self.cmd_upper_bound[11:],
            out=self.right_j2_tau_cmd,
        )

    def update(self, actions):
        """
        actions is a 13 element vector of +1, 0, or -1.
        """
        self.prev_elevator_height_cmd = self.elevator_height_cmd
        self.prev_left_j0_tau_cmd = np.copy(self.left_j0_tau_cmd)
        self.prev_left_j1_tau_cmd = np.copy(self.left_j1_tau_cmd)
        self.prev_left_j2_tau_cmd = np.copy(self.left_j2_tau_cmd)
        self.prev_right_j0_tau_cmd = np.copy(self.right_j0_tau_cmd)
        self.prev_right_j1_tau_cmd = np.copy(self.right_j1_tau_cmd)
        self.prev_right_j2_tau_cmd = np.copy(self.right_j2_tau_cmd)

        #update new values.
        self.elevator_height_cmd += actions[0] * 10  #mm
        self.left_j0_tau_cmd += actions[1:3] * 10  #kPa
        self.left_j1_tau_cmd += actions[3:5] * 10
        self.left_j2_tau_cmd += actions[5:7] * 10
        self.right_j0_tau_cmd += actions[7:9] * 10
        self.right_j1_tau_cmd += actions[9:11] * 10
        self.right_j2_tau_cmd += actions[11:] * 10

        self._saturate()

        return self
