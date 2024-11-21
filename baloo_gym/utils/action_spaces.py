import numpy as np


class NormalizedAction:
    shape = (25, )

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


class NormalizedDifferentialPressure:
    shape = (13, )

    def __init__(self, normalized_action_vector):
        self.elevator_height = normalized_action_vector[0]
        self.left_j0_delta_pressure = normalized_action_vector[1:3]
        self.left_j1_delta_pressure = normalized_action_vector[3:5]
        self.left_j2_delta_pressure = normalized_action_vector[5:7]
        self.right_j0_delta_pressure = normalized_action_vector[7:9]
        self.right_j1_delta_pressure = normalized_action_vector[9:11]
        self.right_j2_delta_pressure = normalized_action_vector[11:13]

        self.average_pressure = 150
        self.action_lower_bound = np.asarray([-1000] + [-150] * 12)
        self.action_upper_bound = np.asarray([0] + [150] * 12)

    def __repr__(self):
        return f"Action: {self._to_array()}"

    def _to_array(self):
        return np.hstack([
            self.elevator_height,
            self.left_j0_delta_pressure,
            self.left_j1_delta_pressure,
            self.left_j2_delta_pressure,
            self.right_j0_delta_pressure,
            self.right_j1_delta_pressure,
            self.right_j2_delta_pressure,
        ])

    def unnormalize(self):
        unnormalized_actions = (self._to_array() + 1) * (
            self.action_upper_bound -
            self.action_lower_bound) / 2 + self.action_lower_bound

        self.elevator_height = unnormalized_actions[0]
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


class IncrementalTorques:
    """
    This class is used to store the action vector.
    """
    shape = (13, )

    def __init__(self, normalized_action_vector):
        """_summary_
        taus represent the delta pressures between opposing pressure chambers, with x and y

        Example:
        left_j0_pressures = [avg + tau[0]/2, avg - tau[0]/2, avg + tau[1]/2, avg - tau[1]/2]

        Note: divide by 2 because adding/subtracting tau to both chambers results in overall difference of tau

        Args:
            normalized_action_vector (_type_): _description_
        """
        self.elevator_height = np.asarray(normalized_action_vector[0])
        self.left_j0_tau = np.asarray(normalized_action_vector[1:3])
        self.left_j1_tau = np.asarray(normalized_action_vector[3:5])
        self.left_j2_tau = np.asarray(normalized_action_vector[5:7])
        self.right_j0_tau = np.asarray(normalized_action_vector[7:9])
        self.right_j1_tau = np.asarray(normalized_action_vector[9:11])
        self.right_j2_tau = np.asarray(normalized_action_vector[11:])

        self.action_lower_bound = np.asarray([-1000] + [-300] * 12)
        self.action_upper_bound = np.asarray([0] + [300] * 12)

        # add flag to declare if action is normalized or not.
        self.is_normalized = True

        self.prev_elevator_height = 0
        self.prev_left_j0_tau = np.zeros(2)
        self.prev_left_j1_tau = np.zeros(2)
        self.prev_left_j2_tau = np.zeros(2)
        self.prev_right_j0_tau = np.zeros(2)
        self.prev_right_j1_tau = np.zeros(2)
        self.prev_right_j2_tau = np.zeros(2)

    def __repr__(self):
        return f"Action: {self._to_array()}"

    def _to_array(self):
        return np.hstack([
            self.elevator_height,
            self.left_j0_tau,
            self.left_j1_tau,
            self.left_j2_tau,
            self.right_j0_tau,
            self.right_j1_tau,
            self.right_j2_tau,
        ])

    def _saturate(self):
        np.clip(
            self.elevator_height,
            self.action_lower_bound[0],
            self.action_upper_bound[0],
            out=self.elevator_height,
        )
        np.clip(
            self.left_j0_tau,
            self.action_lower_bound[1:3],
            self.action_upper_bound[1:3],
            out=self.left_j0_tau,
        )
        np.clip(
            self.left_j1_tau,
            self.action_lower_bound[3:5],
            self.action_upper_bound[3:5],
            out=self.left_j1_tau,
        )
        np.clip(
            self.left_j2_tau,
            self.action_lower_bound[5:7],
            self.action_upper_bound[5:7],
            out=self.left_j2_tau,
        )
        np.clip(
            self.right_j0_tau,
            self.action_lower_bound[7:9],
            self.action_upper_bound[7:9],
            out=self.right_j0_tau,
        )
        np.clip(
            self.right_j1_tau,
            self.action_lower_bound[9:11],
            self.action_upper_bound[9:11],
            out=self.right_j1_tau,
        )
        np.clip(
            self.right_j2_tau,
            self.action_lower_bound[11:],
            self.action_upper_bound[11:],
            out=self.right_j2_tau,
        )

    def increment(self, increment_directions):
        """
        increment_directions is a 13 element vector of +1, 0, or -1.
        """
        self.prev_elevator_height = self.elevator_height
        self.prev_left_j0_tau = np.copy(self.left_j0_tau)
        self.prev_left_j1_tau = np.copy(self.left_j1_tau)
        self.prev_left_j2_tau = np.copy(self.left_j2_tau)
        self.prev_right_j0_tau = np.copy(self.right_j0_tau)
        self.prev_right_j1_tau = np.copy(self.right_j1_tau)
        self.prev_right_j2_tau = np.copy(self.right_j2_tau)

        #update new values.
        self.elevator_height += increment_directions[0] * 10 #mm
        self.left_j0_tau += increment_directions[1:3] * 10 #kPa
        self.left_j1_tau += increment_directions[3:5] * 10
        self.left_j2_tau += increment_directions[5:7] * 10
        self.right_j0_tau += increment_directions[7:9] * 10
        self.right_j1_tau += increment_directions[9:11] * 10
        self.right_j2_tau += increment_directions[11:] * 10

        self._saturate()

        return self
