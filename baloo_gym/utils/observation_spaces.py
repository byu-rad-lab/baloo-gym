import numpy as np


class StateObservation:
    shape = (32, )

    def __init__(
        self,
        object_pos,
        object_vel,
        elevator_pos,
        elevator_vel,
        left_pos,
        right_pos,
        left_vel,
        right_vel,
    ):
        self.object_pos_error = object_pos
        self.object_vel = object_vel
        self.elevator_pos = elevator_pos
        self.elevator_vel = elevator_vel
        self.left_j0_pos = left_pos[0:2]
        self.left_j1_pos = left_pos[2:4]
        self.left_j2_pos = left_pos[4:6]
        self.right_j0_pos = right_pos[0:2]
        self.right_j1_pos = right_pos[2:4]
        self.right_j2_pos = right_pos[4:6]
        self.left_j0_vel = left_vel[0:2]
        self.left_j1_vel = left_vel[2:4]
        self.left_j2_vel = left_vel[4:6]
        self.right_j0_vel = right_vel[0:2]
        self.right_j1_vel = right_vel[2:4]
        self.right_j2_vel = right_vel[4:6]

        object_pos_error_lb = [-2, -2, 0]
        object_vel_lb = [-2] * 3
        elevator_pos_lb = [-1.5]
        elevator_vel_lb = [-5]
        left_j0_pos_lb = [-np.pi] * 2
        left_j1_pos_lb = [-np.pi] * 2
        left_j2_pos_lb = [-np.pi] * 2
        right_j0_pos_lb = [-np.pi] * 2
        right_j1_pos_lb = [-np.pi] * 2
        right_j2_pos_lb = [-np.pi] * 2
        left_j0_vel_lb = [-2 * np.pi] * 2
        left_j1_vel_lb = [-2 * np.pi] * 2
        left_j2_vel_lb = [-2 * np.pi] * 2
        right_j0_vel_lb = [-2 * np.pi] * 2
        right_j1_vel_lb = [-2 * np.pi] * 2
        right_j2_vel_lb = [-2 * np.pi] * 2

        self.obs_lower_bound = np.asarray(object_pos_error_lb + object_vel_lb +
                                          elevator_pos_lb + elevator_vel_lb +
                                          left_j0_pos_lb + left_j1_pos_lb +
                                          left_j2_pos_lb + right_j0_pos_lb +
                                          right_j1_pos_lb + right_j2_pos_lb +
                                          left_j0_vel_lb + left_j1_vel_lb +
                                          left_j2_vel_lb + right_j0_vel_lb +
                                          right_j1_vel_lb + right_j2_vel_lb)

        object_pos_error_ub = [2, 2, 2]
        object_vel_ub = [2] * 3
        elevator_pos_ub = [0]
        elevator_vel_ub = [5]
        left_j0_pos_ub = [np.pi] * 2
        left_j1_pos_ub = [np.pi] * 2
        left_j2_pos_ub = [np.pi] * 2
        right_j0_pos_ub = [np.pi] * 2
        right_j1_pos_ub = [np.pi] * 2
        right_j2_pos_ub = [np.pi] * 2
        left_j0_vel_ub = [2 * np.pi] * 2
        left_j1_vel_ub = [2 * np.pi] * 2
        left_j2_vel_ub = [2 * np.pi] * 2
        right_j0_vel_ub = [2 * np.pi] * 2
        right_j1_vel_ub = [2 * np.pi] * 2
        right_j2_vel_ub = [2 * np.pi] * 2

        self.obs_upper_bound = np.asarray(object_pos_error_ub + object_vel_ub +
                                          elevator_pos_ub + elevator_vel_ub +
                                          left_j0_pos_ub + left_j1_pos_ub +
                                          left_j2_pos_ub + right_j0_pos_ub +
                                          right_j1_pos_ub + right_j2_pos_ub +
                                          left_j0_vel_ub + left_j1_vel_ub +
                                          left_j2_vel_ub + right_j0_vel_ub +
                                          right_j1_vel_ub + right_j2_vel_ub)

    def to_array(self):
        return np.hstack([
            self.object_pos_error,
            self.object_vel,
            self.elevator_pos,
            self.elevator_vel,
            self.left_j0_pos,
            self.left_j1_pos,
            self.left_j2_pos,
            self.right_j0_pos,
            self.right_j1_pos,
            self.right_j2_pos,
            self.left_j0_vel,
            self.left_j1_vel,
            self.left_j2_vel,
            self.right_j0_vel,
            self.right_j1_vel,
            self.right_j2_vel,
        ])

    def __repr__(self):
        return f"{self.to_array()}"

    def normalize_and_center(self):
        return (2 * (self.to_array() - self.obs_lower_bound) /
                (self.obs_upper_bound - self.obs_lower_bound) - 1)


class RelativeObservation:
    """
    This class includes some extra information by using relative position/velocity of object relative to torso chest. 

    The object frame is at the COM of the object, and the chest frame is at the torso chest.
    The relative position and velocity are in the world frame and defined as:
    relative_position = object_position - chest_position
    relative_velocity = object_velocity - chest_velocity

    size is 38: 3,3,1,1,6,6,6,6,3,3
    """
    shape = (38, )

    def __init__(
        self,
        object_pos,
        object_vel,
        elevator_pos,
        elevator_vel,
        left_pos,
        right_pos,
        left_vel,
        right_vel,
        chest_pos,
        chest_vel,
    ):
        self.object_pos_error = object_pos
        self.object_vel = object_vel
        self.elevator_pos = elevator_pos
        self.elevator_vel = elevator_vel
        self.left_j0_pos = left_pos[0:2]
        self.left_j1_pos = left_pos[2:4]
        self.left_j2_pos = left_pos[4:6]
        self.right_j0_pos = right_pos[0:2]
        self.right_j1_pos = right_pos[2:4]
        self.right_j2_pos = right_pos[4:6]
        self.left_j0_vel = left_vel[0:2]
        self.left_j1_vel = left_vel[2:4]
        self.left_j2_vel = left_vel[4:6]
        self.right_j0_vel = right_vel[0:2]
        self.right_j1_vel = right_vel[2:4]
        self.right_j2_vel = right_vel[4:6]

        self.relative_object_pos = object_pos - chest_pos
        self.relative_object_vel = object_vel - chest_vel

        object_pos_error_lb = [-2, -2, 0]
        object_vel_lb = [-2] * 3
        elevator_pos_lb = [-1.5]
        elevator_vel_lb = [-5]
        left_j0_pos_lb = [-np.pi] * 2
        left_j1_pos_lb = [-np.pi] * 2
        left_j2_pos_lb = [-np.pi] * 2
        right_j0_pos_lb = [-np.pi] * 2
        right_j1_pos_lb = [-np.pi] * 2
        right_j2_pos_lb = [-np.pi] * 2
        left_j0_vel_lb = [-2 * np.pi] * 2
        left_j1_vel_lb = [-2 * np.pi] * 2
        left_j2_vel_lb = [-2 * np.pi] * 2
        right_j0_vel_lb = [-2 * np.pi] * 2
        right_j1_vel_lb = [-2 * np.pi] * 2
        right_j2_vel_lb = [-2 * np.pi] * 2

        relative_object_pos_error_lb = [-2, -2, -2]
        relative_object_vel_lb = [-2] * 3

        self.obs_lower_bound = np.asarray(object_pos_error_lb + object_vel_lb +
                                          elevator_pos_lb + elevator_vel_lb +
                                          left_j0_pos_lb + left_j1_pos_lb +
                                          left_j2_pos_lb + right_j0_pos_lb +
                                          right_j1_pos_lb + right_j2_pos_lb +
                                          left_j0_vel_lb + left_j1_vel_lb +
                                          left_j2_vel_lb + right_j0_vel_lb +
                                          right_j1_vel_lb + right_j2_vel_lb +
                                          relative_object_pos_error_lb +
                                          relative_object_vel_lb)
        object_pos_error_ub = [2, 2, 2]
        object_vel_ub = [2] * 3
        elevator_pos_ub = [0]
        elevator_vel_ub = [5]
        left_j0_pos_ub = [np.pi] * 2
        left_j1_pos_ub = [np.pi] * 2
        left_j2_pos_ub = [np.pi] * 2
        right_j0_pos_ub = [np.pi] * 2
        right_j1_pos_ub = [np.pi] * 2
        right_j2_pos_ub = [np.pi] * 2
        left_j0_vel_ub = [2 * np.pi] * 2
        left_j1_vel_ub = [2 * np.pi] * 2
        left_j2_vel_ub = [2 * np.pi] * 2
        right_j0_vel_ub = [2 * np.pi] * 2
        right_j1_vel_ub = [2 * np.pi] * 2
        right_j2_vel_ub = [2 * np.pi] * 2

        relative_object_pos_error_ub = [2, 2, 2]
        relative_object_vel_ub = [2] * 3

        self.obs_upper_bound = np.asarray(object_pos_error_ub + object_vel_ub +
                                          elevator_pos_ub + elevator_vel_ub +
                                          left_j0_pos_ub + left_j1_pos_ub +
                                          left_j2_pos_ub + right_j0_pos_ub +
                                          right_j1_pos_ub + right_j2_pos_ub +
                                          left_j0_vel_ub + left_j1_vel_ub +
                                          left_j2_vel_ub + right_j0_vel_ub +
                                          right_j1_vel_ub + right_j2_vel_ub +
                                          relative_object_pos_error_ub +
                                          relative_object_vel_ub)

    def to_array(self):
        return np.hstack([
            self.object_pos_error,
            self.object_vel,
            self.elevator_pos,
            self.elevator_vel,
            self.left_j0_pos,
            self.left_j1_pos,
            self.left_j2_pos,
            self.right_j0_pos,
            self.right_j1_pos,
            self.right_j2_pos,
            self.left_j0_vel,
            self.left_j1_vel,
            self.left_j2_vel,
            self.right_j0_vel,
            self.right_j1_vel,
            self.right_j2_vel,
            self.relative_object_pos,
            self.relative_object_vel,
        ])

    def __repr__(self):
        return f"{self.to_array()}"

    def normalize_and_center(self):
        return (2 * (self.to_array() - self.obs_lower_bound) /
                (self.obs_upper_bound - self.obs_lower_bound) - 1)


class StateObservationPressure:
    #class attributes.
    #designed this way to be able to standardize and unstandardize observations easily without having class isntantiations.
    shape = (32 + 24, )

    object_pos_error_lb = [-2, -2, -2]
    object_vel_lb = [-2] * 3
    elevator_pos_lb = [-1.5]
    elevator_vel_lb = [-5]
    left_j0_pos_lb = [-np.pi] * 2
    left_j1_pos_lb = [-np.pi] * 2
    left_j2_pos_lb = [-np.pi] * 2
    right_j0_pos_lb = [-np.pi] * 2
    right_j1_pos_lb = [-np.pi] * 2
    right_j2_pos_lb = [-np.pi] * 2
    left_j0_vel_lb = [-2 * np.pi] * 2
    left_j1_vel_lb = [-2 * np.pi] * 2
    left_j2_vel_lb = [-2 * np.pi] * 2
    right_j0_vel_lb = [-2 * np.pi] * 2
    right_j1_vel_lb = [-2 * np.pi] * 2
    right_j2_vel_lb = [-2 * np.pi] * 2
    left_j0_p_lb = [0] * 4
    left_j1_p_lb = [0] * 4
    left_j2_p_lb = [0] * 4
    right_j0_p_lb = [0] * 4
    right_j1_p_lb = [0] * 4
    right_j2_p_lb = [0] * 4

    object_pos_error_ub = [2, 2, 2]
    object_vel_ub = [2] * 3
    elevator_pos_ub = [0]
    elevator_vel_ub = [5]
    left_j0_pos_ub = [np.pi] * 2
    left_j1_pos_ub = [np.pi] * 2
    left_j2_pos_ub = [np.pi] * 2
    right_j0_pos_ub = [np.pi] * 2
    right_j1_pos_ub = [np.pi] * 2
    right_j2_pos_ub = [np.pi] * 2
    left_j0_vel_ub = [2 * np.pi] * 2
    left_j1_vel_ub = [2 * np.pi] * 2
    left_j2_vel_ub = [2 * np.pi] * 2
    right_j0_vel_ub = [2 * np.pi] * 2
    right_j1_vel_ub = [2 * np.pi] * 2
    right_j2_vel_ub = [2 * np.pi] * 2
    left_j0_p_ub = [400] * 4
    left_j1_p_ub = [400] * 4
    left_j2_p_ub = [400] * 4
    right_j0_p_ub = [400] * 4
    right_j1_p_ub = [400] * 4
    right_j2_p_ub = [400] * 4

    obs_lower_bound = np.asarray(object_pos_error_lb + object_vel_lb +
                                 elevator_pos_lb + elevator_vel_lb +
                                 left_j0_pos_lb + left_j1_pos_lb +
                                 left_j2_pos_lb + right_j0_pos_lb +
                                 right_j1_pos_lb + right_j2_pos_lb +
                                 left_j0_vel_lb + left_j1_vel_lb +
                                 left_j2_vel_lb + right_j0_vel_lb +
                                 right_j1_vel_lb + right_j2_vel_lb +
                                 left_j0_p_lb + left_j1_p_lb + left_j2_p_lb +
                                 right_j0_p_lb + right_j1_p_lb + right_j2_p_lb)

    obs_upper_bound = np.asarray(object_pos_error_ub + object_vel_ub +
                                 elevator_pos_ub + elevator_vel_ub +
                                 left_j0_pos_ub + left_j1_pos_ub +
                                 left_j2_pos_ub + right_j0_pos_ub +
                                 right_j1_pos_ub + right_j2_pos_ub +
                                 left_j0_vel_ub + left_j1_vel_ub +
                                 left_j2_vel_ub + right_j0_vel_ub +
                                 right_j1_vel_ub + right_j2_vel_ub +
                                 left_j0_p_ub + left_j1_p_ub + left_j2_p_ub +
                                 right_j0_p_ub + right_j1_p_ub + right_j2_p_ub)

    def __init__(
        self,
        object_pos_error,
        object_vel,
        elevator_pos,
        elevator_vel,
        left_pos,
        right_pos,
        left_vel,
        right_vel,
        left_j0_pressures,
        left_j1_pressures,
        left_j2_pressures,
        right_j0_pressures,
        right_j1_pressures,
        right_j2_pressures,
    ):
        self.object_pos_error = object_pos_error  #defined as des_position - actual_position
        self.object_vel = object_vel
        self.elevator_pos = elevator_pos
        self.elevator_vel = elevator_vel
        self.left_j0_pos = left_pos[0:2]
        self.left_j1_pos = left_pos[2:4]
        self.left_j2_pos = left_pos[4:6]
        self.right_j0_pos = right_pos[0:2]
        self.right_j1_pos = right_pos[2:4]
        self.right_j2_pos = right_pos[4:6]
        self.left_j0_vel = left_vel[0:2]
        self.left_j1_vel = left_vel[2:4]
        self.left_j2_vel = left_vel[4:6]
        self.right_j0_vel = right_vel[0:2]
        self.right_j1_vel = right_vel[2:4]
        self.right_j2_vel = right_vel[4:6]
        self.left_j0_p = left_j0_pressures
        self.left_j1_p = left_j1_pressures
        self.left_j2_p = left_j2_pressures
        self.right_j0_p = right_j0_pressures
        self.right_j1_p = right_j1_pressures
        self.right_j2_p = right_j2_pressures

    def to_array(self):
        return np.hstack([
            self.object_pos_error,
            self.object_vel,
            self.elevator_pos,
            self.elevator_vel,
            self.left_j0_pos,
            self.left_j1_pos,
            self.left_j2_pos,
            self.right_j0_pos,
            self.right_j1_pos,
            self.right_j2_pos,
            self.left_j0_vel,
            self.left_j1_vel,
            self.left_j2_vel,
            self.right_j0_vel,
            self.right_j1_vel,
            self.right_j2_vel,
            self.left_j0_p,
            self.left_j1_p,
            self.left_j2_p,
            self.right_j0_p,
            self.right_j1_p,
            self.right_j2_p,
        ])

    def __repr__(self):
        return f"{self.to_array()}"

    def normalize_and_center(self):
        return (2 *
                (self.to_array() - StateObservationPressure.obs_lower_bound) /
                (StateObservationPressure.obs_upper_bound -
                 StateObservationPressure.obs_lower_bound) - 1)

    @staticmethod
    def from_standardized_array(observation_array):

        #unnormalize the observation array
        observation_array = 0.5 * (observation_array + 1) * (
            StateObservationPressure.obs_upper_bound - StateObservationPressure
            .obs_lower_bound) + StateObservationPressure.obs_lower_bound

        return StateObservationPressure(
            object_pos_error=observation_array[0:3],
            object_vel=observation_array[3:6],
            elevator_pos=observation_array[6],
            elevator_vel=observation_array[7],
            left_pos=observation_array[8:14],
            right_pos=observation_array[14:20],
            left_vel=observation_array[20:26],
            right_vel=observation_array[26:32],
            left_j0_pressures=observation_array[32:36],
            left_j1_pressures=observation_array[36:40],
            left_j2_pressures=observation_array[40:44],
            right_j0_pressures=observation_array[44:48],
            right_j1_pressures=observation_array[48:52],
            right_j2_pressures=observation_array[52:56],
        )



    
