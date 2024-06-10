import numpy as np


class Observation:
    # to help with bookkeeping, this class is used to store the observation vector
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
        self.object_pos = object_pos
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

        self.obs_lower_bound = np.asarray([-2, -2, 0] + [-2] * 3 + [-1.5] +
                                          [-5] + [-np.pi] * 6 + [-np.pi] * 6 +
                                          [-2 * np.pi] * 6 + [-2 * np.pi] * 6)

        self.obs_upper_bound = np.asarray([2, 2, 2] + [2] * 3 + [0] + [5] +
                                          [np.pi] * 6 + [np.pi] * 6 +
                                          [2 * np.pi] * 6 + [2 * np.pi] * 6)

    def to_array(self):
        return np.hstack([
            self.object_pos,
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
