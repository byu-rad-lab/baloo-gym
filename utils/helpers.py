from baloo_mujoco_sim.utils.baloo_mj_api import (
    get_box_position,
    get_box_vel,
    get_elevator_height,
    get_elevator_vel,
    get_joint_angles,
    get_joint_vel,
)

import numpy as np


def get_sensor_data(model, data):
    left_pos = []
    left_vel = []
    right_pos = []
    right_vel = []
    for i in range(3):
        left_pos.append(get_joint_angles(model, data, "left", i))
        left_vel.append(get_joint_vel(model, data, "left", i))
        right_pos.append(get_joint_angles(model, data, "right", i))
        right_vel.append(get_joint_vel(model, data, "right", i))

    object_pos = get_box_position(model, data)
    object_vel = get_box_vel(model, data)
    elevator_pos = get_elevator_height(model, data)
    elevator_vel = get_elevator_vel(model, data)

    return {
        "object_pos": object_pos,
        "object_vel": object_vel,
        "elevator_pos": elevator_pos,
        "elevator_vel": elevator_vel,
        "left_pos": np.hstack(left_pos),
        "right_pos": np.hstack(right_pos),
        "left_vel": np.hstack(left_vel),
        "right_vel": np.hstack(right_vel),
    }
