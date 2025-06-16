'''
Run this script after regenerating each model with N = 2,4,8,16,32,64, and save the data. Then compare each agasint N=64.
'''

import time
import baloo_mujoco_sim as baloo_mj
import mujoco
import numpy as np
from baloo_gym.policies.open_loop_hugger import OpenLoopHuggerPolicy
from baloo_gym.envs.baloo_v9 import LowPassFilter
from baloo_mujoco_sim.utils.baloo_mj_api import (
    set_elevator_cmd,
    set_joint_pressure_commands,
    get_joint_angles,
    get_disk_position,
    get_disk_quat,
    get_elevator_height,
)

import mujoco.viewer

from baloo_gym.utils.action_spaces import NormalizedDifferentialPressure

model = mujoco.MjModel.from_xml_path(baloo_mj.XML_PATH)
data = mujoco.MjData(model)

num_disks = int(input("How many disks is the model configured for? "))

policy = OpenLoopHuggerPolicy(N=50)
lpf = LowPassFilter(rise_time=3, sampling_period=.05)


def map_actions_to_commands(actions):
    '''
    actions come in from the policy in a normalized form. This function
    needs to map them to actual commands that can be sent to the robot. 
    '''

    pressure_actions = actions[1:]
    filtered_pressure_actions = lpf.apply_filter(pressure_actions)

    filtered_actions = np.zeros_like(actions)
    filtered_actions[0] = actions[0]
    filtered_actions[1:] = filtered_pressure_actions

    unnormalized_actions = NormalizedDifferentialPressure(
        filtered_actions).unnormalize()

    #make this more concise later with a loop
    commands = np.zeros(25)
    commands[0] = unnormalized_actions.elevator_height_cmd
    commands[1] = 150 + unnormalized_actions.left_j0_delta_pressure[0]
    commands[2] = 150 - unnormalized_actions.left_j0_delta_pressure[0]

    commands[3] = 150 + unnormalized_actions.left_j0_delta_pressure[1]
    commands[4] = 150 - unnormalized_actions.left_j0_delta_pressure[1]

    commands[5] = 150 + unnormalized_actions.left_j1_delta_pressure[0]
    commands[6] = 150 - unnormalized_actions.left_j1_delta_pressure[0]

    commands[7] = 150 + unnormalized_actions.left_j1_delta_pressure[1]
    commands[8] = 150 - unnormalized_actions.left_j1_delta_pressure[1]

    commands[9] = 150 + unnormalized_actions.left_j2_delta_pressure[0]
    commands[10] = 150 - unnormalized_actions.left_j2_delta_pressure[0]

    commands[11] = 150 + unnormalized_actions.left_j2_delta_pressure[1]
    commands[12] = 150 - unnormalized_actions.left_j2_delta_pressure[1]

    commands[13] = 150 + unnormalized_actions.right_j0_delta_pressure[0]
    commands[14] = 150 - unnormalized_actions.right_j0_delta_pressure[0]

    commands[15] = 150 + unnormalized_actions.right_j0_delta_pressure[1]
    commands[16] = 150 - unnormalized_actions.right_j0_delta_pressure[1]

    commands[17] = 150 + unnormalized_actions.right_j1_delta_pressure[0]
    commands[18] = 150 - unnormalized_actions.right_j1_delta_pressure[0]

    commands[19] = 150 + unnormalized_actions.right_j1_delta_pressure[1]
    commands[20] = 150 - unnormalized_actions.right_j1_delta_pressure[1]

    commands[21] = 150 + unnormalized_actions.right_j2_delta_pressure[0]
    commands[22] = 150 - unnormalized_actions.right_j2_delta_pressure[0]

    commands[23] = 150 + unnormalized_actions.right_j2_delta_pressure[1]
    commands[24] = 150 - unnormalized_actions.right_j2_delta_pressure[1]

    return commands


left_q0_traj = []
left_q1_traj = []
left_q2_traj = []

right_q0_traj = []
right_q1_traj = []
right_q2_traj = []

left_j0_tip_pose_traj = []
left_j1_tip_pose_traj = []
left_j2_tip_pose_traj = []
left_j0_base_pose_traj = []
left_j1_base_pose_traj = []
left_j2_base_pose_traj = []

right_j0_tip_pose_traj = []
right_j1_tip_pose_traj = []
right_j2_tip_pose_traj = []
right_j0_base_pose_traj = []
right_j1_base_pose_traj = []
right_j2_base_pose_traj = []

with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.sync()

    start = time.time()

    while viewer.is_running() and data.time < 60:
        actions, _ = policy.predict(get_elevator_height(model, data))

        # map actions to actual commands witg baloo_v9
        commands = map_actions_to_commands(actions)

        #take actions and set appropriate data in mujoco sim
        set_elevator_cmd(model, data, commands[0])
        set_joint_pressure_commands(model, data, "left", 0, commands[1:5])
        set_joint_pressure_commands(model, data, "left", 1, commands[5:9])
        set_joint_pressure_commands(model, data, "left", 2, commands[9:13])

        set_joint_pressure_commands(model, data, "right", 0, commands[13:17])
        set_joint_pressure_commands(model, data, "right", 1, commands[17:21])
        set_joint_pressure_commands(model, data, "right", 2, commands[21:])

        mujoco.mj_step(model, data, nstep=1)
        viewer.sync()

        #get joint positions as well as the actual tip pose of the last disk of each joint
        left_q0 = get_joint_angles(model, data, "left", 0)
        left_q1 = get_joint_angles(model, data, "left", 1)
        left_q2 = get_joint_angles(model, data, "left", 2)

        right_q0 = get_joint_angles(model, data, "right", 0)
        right_q1 = get_joint_angles(model, data, "right", 1)
        right_q2 = get_joint_angles(model, data, "right", 2)

        left_j0_tip_pos = get_disk_position(model, data, "left", 0, -1)
        left_j1_tip_pos = get_disk_position(model, data, "left", 1, -1)
        left_j2_tip_pos = get_disk_position(model, data, "left", 2, -1)
        left_j0_base_pos = get_disk_position(model, data, "left", 0, 0)
        left_j1_base_pos = get_disk_position(model, data, "left", 1, 0)
        left_j2_base_pos = get_disk_position(model, data, "left", 2, 0)

        right_j0_tip_pos = get_disk_position(model, data, "right", 0, -1)
        right_j1_tip_pos = get_disk_position(model, data, "right", 1, -1)
        right_j2_tip_pos = get_disk_position(model, data, "right", 2, -1)
        right_j0_base_pos = get_disk_position(model, data, "right", 0, 0)
        right_j1_base_pos = get_disk_position(model, data, "right", 1, 0)
        right_j2_base_pos = get_disk_position(model, data, "right", 2, 0)

        left_j0_tip_quat = get_disk_quat(model, data, "left", 0, -1)
        left_j1_tip_quat = get_disk_quat(model, data, "left", 1, -1)
        left_j2_tip_quat = get_disk_quat(model, data, "left", 2, -1)
        left_j0_base_quat = get_disk_quat(model, data, "left", 0, 0)
        left_j1_base_quat = get_disk_quat(model, data, "left", 1, 0)
        left_j2_base_quat = get_disk_quat(model, data, "left", 2, 0)

        right_j0_tip_quat = get_disk_quat(model, data, "right", 0, -1)
        right_j1_tip_quat = get_disk_quat(model, data, "right", 1, -1)
        right_j2_tip_quat = get_disk_quat(model, data, "right", 2, -1)
        right_j0_base_quat = get_disk_quat(model, data, "right", 0, 0)
        right_j1_base_quat = get_disk_quat(model, data, "right", 1, 0)
        right_j2_base_quat = get_disk_quat(model, data, "right", 2, 0)

        #save this data
        left_q0_traj.append(left_q0.copy())
        left_q1_traj.append(left_q1.copy())
        left_q2_traj.append(left_q2.copy())

        right_q0_traj.append(right_q0.copy())
        right_q1_traj.append(right_q1.copy())
        right_q2_traj.append(right_q2.copy())

        left_j0_tip_pose_traj.append(
            np.hstack([left_j0_tip_pos, left_j0_tip_quat]))
        left_j1_tip_pose_traj.append(
            np.hstack([left_j1_tip_pos, left_j1_tip_quat]))
        left_j2_tip_pose_traj.append(
            np.hstack([left_j2_tip_pos, left_j2_tip_quat]))

        left_j0_base_pose_traj.append(
            np.hstack([left_j0_base_pos, left_j0_base_quat]))
        left_j1_base_pose_traj.append(
            np.hstack([left_j1_base_pos, left_j1_base_quat]))
        left_j2_base_pose_traj.append(
            np.hstack([left_j2_base_pos, left_j2_base_quat]))

        right_j0_tip_pose_traj.append(
            np.hstack([right_j0_tip_pos, right_j0_tip_quat]))
        right_j1_tip_pose_traj.append(
            np.hstack([right_j1_tip_pos, right_j1_tip_quat]))
        right_j2_tip_pose_traj.append(
            np.hstack([right_j2_tip_pos, right_j2_tip_quat]))

        right_j0_base_pose_traj.append(
            np.hstack([right_j0_base_pos, right_j0_base_quat]))
        right_j1_base_pose_traj.append(
            np.hstack([right_j1_base_pos, right_j1_base_quat]))
        right_j2_base_pose_traj.append(
            np.hstack([right_j2_base_pos, right_j2_base_quat]))

#save data to file
np.save(f"{num_disks}-disks/left_q0_traj.npy", np.array(left_q0_traj))
np.save(f"{num_disks}-disks/left_q1_traj.npy", np.array(left_q1_traj))
np.save(f"{num_disks}-disks/left_q2_traj.npy", np.array(left_q2_traj))

np.save(f"{num_disks}-disks/right_q0_traj.npy", np.array(right_q0_traj))
np.save(f"{num_disks}-disks/right_q1_traj.npy", np.array(right_q1_traj))
np.save(f"{num_disks}-disks/right_q2_traj.npy", np.array(right_q2_traj))

np.save(f"{num_disks}-disks/left_j0_base_pose_traj.npy",
        np.array(left_j0_base_pose_traj))
np.save(f"{num_disks}-disks/left_j0_tip_pose_traj.npy",
        np.array(left_j0_tip_pose_traj))

np.save(f"{num_disks}-disks/left_j1_base_pose_traj.npy",
        np.array(left_j1_base_pose_traj))
np.save(f"{num_disks}-disks/left_j1_tip_pose_traj.npy",
        np.array(left_j1_tip_pose_traj))

np.save(f"{num_disks}-disks/left_j2_base_pose_traj.npy",
        np.array(left_j2_base_pose_traj))
np.save(f"{num_disks}-disks/left_j2_tip_pose_traj.npy",
        np.array(left_j2_tip_pose_traj))

np.save(f"{num_disks}-disks/right_j0_base_pose_traj.npy",
        np.array(right_j0_base_pose_traj))
np.save(f"{num_disks}-disks/right_j0_tip_pose_traj.npy",
        np.array(right_j0_tip_pose_traj))

np.save(f"{num_disks}-disks/right_j1_base_pose_traj.npy",
        np.array(right_j1_base_pose_traj))
np.save(f"{num_disks}-disks/right_j1_tip_pose_traj.npy",
        np.array(right_j1_tip_pose_traj))

np.save(f"{num_disks}-disks/right_j2_base_pose_traj.npy",
        np.array(right_j2_base_pose_traj))
np.save(f"{num_disks}-disks/right_j2_tip_pose_traj.npy",
        np.array(right_j2_tip_pose_traj))

#plot left q0 trajectory just as test
import matplotlib.pyplot as plt

plt.figure()
plt.plot(left_q0_traj)

plt.figure()
plt.plot(left_j0_tip_pose_traj)

plt.show()
