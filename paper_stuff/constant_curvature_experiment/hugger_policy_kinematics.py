import time
import baloo_mujoco_sim as baloo_mj
import mujoco
import numpy as np
from baloo_gym.policies.open_loop_hugger import OpenLoopHuggerPolicy
from baloo_mujoco_sim.utils.baloo_mj_api import (
    set_elevator_cmd,
    set_joint_pressure_commands,
    get_joint_angles,
    get_disk_position,
    get_disk_quat,
    get_elevator_height,
)

import mujoco.viewer

from baloo_gym.utils.observation_spaces import StateObservationPressure

model = mujoco.MjModel.from_xml_path(baloo_mj.XML_PATH)
data = mujoco.MjData(model)

policy = OpenLoopHuggerPolicy(N=2000)

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

    while viewer.is_running() and data.time < 30:
        actions, _ = policy.predict(get_elevator_height(model, data))

        #take actions and set appropriate data in mujoco sim
        set_elevator_cmd(model, data, actions[0])
        set_joint_pressure_commands(model, data, "left", 0, actions[1:5])
        set_joint_pressure_commands(model, data, "left", 1, actions[5:9])
        set_joint_pressure_commands(model, data, "left", 2, actions[9:13])

        set_joint_pressure_commands(model, data, "right", 0, actions[13:17])
        set_joint_pressure_commands(model, data, "right", 1, actions[17:21])
        set_joint_pressure_commands(model, data, "right", 2, actions[21:])

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
np.save("left_q0_traj.npy", np.array(left_q0_traj))
np.save("left_q1_traj.npy", np.array(left_q1_traj))
np.save("left_q2_traj.npy", np.array(left_q2_traj))

np.save("right_q0_traj.npy", np.array(right_q0_traj))
np.save("right_q1_traj.npy", np.array(right_q1_traj))
np.save("right_q2_traj.npy", np.array(right_q2_traj))

np.save("left_j0_base_pose_traj.npy", np.array(left_j0_base_pose_traj))
np.save("left_j0_tip_pose_traj.npy", np.array(left_j0_tip_pose_traj))

np.save("left_j1_base_pose_traj.npy", np.array(left_j1_base_pose_traj))
np.save("left_j1_tip_pose_traj.npy", np.array(left_j1_tip_pose_traj))

np.save("left_j2_base_pose_traj.npy", np.array(left_j2_base_pose_traj))
np.save("left_j2_tip_pose_traj.npy", np.array(left_j2_tip_pose_traj))

np.save("right_j0_base_pose_traj.npy", np.array(right_j0_base_pose_traj))
np.save("right_j0_tip_pose_traj.npy", np.array(right_j0_tip_pose_traj))

np.save("right_j1_base_pose_traj.npy", np.array(right_j1_base_pose_traj))
np.save("right_j1_tip_pose_traj.npy", np.array(right_j1_tip_pose_traj))

np.save("right_j2_base_pose_traj.npy", np.array(right_j2_base_pose_traj))
np.save("right_j2_tip_pose_traj.npy", np.array(right_j2_tip_pose_traj))

#plot left q0 trajectory just as test
import matplotlib.pyplot as plt

plt.figure()
plt.plot(left_q0_traj)

plt.figure()
plt.plot(left_j0_tip_pose_traj)

plt.show()
