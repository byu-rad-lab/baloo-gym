# def record_rollout(env, policy):
#     obs, info = env.reset()
#     done = False

#     frames = []
#     rewards = []
#     actions = []
#     observations = []

# if baseline:
#    policy = OpenLoopHuggerPolicy()
#    env = OpenLoopBaselineWrapper(env) # makes things compatible with policy without changing anything else from env.

#     while not done:
#         action, _states = policy.predict(obs)
#         observations.append(obs)
#         actions.append(action)
#         frames.append(env.render())
#         obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         rewards.append(reward)

#     env.close()

import numpy as np
from baloo_gym.utils.observation_spaces import StateObservationPressure

#needs to have a policy that takes in an observation from env, then return actions
# so policy would see where elevator is and get a state and step on trajectory.

# then env needs to map actions to commands, step, and return same reward as eval env.


class OpenLoopHuggerPolicy:
    def __init__(self, N):
        self.step_along_trajectory = 0
        self.state = "APPROACH"

        #copied from https://github.com/byu-rad-lab/baloo-data-analysis/blob/main/whole_arm_experiments/open_loop_pressure_hugger.py
        self.left_lift_j0_pressure_traj = np.linspace(
            np.zeros(4), np.array([170, 0, 90, 90]), N)
        self.left_grab_j0_pressure_traj = np.linspace(
            np.array([170, 0, 90, 90]), np.array([90, 90, 300, 0]), N)

        self.left_letgo_j0_pressure_traj = np.linspace(
            np.array([90, 90, 300, 0]), np.array([170, 0, 90, 90]), N)

        self.left_lift_j1_pressure_traj = np.linspace(np.zeros(4),
                                                      np.array([90, 0, 90, 0]),
                                                      N)
        self.left_grab_j1_pressure_traj = np.linspace(
            np.array([90, 0, 90, 0]), np.array([0, 200, 300, 0]), N)
        self.left_letgo_j1_pressure_traj = np.linspace(
            np.array([0, 200, 300, 0]), np.array([90, 0, 90, 0]), N)

        self.right_lift_j0_pressure_traj = np.linspace(
            np.zeros(4), np.array([90, 90, 200, 0]), N)

        self.right_grab_j0_pressure_traj = np.linspace(
            np.array([90, 90, 200, 0]), np.array([300, 0, 100, 200]), N)

        self.right_letgo_j0_pressure_traj = np.linspace(
            np.array([300, 0, 100, 200]), np.array([90, 90, 200, 0]), N)

        self.right_lift_j1_pressure_traj = np.linspace(
            np.zeros(4), np.array([90, 0, 90, 0]), N)

        self.right_grab_j1_pressure_traj = np.linspace(
            np.array([90, 0, 90, 0]), np.array([90, 0, 90, 300]), N)

        self.right_letgo_j1_pressure_traj = np.linspace(
            np.array([90, 0, 90, 300]), np.array([90, 0, 90, 0]), N)

        self.actions = np.zeros(25)

    def predict(self, obs: StateObservationPressure):
        #get elevator height out of observation [0,1,2]
        #recall that timeawareobservation appends the time step to the end of the observation, so we ignore it here.
        mujoco_observation = StateObservationPressure.from_standardized_array(
            obs[:-1])
        if self.state == "APPROACH":
            #command elevator to -.85
            if self.step_along_trajectory < 100:
                self.actions[0] = -850
                #command arms to appropriate point along pressure trajectory
                self.actions[1:5] = self.left_lift_j0_pressure_traj[
                    self.step_along_trajectory]
                self.actions[5:9] = self.left_lift_j1_pressure_traj[
                    self.step_along_trajectory]
                self.actions[9:13] = np.zeros(4)

                self.actions[13:17] = self.right_lift_j0_pressure_traj[
                    self.step_along_trajectory]
                self.actions[17:21] = self.right_lift_j1_pressure_traj[
                    self.step_along_trajectory]
                self.actions[21:25] = np.zeros(4)

                self.prev_actions = self.actions

                self.step_along_trajectory += 1

            #if elevator and pressures are both close, move to GRASP
            if np.isclose(mujoco_observation.elevator_pos, -.85,
                          atol=.1) and self.step_along_trajectory == 100:
                self.state = "GRASP"
                print(f"Changing state to GRASP")
                self.step_along_trajectory = 0

        elif self.state == "GRASP":
            #command arms along grasping trajectory
            self.actions[0] = -850
            self.actions[1:5] = self.left_grab_j0_pressure_traj[
                self.step_along_trajectory]
            self.actions[5:9] = self.left_grab_j1_pressure_traj[
                self.step_along_trajectory]
            self.actions[9:13] = np.zeros(4)

            self.actions[13:17] = self.right_grab_j0_pressure_traj[
                self.step_along_trajectory]
            self.actions[17:21] = self.right_grab_j1_pressure_traj[
                self.step_along_trajectory]
            self.actions[21:25] = np.zeros(4)

            self.prev_actions = self.actions
            self.step_along_trajectory += 1

            # if arms are close, move to lift
            if self.step_along_trajectory == 100:
                self.state = "LIFT"
                self.step_along_trajectory = 0

        elif self.state == "LIFT":
            #command elevator to 0
            self.actions = self.prev_actions
            self.actions[0] = 0
            self.prev_actions = self.actions

        return self.actions, None
