import numpy as np
from baloo_gym.utils.observation_spaces import StateObservationObjectOnly
from stable_baselines3.common.policies import BasePolicy


class OpenLoopHuggerPolicy(BasePolicy):

    def __init__(self, N):

        super().__init__(observation_space=None, action_space=None)
        self.min_height = -.9
        self.step_along_trajectory = 0
        self.state = "APPROACH"
        self.prev_actions = np.zeros(13)

        self.N = N
        self.actions_lb = np.asarray([-900] + [-150] * 12)
        self.actions_ub = np.asarray([0] + [150] * 12)

        self.avg_pressure = 150
        self.left_lift_j0_delta_traj = np.linspace(
            np.array([0, 0]),
            np.array([85, 0]),
            N,
        )

        self.left_grab_j0_delta_traj = np.linspace(
            np.array([85, 0]),
            np.array([0, 150]),
            N,
        )

        self.left_lift_j1_delta_traj = np.linspace(
            np.array([0, 0]),
            np.array([45, 45]),
            N,
        )

        self.left_grab_j1_delta_traj = np.linspace(
            np.array([45, 45]),
            np.array([-100, 150]),
            N,
        )

        self.right_lift_j0_delta_traj = np.linspace(
            np.array([0, 0]),
            np.array([0, 100]),
            N,
        )

        self.right_grab_j0_delta_traj = np.linspace(
            np.array([0, 100]),
            np.array([150, -50]),
            N,
        )

        self.right_lift_j1_delta_traj = np.linspace(
            np.array([0, 0]),
            np.array([45, 45]),
            N,
        )

        self.right_grab_j1_delta_traj = np.linspace(
            np.array([45, 45]),
            np.array([45, -105]),
            N,
        )

    def _predict(self, observation, deterministic=False):
        return None

    def normalize_actions(self, commands):
        # normalize to -1 to 1
        return 2 * (commands - self.actions_lb) / (self.actions_ub -
                                                   self.actions_lb) - 1

    def predict(self, obs, deterministic=True):
        #get elevator height out of observation [0,1,2]
        #recall that timeawareobservation appends the time step to the end of the observation, so we ignore it here.
        actions = self.prev_actions.copy()
        if len(obs) > 1:
            #to be compatible with gym env, accept whole observation vector
            mujoco_observation = StateObservationObjectOnly.from_standardized_array(
                obs)
            elevator_height = mujoco_observation.elevator_pos
        else:
            #or assume its the height since that's the only thing we need here.
            elevator_height = obs

        if self.state == "APPROACH":
            #command elevator to -.85
            if self.step_along_trajectory < self.N:
                actions[0] = -900

                actions[1] = self.left_lift_j0_delta_traj[
                    self.step_along_trajectory][0]
                actions[2] = self.left_lift_j0_delta_traj[
                    self.step_along_trajectory][1]
                actions[3] = self.left_lift_j1_delta_traj[
                    self.step_along_trajectory][0]
                actions[4] = self.left_lift_j1_delta_traj[
                    self.step_along_trajectory][1]
                actions[5] = 0
                actions[6] = 0

                actions[7] = self.right_lift_j0_delta_traj[
                    self.step_along_trajectory][0]
                actions[8] = self.right_lift_j0_delta_traj[
                    self.step_along_trajectory][1]
                actions[9] = self.right_lift_j1_delta_traj[
                    self.step_along_trajectory][0]
                actions[10] = self.right_lift_j1_delta_traj[
                    self.step_along_trajectory][1]
                actions[11] = 0
                actions[12] = 0

                self.prev_actions = actions.copy()

                self.step_along_trajectory += 1

            #if elevator and pressures are both close, move to GRASP
            if np.isclose(elevator_height, -.900,
                          atol=.1) and self.step_along_trajectory == self.N:
                self.state = "GRASP"
                self.step_along_trajectory = 0

        elif self.state == "GRASP":
            actions[0] = -900

            actions[1] = self.left_grab_j0_delta_traj[
                self.step_along_trajectory][0]
            actions[2] = self.left_grab_j0_delta_traj[
                self.step_along_trajectory][1]
            actions[3] = self.left_grab_j1_delta_traj[
                self.step_along_trajectory][0]
            actions[4] = self.left_grab_j1_delta_traj[
                self.step_along_trajectory][1]
            actions[5] = 0
            actions[6] = 0

            actions[7] = self.right_grab_j0_delta_traj[
                self.step_along_trajectory][0]
            actions[8] = self.right_grab_j0_delta_traj[
                self.step_along_trajectory][1]
            actions[9] = self.right_grab_j1_delta_traj[
                self.step_along_trajectory][0]
            actions[10] = self.right_grab_j1_delta_traj[
                self.step_along_trajectory][1]
            actions[11] = 0
            actions[12] = 0

            self.prev_actions = actions.copy()
            self.step_along_trajectory += 1

            # if arms are close, move to lift
            if self.step_along_trajectory == self.N:
                self.state = "LIFT"
                self.step_along_trajectory = 0

        elif self.state == "LIFT":
            #command elevator to 0
            actions = self.prev_actions.copy()
            actions[0] = 0

        norm_actions = self.normalize_actions(actions)

        return norm_actions, None

    def restart(self):
        self.step_along_trajectory = 0
        self.state = "APPROACH"
        self.prev_actions = np.zeros(13)
