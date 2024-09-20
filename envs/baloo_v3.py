from envs.baloo_base import BalooBase
from gymnasium import spaces
import numpy as np
from utils.observation import RelativeObservation
from utils.helpers import get_sensor_data
from baloo_mujoco_sim.utils.baloo_mj_api import get_elevator_vel


class IncrementalTorques:
    """
    This class is used to store the action vector.
    """
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
        self.elevator_height += increment_directions[0] * 50
        self.left_j0_tau += increment_directions[1:3] * 1
        self.left_j1_tau += increment_directions[3:5] * 1
        self.left_j2_tau += increment_directions[5:7] * 1
        self.right_j0_tau += increment_directions[7:9] * 1
        self.right_j1_tau += increment_directions[9:11] * 1
        self.right_j2_tau += increment_directions[11:] * 1

        self._saturate()

        return self


class BalooV3(BalooBase):
    '''
    BalooV3 implements an environment where the actions 
    are incremental torques on the arms and elevator.

    This was done in an effort to shrink the action space
    to hopefully speed up learning. 

    It also builds on V2 by adding some extras into the state representation, like the relative positive and velocity 
    between the torso and the manipuland. 

    Might also include some sort of tactile sensing in the state as well. 
    '''
    def __init__(
        self,
        render_mode=None,
        camera_name=None,
        ctrl_timestep=0.01,
        render_width=320,
        render_height=240,
    ):
        super().__init__(
            render_mode=render_mode,
            camera_name=camera_name,
            ctrl_timestep=ctrl_timestep,
            render_width=render_width,
            render_height=render_height,
        )

        #action space is incremental position on elevator (1) then torques on arms (12)
        self.action_space = self.action_space = spaces.MultiDiscrete([3] * 13)

        #see Observation class in utils/observation.py for more details
        self.observation_space = spaces.Box(-1,
                                            1,
                                            shape=RelativeObservation.shape,
                                            dtype=np.float32)

        self.current_actions = IncrementalTorques(np.zeros(13))

    def get_observation_from_mujoco(self):
        chest_pos = self.data.geom('chest').xpos
        chest_vel = np.array(
            [0, 0, get_elevator_vel(self.model, self.data)[0]])
        rawObs = RelativeObservation(**get_sensor_data(self.model, self.data),
                                     chest_pos=chest_pos,
                                     chest_vel=chest_vel)

        return rawObs.normalize_and_center().astype(
            self.observation_space.dtype)

    def map_action_to_commands(self, action):
        # ! gym 0.29.0 added start parameter for multidiscrete action space [-1,1] and sb3 2.1.0 I don't think follows it. [0,2]
        # ! downgrading to 0.28.1 and 2.0.0 eliminates the start param issue, but now actions can only be 0,1,or2.

        action = np.asarray(action) - 1  # shift to be between -1 and 1.

        self.current_actions.increment(action)

        #make this more concise later with a loop
        commands = np.zeros(25)
        commands[0] = self.current_actions.elevator_height
        commands[1] = 150 + self.current_actions.left_j0_tau[0] / 2
        commands[2] = 150 - self.current_actions.left_j0_tau[0] / 2

        commands[3] = 150 + self.current_actions.left_j0_tau[1] / 2
        commands[4] = 150 - self.current_actions.left_j0_tau[1] / 2

        commands[5] = 150 + self.current_actions.left_j1_tau[0] / 2
        commands[6] = 150 - self.current_actions.left_j1_tau[0] / 2

        commands[7] = 150 + self.current_actions.left_j1_tau[1] / 2
        commands[8] = 150 - self.current_actions.left_j1_tau[1] / 2

        commands[9] = 150 + self.current_actions.left_j2_tau[0] / 2
        commands[10] = 150 - self.current_actions.left_j2_tau[0] / 2

        commands[11] = 150 + self.current_actions.left_j2_tau[1] / 2
        commands[12] = 150 - self.current_actions.left_j2_tau[1] / 2

        commands[13] = 150 + self.current_actions.right_j0_tau[0] / 2
        commands[14] = 150 - self.current_actions.right_j0_tau[0] / 2

        commands[15] = 150 + self.current_actions.right_j0_tau[1] / 2
        commands[16] = 150 - self.current_actions.right_j0_tau[1] / 2

        commands[17] = 150 + self.current_actions.right_j1_tau[0] / 2
        commands[18] = 150 - self.current_actions.right_j1_tau[0] / 2

        commands[19] = 150 + self.current_actions.right_j1_tau[1] / 2
        commands[20] = 150 - self.current_actions.right_j1_tau[1] / 2

        commands[21] = 150 + self.current_actions.right_j2_tau[0] / 2
        commands[22] = 150 - self.current_actions.right_j2_tau[0] / 2

        commands[23] = 150 + self.current_actions.right_j2_tau[1] / 2
        commands[24] = 150 - self.current_actions.right_j2_tau[1] / 2

        return commands

    def calculate_reward(self):
        return 0
