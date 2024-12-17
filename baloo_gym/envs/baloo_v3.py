from baloo_gym.envs.baloo_base import BalooBase
from gymnasium import spaces
import numpy as np
from baloo_gym.utils.observation_spaces import RelativeObservation
from baloo_gym.utils.helpers import get_sensor_data
from baloo_mujoco_sim.utils.baloo_mj_api import get_elevator_vel, get_joint_pressures
from baloo_gym.utils.action_spaces import IncrementalTorques


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
        desired_box_pos=None,
        randomize_initial_height=False,
        randomize_object_size=False,
        randomize_object_mass=False,
    ):
        super().__init__(
            render_mode=render_mode,
            camera_name=camera_name,
            ctrl_timestep=ctrl_timestep,
            render_width=render_width,
            render_height=render_height,
            randomize_initial_height=randomize_initial_height
        )

        #action space is incremental position on elevator (1) then torques on arms (12)
        action_size = IncrementalTorques.shape[0]
        self.action_space = self.action_space = spaces.MultiDiscrete(
            [3] * action_size)

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

        # print(rawObs.left_j0_pos)
        # print(
        #     f"Joint0 Pressures: {get_joint_pressures(self.model, self.data, 'left', 0)}"
        # )

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

    def reset(self, seed=None, options=None):
        self.current_actions = IncrementalTorques(np.zeros(13))
        return super().reset(seed=seed, options=options)
