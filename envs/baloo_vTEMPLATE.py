from envs.baloo_base import BalooBase

class BalooVTemplate(BalooBase):
    '''
    BalooV0 implements an environment where 
    '''
    def __init__(
        self,
        render_mode=None,
        camera_id=None,
        camera_name=None,
        ctrl_timestep=0.01,
    ):
        super().__init__(render_mode=render_mode,
                         camera_id=camera_id,
                         camera_name=camera_name,
                         ctrl_timestep=ctrl_timestep)

        #TODO: define action space
        self.action_space = None

        #TODO: define observation space
        self.observation_space = None

    def get_observation_from_mujoco(self):
        #TODO: implement observation space using mujoco data
        pass

    def map_action_to_commands(self, action):
        #TODO: map actions to commands that mujoco accepts
        pass

    def calculate_reward(self):
        #TODO: implement reward function
        pass
