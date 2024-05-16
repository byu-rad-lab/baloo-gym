import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod
import mujoco
from utils.mujoco_rendering import MujocoRenderer


class BalooBase(gym.Env, ABC):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}
    '''
    This class takes care of all of the mujoco stuff and gym stuff. Users need to inherit from this
    and implement their own observations, rewards, and action spaces. 

    Then the appropriate functions need to be implemented to map the actions to the mujoco model, 
    which accepts pressure commands as inputs. 

    WORKFLOW to implement a new environment:
    1. Decide on action_space and implement map_action_to_pressure_commands(). This m
    2. Decide on observation_space and implement get_observation_from_mujoco()
    3. Design reward function and implement calculate_reward(). This must return a float. todo: MAYBE ONLY VIA WRAPPER?
    '''
    def __init__(
        self,
        render_mode=None,
        camera_id=None,
        camera_name=None,
        xml_path=None,
        ctrl_timestep=0.01,
    ):
        super().__init__()

        self.action_space = None
        self.observation_space = None

        self.camera_id = camera_id
        self.camera_name = camera_name
        self.render_mode = render_mode
        self.xml_path = xml_path

        self._reinitialize_states()

        self.simulation_timestep = self.model.opt.timestep
        self.control_timestep = ctrl_timestep  # seconds

        assert int(self.control_timestep % self.simulation_timestep) == 0
        self.sim_steps_per_control_step = int(self.control_timestep /
                                              self.simulation_timestep)

    @abstractmethod
    def map_action_to_pressure_commands(self, action):
        pass

    @abstractmethod
    def get_observation_from_mujoco(self):
        pass

    @abstractmethod
    def calculate_reward(self) -> float:
        pass

    def _reinitialize_states(self):
        self._initialize_model_from_xml()

    def _initialize_model_from_xml(self):
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.mujoco_renderer = MujocoRenderer(self.model,
                                              self.data,
                                              max_geom=100000)

    def step(self, action):

        pressure_commands = self.map_action_to_pressure_commands(action)

        assert len(
            pressure_commands
        ) == 24, "Pressure commands must be a list of 24 floats, in correct order."

        # apply pressure commands to the model for forward simulation
        #todo: check this order of commands, or implement name-based method to fill in commands
        self.data.ctrl[1:] = pressure_commands

        # step the model forward in time however many steps are needed to match the control timestep
        mujoco.mj_step(self.model,
                       self.data,
                       nstep=self.sim_steps_per_control_step)

        observation = self.get_observation_from_mujoco()

        info = {}
        reward = self.calculate_reward()
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # self._setup_visualization()
        self._reinitialize_states()
        # todo: randomly set position (and size) of box eventually.

        observation = self.get_observation_from_mujoco()
        info = {}

        return observation, info

    def render(self):
        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
