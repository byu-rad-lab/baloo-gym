import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod
import mujoco
from baloo_gym.utils.mujoco_rendering import MujocoRenderer
import baloo_mujoco_sim as baloo_mj
from baloo_mujoco_sim.utils.baloo_mj_api import (
    set_elevator_cmd,
    set_joint_pressure_commands,
)
import os
import numpy as np


class BalooBase(gym.Env, ABC):
    metadata = {"render_modes": ["human", "rgb_array"]}
    '''
    This class takes care of all of the mujoco stuff and gym stuff. Users need to inherit from this
    and implement their own observations, rewards, and action spaces. 

    Then the appropriate functions need to be implemented to map the actions to the mujoco model, 
    which accepts pressure commands as inputs. 

    WORKFLOW to implement a new environment:
    1. Decide on action_space and implement map_action_to_pressure_commands().
    2. Decide on observation_space and implement get_observation_from_mujoco()
    3. Design reward function and implement calculate_reward(). This can be overridden with a wrapper since 
    this will likely change frequently.
    '''

    def __init__(
        self,
        render_mode=None,
        camera_name=None,
        ctrl_timestep=0.01,
        render_width=320,
        render_height=240,
    ):
        super().__init__()

        self.action_space = None
        self.observation_space = None

        self.camera_name = camera_name
        self.render_mode = render_mode

        self.render_width = render_width
        self.render_height = render_height

        self.len_command = 25

        self.xml_path = baloo_mj.XML_PATH

        self.first_load = True
        self._reinitialize_states()

        self.simulation_timestep = self.model.opt.timestep
        self.control_timestep = ctrl_timestep  # seconds
        self.metadata["render_fps"] = int(1 / self.control_timestep)

        assert int(self.control_timestep % self.simulation_timestep) == 0
        self.sim_steps_per_control_step = int(self.control_timestep /
                                              self.simulation_timestep)

    @abstractmethod
    def map_action_to_commands(self, action):
        pass

    @abstractmethod
    def get_observation_from_mujoco(self):
        pass

    @abstractmethod
    def calculate_reward(self) -> float:
        pass

    def _reinitialize_states(self):
        self._initialize_model_from_xml()

    def _get_to_equilibrium(self):
        #set pressures to mean pressure and wait for arms to settle.
        set_joint_pressure_commands(self.model, self.data, "left", 0,
                                    [150] * 4)
        set_joint_pressure_commands(self.model, self.data, "left", 1,
                                    [150] * 4)
        set_joint_pressure_commands(self.model, self.data, "left", 2,
                                    [150] * 4)
        set_joint_pressure_commands(self.model, self.data, "right", 0,
                                    [150] * 4)
        set_joint_pressure_commands(self.model, self.data, "right", 1,
                                    [150] * 4)
        set_joint_pressure_commands(self.model, self.data, "right", 2,
                                    [150] * 4)

        initial_height = np.random.uniform(-800, 0)
        set_elevator_cmd(self.model, self.data, initial_height)

        mujoco.mj_step(self.model,
                       self.data,
                       nstep=int(15 / self.model.opt.timestep))

        self.data.time = 0

    def _initialize_model_from_xml(self):
        if self.first_load:
            print(f"Loading {os.path.basename(self.xml_path)} model.")
            self.first_load = False

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.vis.global_.offwidth = self.render_width
        self.model.vis.global_.offheight = self.render_height
        self.data = mujoco.MjData(self.model)

        #send in either camera_id or camera_name
        self.mujoco_renderer = MujocoRenderer(self.model,
                                              self.data,
                                              width=self.render_width,
                                              height=self.render_height,
                                              camera_name=self.camera_name,
                                              max_geom=100000)

        self._get_to_equilibrium()

    def step(self, action):
        commands = self.map_action_to_commands(action)

        assert len(
            commands
        ) == 25, "Commands must be a list of 25 floats: [elevator_height, left_j0, left_j1, left_j2, right_j0, right_j1, right_j2]"

        set_elevator_cmd(self.model, self.data, commands[0])
        for i in range(3):
            set_joint_pressure_commands(self.model, self.data, "left", i,
                                        commands[1 + (i * 4):5 + (i * 4)])
            set_joint_pressure_commands(self.model, self.data, "right", i,
                                        commands[13 + (i * 4):17 + (i * 4)])

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
        # print("resetting")
        super().reset(seed=seed)

        self._reinitialize_states()

        observation = self.get_observation_from_mujoco()
        info = {}

        return observation, info

    def render(self):
        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
