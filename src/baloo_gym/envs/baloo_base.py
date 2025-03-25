import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod
import mujoco
from baloo_gym.utils.mujoco_rendering import MujocoRenderer
import baloo_mujoco_sim as baloo_mj
from baloo_mujoco_sim.utils.baloo_mj_api import (
    set_elevator_cmd,
    set_joint_pressure_commands,
    set_box_position,
    set_box_size,
    set_box_mass,
    set_box_quat,
)
import os
import numpy as np
import random
from itertools import product
from scipy.stats import qmc
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


@dataclass
class Object:
    name: str
    size: np.ndarray
    mass: float
    color: np.ndarray


def sample_lhs(seed):
    # Define the parameter ranges (5 points each)
    xsize = np.linspace(0.2, 0.6, 5)
    ysize = np.linspace(0.2, 0.6, 5)
    zsize = np.linspace(0.5, 1.25, 5)
    mass = np.linspace(.5, 10, 5)

    # Create the Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=4,
                                 seed=seed)  # 4 dimensions (x, y, z, mass)

    # Generate 5 samples (one for each range)
    num_samples = 1000
    lhs_samples = sampler.random(n=num_samples)

    # Scale the LHS samples to the respective parameter ranges
    x_samples = np.interp(lhs_samples[:, 0], [0, 1],
                          [xsize.min(), xsize.max()])
    y_samples = np.interp(lhs_samples[:, 1], [0, 1],
                          [ysize.min(), ysize.max()])
    z_samples = np.interp(lhs_samples[:, 2], [0, 1],
                          [zsize.min(), zsize.max()])
    mass_samples = np.interp(lhs_samples[:, 3], [0, 1],
                             [mass.min(), mass.max()])

    # Combine the scaled samples into a list of tuples
    sampled_points = [
        (x, y, z, m)
        for x, y, z, m in zip(x_samples, y_samples, z_samples, mass_samples)
    ]

    return sampled_points


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
        randomize_initial_height=False,
        randomize_object_size=False,
        randomize_object_mass=False,
        object_size=None,
        object_mass=None,
        randomize_object_quat=False,
        randomize_object_pos=False,
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

        self.object_combinations = sample_lhs(0)

        self.first_load = True
        self.randomize_initial_height = randomize_initial_height
        self.randomize_object_size = randomize_object_size
        self.randomize_object_mass = randomize_object_mass
        self.randomize_object_quat = randomize_object_quat
        self.randomize_object_pos = randomize_object_pos

        self.object_size = object_size
        self.object_mass = object_mass
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

        if self.randomize_initial_height:
            initial_height_cmd = np.random.randint(-900, 0)
            set_elevator_cmd(self.model, self.data, initial_height_cmd)

        mujoco.mj_step(self.model,
                       self.data,
                       nstep=int(30 / self.model.opt.timestep))

        self.data.time = 0

    def _initialize_model_from_xml(self):
        """
        ! need to load model via from_string(). from_file() is not working in 3.2.3.
        ! (see https://github.com/google-deepmind/mujoco/issues/2142)
        ! but need to upgrade python > 3.8 to upgrade past 3.2.3. 
        """
        if self.first_load:
            self.first_load = False

            # load xml file into string
            with open(baloo_mj.XML_PATH, 'r') as file:
                xml_string = file.read()

            self.mjspec = mujoco.MjSpec()
            self.mjspec.from_string(xml_string)
            # print(f"Loading {os.path.basename(self.xml_path)} model.")
            self.model = self.mjspec.compile()
            self.data = mujoco.MjData(self.model)

        # change whatever you want in spec before compiling model for use during episode
        if self.randomize_object_size:
            self.object_attr = random.choice(self.object_combinations)
            # xsize, ysize = np.random.uniform(0.2, 0.6, 2)
            # zsize = np.random.uniform(0.5, 1.25)
            xsize, ysize, zsize = self.object_attr[0:3]

            set_box_size(self.mjspec, xsize, ysize, zsize)
        else:
            if self.object_size is not None:
                xsize, ysize, zsize = self.object_size
                set_box_size(self.mjspec, xsize, ysize, zsize)

        if self.randomize_object_mass:
            if not self.randomize_object_size:
                raise ValueError(
                    "randomize_object_mass without size is not yet supported.")

            # mass = self.object.mass
            # mass = np.random.uniform(5, 20)
            set_box_mass(self.mjspec, self.object_attr[3])

        else:
            if self.object_mass is not None:
                mass = self.object_mass
                set_box_mass(self.mjspec, mass)

        #recompile model and reset data for episode.
        self.model = self.mjspec.compile()
        self.model.vis.global_.offwidth = self.render_width
        self.model.vis.global_.offheight = self.render_height
        mujoco.mj_resetData(self.model, self.data)

        #### SET SIZES, POSITIONS, AND ORIENTATIONS ####
        if self.randomize_object_pos:
            #set random x and y position for box, height is set by box size
            world2chest_front = 35e-2

            #ensure that the box (even rotated) is not underneath the chest
            offset = np.sqrt((ysize / 2)**2 +
                             (xsize / 2)**2) + world2chest_front

            distance_from_chest = np.random.uniform(0, 15e-2)
            y_box = offset + distance_from_chest
            x_box = np.random.uniform(-0.15, 0.15)
        else:
            distance_from_chest = 40e-2
            y_box = ysize / 2 + distance_from_chest
            x_box = 0

        if self.randomize_object_size:
            #place box on the ground, according to new size.
            z_box = zsize / 2

        else:
            if self.object_size is not None:
                xsize, ysize, zsize = self.object_size
                z_box = zsize / 2

        set_box_position(self.model, self.data, x_box, y_box, z_box)

        if self.randomize_object_quat:
            #get random rotation about z axis
            random_rotation = np.random.uniform(-np.pi / 6, np.pi / 6)
            # print(f"random rotation: {random_rotation}")
            rz = R.from_euler('z', random_rotation, degrees=False)
            quat = np.roll(rz.as_quat(), 1)
            #set box orientation
            set_box_quat(self.model,
                         self.data,
                         qw=quat[0],
                         qx=quat[1],
                         qy=quat[2],
                         qz=quat[3])

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
