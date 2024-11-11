import gymnasium as gym
from domain_randomization.methods.curriculum_randomizer import CurriculumDomainRandomizer
from baloo_mujoco_sim.utils.baloo_mj_api import (set_mocap_size,
                                                 set_box_position,
                                                 get_box_position)
from typing import Literal, Union, List

import numpy as np
import mujoco


class CurriculumEnv(gym.Wrapper):

    def __init__(
        self, env,
        curriculum_selection: List[Literal["manipuland_initial_position",
                                           "perturbations", "object_mass"]]):
        super().__init__(env)

        self.curriculum_selection = curriculum_selection
        print("Curriculum Selection: ", self.curriculum_selection)

        if "manipuland_initial_position" in self.curriculum_selection:
            self.position_randomizer = CurriculumDomainRandomizer(
                identifier="curriculum_randomizer", ramp_steps=100)

            self.position_randomizer.add("initial_xpos", "uniform",
                                         dict(low=-0.5, high=0.5))
            self.position_randomizer.add("initial_ypos", "uniform",
                                         dict(low=0.5, high=1))
            self.position_randomizer.init_history()

            # print(self.position_randomizer.summary())

        if "perturbations" in self.curriculum_selection:
            self.perturbation_randomizer = CurriculumDomainRandomizer(
                identifier="curriculum_randomizer", ramp_steps=100)

            self.perturbation_randomizer.add("perturbation_magnitude",
                                             "uniform", dict(low=0, high=0.5))
            self.perturbation_randomizer.init_history()

        if "object_mass" in self.curriculum_selection:
            self.object_mass_randomizer = CurriculumDomainRandomizer(
                identifier="curriculum_randomizer", ramp_steps=100)

            self.object_mass_randomizer.add("object_mass", "uniform",
                                            dict(low=0.1, high=1))
            self.object_mass_randomizer.init_history()

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed, options)
        self._adjust_difficulty()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def _adjust_difficulty(self):
        #every time I sample, the next one gets harder.
        if "manipuland_initial_position" in self.curriculum_selection:
            box_position = get_box_position(self.unwrapped.model,
                                            self.unwrapped.data)

            params = self.position_randomizer.sample()

            new_box_pos = np.array([
                params["initial_xpos"], params["initial_ypos"], box_position[2]
            ])
            set_box_position(self.unwrapped.model, self.unwrapped.data,
                             new_box_pos)

        if "perturbations" in self.curriculum_selection:
            raise NotImplementedError

        if "object_mass" in self.curriculum_selection:
            raise NotImplementedError

    def render(self):
        super().render()  #to initialize viewer.
        if "perturbations" in self.curriculum_selection:
            if not self.perturbation_visual_initialized:
                self.perturbation_visual_initialized == True

                #this will add the marker the next time the scene is rendered.
                # everything in the marker list will be re-rendered each time.
                # so to change postion, just change the position of the marker in the list.
                #this is a bit hacky, since the renderer doesn't really expose this functionality.
                self.arrowid = self.unwrapped.num_visual_geoms

                self.unwrapped.mujoco_renderer.viewer.add_marker(
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=np.array([0.1, 0.1, 0]),
                    pos=np.array([0, 1, 0]),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0, 1, 0, 1]),
                )
                self.unwrapped.num_visual_geoms += 1
                return super().render()

            box_pos = self.unwrapped.data.body('box').xipos

            applied_force = np.random.uniform(-1, 1, 3)
            mag_app_force = np.linalg.norm(applied_force)
            print("Applying force: ", applied_force)
            self.unwrapped.mujoco_renderer.viewer._markers[self.arrowid][
                'pos'] = box_pos  #force is always applied on box COM
            self.unwrapped.mujoco_renderer.viewer._markers[
                self.arrowid]['rgba'] = np.array([0, 1, 0, 1])
            self.unwrapped.mujoco_renderer.viewer._markers[
                self.arrowid]['size'] = np.array([0.1, 0.1, mag_app_force
                                                  ])  #changes with magnitude

            #reset the orientation of the arrow
            u = np.cross(np.array([0, 0, 1]), applied_force)
            theta = np.arccos(
                np.dot(np.array([0, 0, 1]), applied_force) /
                (np.linalg.norm(applied_force) + 1e-6))

            skew = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]],
                             [-u[1], u[0], 0]])

            R = np.eye(
                3) + np.sin(theta) * skew + (1 - np.cos(theta)) * skew @ skew

            self.unwrapped.mujoco_renderer.viewer._markers[
                self.arrowid]['mat'] = R.flatten()

        return super().render()
