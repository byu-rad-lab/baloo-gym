import gymnasium as gym
from baloo_mujoco_sim.utils.baloo_mj_api import detect_box_on_ground, apply_wrench_to_body
import numpy as np
import mujoco

import numpy as np
from scipy.spatial.transform import Rotation as R
import threading


def rotation_matrix_from_vector_to_vector(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    if np.linalg.norm(cross) < 1e-8:
        if dot > 0.999999:
            return np.eye(3)  # Same direction
        else:
            # 180 deg rotation: pick any orthogonal axis
            axis = np.array([1, 0, 0]) if abs(v1[0]) < 0.99 else np.array(
                [0, 1, 0])
            orthogonal = np.cross(v1, axis)
            orthogonal /= np.linalg.norm(orthogonal)
            return R.from_rotvec(np.pi * orthogonal).as_matrix()
    else:
        axis = cross / np.linalg.norm(cross)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        return R.from_rotvec(angle * axis).as_matrix()


class ObjectPerturbationWrapper(gym.Wrapper):
    """
    A wrapper for the environment that applies perturbations to objects.
    """

    def __init__(self, env):
        super().__init__(env)
        self.perturbation_active: bool = False
        self.perturbation_duration: int = int(1 /
                                              self.unwrapped.control_timestep)
        self.perturbation_magnitude: float = 100.0
        self.step_counter: int = 0
        self.arrow_initialized: bool = False
        self.force_direction = np.array([0, 0, 1])
        self.marker_lock = threading.Lock()

    def step(self, action):
        """
        Take a step in the environment and apply perturbations.
        """
        # Apply perturbations on the object once it is off the ground
        force = np.zeros(3)
        if not detect_box_on_ground(self.unwrapped.model, self.unwrapped.data):
            if self.perturbation_active:
                if self.step_counter == 0:
                    self.force_direction = np.random.uniform(-1, 1, 3)
                    self.force_direction /= np.linalg.norm(
                        self.force_direction)  # Normalize the direction
                    force = self.force_direction * self.perturbation_magnitude

            apply_wrench_to_body(self.unwrapped.model,
                                 self.unwrapped.data,
                                 "box",
                                 force=force,
                                 torque=np.zeros(3))
            self.step_counter += 1

        with self.marker_lock:
            #set position of arrow
            box_pos = self.unwrapped.data.body('box').xipos
            self.unwrapped.mujoco_renderer.viewer._markers[
                self.arrowid]['pos'] = box_pos

            #orient arrow along force application
            v1 = np.array([0, 0, 1])
            R = rotation_matrix_from_vector_to_vector(v1, self.force_direction)
            self.unwrapped.mujoco_renderer.viewer._markers[
                self.arrowid]['mat'] = R.flatten()

            #show arrow or not
            self.unwrapped.mujoco_renderer.viewer._markers[
                self.arrowid]['rgba'] = np.array(
                    [1, 0, 0, self.perturbation_active])

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.step_counter == self.perturbation_duration:
            #switch from active to not, or vice versa
            self.perturbation_active = not self.perturbation_active
            self.step_counter = 0

        return obs, reward, terminated, truncated, info

    def render(self):
        if not self.arrow_initialized:
            super().render()  #to initialize viewer.
            self.arrow_initialized = True

            #this will add the marker the next time the scene is rendered.
            # everything in the marker list will be re-rendered each time.
            # so to change postion, just change the position of the marker in the list.
            #this is a bit hacky, since the renderer doesn't really expose this functionality.
            self.arrowid = len(self.unwrapped.mujoco_renderer.viewer._markers)
            self.unwrapped.mujoco_renderer.viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=[0.05, .05, 2.0],
                pos=np.array([1, 0, 0]),
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 0]),
            )

        return super().render()
