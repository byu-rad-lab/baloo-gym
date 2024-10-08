import gymnasium as gym
from baloo_mujoco_sim.utils.baloo_mj_api import (
    set_mocap_size, )


class CurriculumEnv(gym.Wrapper):
    def __init__(self, env, initial_difficulty=0):
        super().__init__(env)
        self.difficulty = initial_difficulty
        self.sphere_radius = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed, options)
        self.adjust_difficulty()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def adjust_difficulty(self):
        
        self.sphere_radius = .5  #for now

        #update mujoco model to show sphere. this isn't working rn. the sphere isn't
        set_mocap_size(self.env.unwrapped.model, self.env.unwrapped.data,
                       "object_force_field", [self.sphere_radius])
