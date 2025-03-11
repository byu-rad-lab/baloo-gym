#this wrapper is meant to wrap a given baloo_vX. gym environment and make it compatiblae with the open-loop grasping baselines.
import gymnasium as gym


class OpenLoopBaselineWrapper(gym.Wrapper):
    """
    Now this class can just overrides the calculate_reward() method to return a force based reward. 
    """
    def __init__(self, env):
        super().__init__(env)

        #manually override the child class implementation of map_action_to_commands(). This is a bit hacky, but alternative is a big refactor.
        self.unwrapped.map_action_to_commands = self.modified_map_action_to_commands  #this doesn't work

    def step(self, action):
        """Step function that calls the parent step function and then calculates the reward."""

        # call baloo_base step function, but can't because it calls other env functions that I can't change here.
        observation, reward, terminated, truncated, info = self.env.step(
            action)

        return observation, reward, terminated, truncated, info

    def modified_map_action_to_commands(self, action):
        '''
        open_loop_hugger.py will return actual mujcoco controls 
        since it's a state machine I wrote, so I need to override 
        baloo_vX map_action_to_commands() function.
        '''
        return action
