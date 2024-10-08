#this wrapper is meant to wrap a given baloo_vX. gym environment and make it compatiblae with the open-loop grasping baselines.
import gymnasium as gym


class OpenLoopBaselineWrapper(gym.Wrapper):
    """
    Now this class can just overrides the calculate_reward() method to return a force based reward. 
    """
    def __init__(self, env):
        """Constructor for the Reward wrapper."""
        # need to make sure that 
        super().__init__(env)

    def step(self, action):
        """Step function that calls the parent step function and then calculates the reward."""

        # call baloo_base step function
        observation, reward, terminated, truncated, info = self.env.step(
            action)

        return observation, reward, terminated, truncated, info

    def map_action_to_commands(self, action):
        '''
        open_loop_hugger.py will return actual mujcoco controls 
        since it's a state machine I wrote, so I need to override 
        baloo_vX map_action_to_commands() function.
        '''

        return action
