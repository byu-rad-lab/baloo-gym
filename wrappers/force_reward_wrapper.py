import gymnasium as gym
from baloo_mujoco_sim.utils.baloo_mj_api import get_contact_forces_on_body
import numpy as np
from scipy.stats import entropy


class GlobalContactForces:
    # class that organizes the 2d numpy array of contact forces on the box
    def __init__(self, contact_forces):
        self.contact_forces = contact_forces  # ncon x 3 array of contact forces

    def calc_net_force(self):
        # returns the net force on the box
        if self.contact_forces.size == 0:
            return np.zeros(3)
        else:
            return np.sum(self.contact_forces, axis=0)

    def get_num_contacts(self):
        # returns the number of contacts on the box
        return self.contact_forces.shape[0]

    def get_horizontal_force_directions(self):
        """
        +y
        ^
        |
        |-----> +x

        Theta is defined as angle from positive global x axis to the direction of the force (i.e. positive theta is a positive rotation about z)

        This is calculated using the arctan2(fy/fx) where fy and fx are the components of the force vector in the global y and x axes respectively.

        Examples:
        F = [fx = 1, fy = 1]. arctan2(fy,fx) = pi/4.
        F = [fx = -1, fy = 1]. arctan2(fy,fx) = 3*pi/4.
        F = [fx = -1, fy = -1]. arctan2(fy,fx) = -3*pi/4.
        F = [fx = 1, fy = -1]. arctan2(fy,fx) = -pi/4.

        Returns:
            thetas: a list of angles in radians from -pi to pi
        """
        if self.get_num_contacts() > 0:
            thetas = []
            for force in self.contact_forces:
                thetas.append(np.arctan2(force[1], force[0]))
            return thetas
        else:
            raise ValueError("No contacts on box")

    def get_force_direction_entropy(self, angle_bin_degrees=20):
        """Function that calculates the entropy of the distribution of force directions on the box. Theta is considered a random variable with some unknown distribution. By binning the force directions into angle_bin_degrees bins, we can calculate the entropy of the distribution of force directions on the box. The entropy is a metric that encourages uniform distributions and more contacts.

        Args:
            angle_bin_degrees (int, optional): Resolution of bins into which contact force angles will be grouped for probabilities. Defaults to 20.

        Returns:
            float: entropy of current theta distribution
        """
        thetas = self.get_horizontal_force_directions()
        counts, bins = np.histogram(
            thetas, bins=20)  # 20 bins = 360/20 = 18 degrees per bin
        probabilities = counts / counts.sum()
        return entropy(probabilities)


class ForceRewardWrapper(gym.Wrapper):
    """
    Now this class can just overrides the calculate_reward() method to return a force based reward. 
    """
    def __init__(self, env):
        """Constructor for the Reward wrapper."""
        super().__init__(env)

    def calculate_reward(self):
        #! wait, dont I want to just add things to the taxel reward already? Maybe I should just use gym.RewardWrapper
        """Calculates the reward to return."""
        contact_forces = get_contact_forces_on_body(self.model, self.data,
                                                    "box")
        global_contact_forces = GlobalContactForces(contact_forces)

        reward = -1
        if global_contact_forces.get_num_contacts() > 0:
            reward += global_contact_forces.get_force_direction_entropy(
                angle_bin_degrees=20)

        return reward
