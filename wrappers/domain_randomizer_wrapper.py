import numpy as np
from tabulate import tabulate
import gymnasium as gym
import numpy as np


class DomainRandomizer:

    distribution_fun_dict = {
        'uniform': np.random.uniform,
        'normal': np.random.normal,
        'choice': np.random.choice,
    }

    def __init__(self, seed=0):
        self.parameter_distributions = {
        }  # parameters name, distribution, and arguments
        self.sampled_parameters = {}  # parameters just sampled

        # reset seed
        np.random.seed(seed)

    def add(self, parameter_name, distribution, distribution_args):
        """
        Adds a parameter to randomize.
        @ parameter_name: str, name of the parameter
        @ distribution: ['uniform', 'normal', 'choice'], distribution of the parameter
        @ distribution_args: dict([arg=value, ...]), parameters of the chosen distribution (see numpy.random)
        """
        # check parameter types
        if type(parameter_name) != str:
            raise TypeError("Parameter \'name\' should be string.")
        if distribution not in DomainRandomizer.distribution_fun_dict.keys():
            raise KeyError(
                "Implemented \'distribution\'(s) are in DomainRandomizer.distribution_fun_dict.keys()"
            )
        if type(distribution_args) != dict:
            raise TypeError(
                "Distribution \'args\' should be provided as a dict.")

        # add parameter
        if parameter_name not in self.parameter_distributions.keys():
            self.parameter_distributions[parameter_name] = {}
            self.parameter_distributions[parameter_name][
                'distribution'] = DomainRandomizer.distribution_fun_dict[
                    distribution]
            self.parameter_distributions[parameter_name][
                'args'] = distribution_args
        else:
            raise Exception("Parameter %s already added to randomizer" %
                            parameter_name)

    def sample(self):
        """
        Samples parameters' values from their associated distributions.

        Returns: dict, sampled parameters
        """
        for key, item in self.parameter_distributions.items():
            self.sampled_parameters[key] = item['distribution'](
                **item['args'])  # unrol dictionary item['args']

        return self.sampled_parameters

    def summary(self):
        """
        Prints a summary of the parameters, distributions, and distribution arguments.
        """
        # table header
        table_rows = [['Parameter', 'Distribution', 'Arguments']]

        for param_name, item in self.parameter_distributions.items():
            # extract distribution name
            distribution_name = item['distribution'].__name__

            # create distribution arguments string
            arguments_string = ' '
            for arg_name, value in item['args'].items():
                arguments_string += arg_name + '=' + str(value) + ' '

            # append row
            table_rows.append(
                [param_name, distribution_name, arguments_string])

        # tabulate table rows
        table = tabulate(tabular_data=table_rows,
                         headers='firstrow',
                         tablefmt='fancy_grid')

        print(table)


class DomainRandomizerWrapper(gym.Wrapper):
    """
    This class is a wrapper that randomizes the domain of the environment on reset. 


    """
    def __init__(self, env):
        super().__init__(env)

        #what parameters to randomize?
        self.domain_randomizer = DomainRandomizer(seed=42)
        # self.domain_randomizer.add('taxel_friction', 'distribution', {'arg': 'value'})
        # self.domain_randomizer.add('pressure_time_constant', 'distribution', {'arg': 'value'})
        # self.domain_randomizer.add('joint_stiffness', 'distribution', {'arg': 'value'})
        # self.domain_randomizer.add('joint_damping', 'distribution', {'arg': 'value'})
        # self.domain_randomizer.add('manipuland_type', 'distribution', {'arg': 'value'})
        self.domain_randomizer.add('manipuland_height', 'distribution',
                                   {'arg': 'value'})
        self.domain_randomizer.add('manipuland_width', 'distribution',
                                   {'arg': 'value'})
        self.domain_randomizer.add('manipuland_length', 'distribution',
                                   {'arg': 'value'})
        self.domain_randomizer.add('manipuland_mass', 'distribution',
                                   {'arg': 'value'})

    def reset(self):
        #this will reload the xml file and reset the environment. This gets to equilibrium as well.
        super().reset(seed=None, options=None)

        # use domain randomizer to change certain parameters dynamically through self.model
        randomized_params = self.domain_randomizer.sample()

        # geometry based changes
        # self.env.unwrapped.model.geom('box').friction = []
        self.env.unwrapped.model.body(
            'box').mass = randomized_params['manipuland_mass']
        new_size = [
            randomized_params['manipuland_height'],
            randomized_params['manipuland_width'],
            randomized_params['manipuland_length']
        ]
        self.env.unwrapped.model.geom('box').size = new_size

        # apply the changes to the model body
        old_pos = self.env.unwrapped.model.body('box').pos
        new_pos = [
            old_pos[0], old_pos[1], randomized_params['manipuland_height'] / 2
        ]
        self.env.unwrapped.model.body('box').pos = new_pos

        # self.env.unwrapped.model.body('box').quat = []
