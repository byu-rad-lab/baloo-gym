import numpy as np
from tabulate import tabulate

from domain_randomization.methods.vanilla_randomizer import VanillaDomainRandomizer


class CurriculumDomainRandomizer(VanillaDomainRandomizer):

    distribution_fun_dict = {
        'uniform': np.random.uniform,
        'normal': np.random.normal,
        'choice': np.random.choice,
        }

    def __init__(self, identifier, ramp_steps=100, seed=None):
        super().__init__(identifier, seed)

        self.parameter_distributions = {}   # parameters name, distribution, and arguments
        self.sampled_parameters = {}        # parameters just sampled

        self.step_counter = 0 # incremented each time a sample is generated
        self.ramp_steps = ramp_steps

        self.dr_history = None # keeps track of all sampled parameters # TODO: use callbacks?

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
        if distribution not in CurriculumDomainRandomizer.distribution_fun_dict.keys():
            raise KeyError("Implemented \'distribution\'(s) are in DomainRandomizer.distribution_fun_dict.keys()")
        if type(distribution_args) != dict:
            raise TypeError("Distribution \'args\' should be provided as a dict.")
        
        # add parameter
        if parameter_name not in self.parameter_distributions.keys():
            self.parameter_distributions[parameter_name] = {}
            self.parameter_distributions[parameter_name]['distribution'] = CurriculumDomainRandomizer.distribution_fun_dict[distribution]
            self.parameter_distributions[parameter_name]['args'] = distribution_args
        else:
            raise Exception("Parameter %s already added to randomizer" % parameter_name)

    def sample(self):
        """
        Samples parameters' values from their associated distributions.

        Returns: dict, sampled parameters
        """
        # sample
        for key, item in self.parameter_distributions.items():
            # correct distribution parameters according to curriculum
            curriculum_args = self.__modify_distribution_arguments(item)

            self.sampled_parameters[key] = item['distribution'](**curriculum_args) # unrol dictionary item['args']

        # update variables
        self.__update_history()
        self.step_counter += 1
        
        return self.sampled_parameters

    def __modify_distribution_arguments(self, item):
        c = self.__compute_curriculum_factor()
        curriculum_args = {} #item_c = item.copy()
            
        if item['distribution'].__name__ == 'normal':
            curriculum_args['loc'] = item['args']['loc']
            curriculum_args['scale'] = item['args']['scale'] * c
        elif item['distribution'].__name__ == 'uniform':
            low = item['args']['low']
            high = item['args']['high']
            mid_value = (low + high) / 2.0
            half_range = (high - low) / 2.0

            curriculum_args['low'] = mid_value - c * half_range
            curriculum_args['high'] = mid_value + c * half_range

        return curriculum_args

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
            table_rows.append([param_name, distribution_name, arguments_string])
        
        # tabulate table rows
        table = tabulate(tabular_data=table_rows,  headers='firstrow', tablefmt='fancy_grid')

        print(table)

    def get_parameters_name(self):
        return self.parameter_distributions.keys()
    
    def init_history(self):
        self.dr_history = {name : list() for name in self.get_parameters_name()}

    def __update_history(self):
        parameter_names = self.get_parameters_name()
        for name in parameter_names:
            self.dr_history[name].append(self.sampled_parameters[name])
    

    def __compute_curriculum_factor(self):
        """
        c goes from 0.0 to 1.0 indicating the level of curriculum reached
        """
        c = min(self.step_counter / float(self.ramp_steps), 1.0)

        return c
    

