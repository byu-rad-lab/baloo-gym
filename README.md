# README #

This repo is an RL training environment for Baloo.

## Getting Started ##

Before you can run the application, you need to set up a virtual environment and install the necessary dependencies. Here's how you can do that:

1. Install [uv](https://docs.astral.sh/uv/) which I use a dependency manager for this project. 
2. Clone this repository, making sure to include the `--recurse-submodules` flag to include submodules. 
3. Install the [baloo_mujoco_sim](https://github.com/byu-rad-lab/baloo_mujoco_sim) repository. It should be cloned in the same directory as this repository.
4. Install this project's dependencies into a virtual environment by running the following command in the root directory of the project.
``` bash
uv sync
```
5. Activate the virtual environment by running the following command:
``` bash
source .venv/bin/activate
```
6. You can now start training.


### GPU Configuration ###
The virtual environment dependencies have been tested with this configuration:
``` bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080        Off |   00000000:0B:00.0  On |                  N/A |
|  0%   41C    P8             28W /  320W |     694MiB /  10240MiB |     30%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```
## Structure ##

The Baloo Base class takes care of all of the mujoco stuff and gym stuff. Users need to inherit from this and implement their own observations, rewards, and action spaces. 

Then the appropriate functions need to be implemented to map the actions to the mujoco model, which accepts pressure commands as inputs. 

WORKFLOW to implement a new environment:
   1. A good place to start for new environments is the baloo_vTEMPLATE.py class you can copy and use as template. It inherites from [```BalooBase```](./envs/baloo_base.py) and shows the necessary functions to create a new environment.
   2. Decide on action_space and implement ```map_action_to_pressure_commands()```.
   3. Decide on observation_space and implement ```get_observation_from_mujoco()```
   4. Design reward function and implement ```calculate_reward()```. The ```calculate_reward()``` function can be overridden with a wrapper since this will likely change frequently.
   5. If you have other things that you want to implement to change the environment, you can implement a wrapper in the [```wrappers```](./wrappers/) folder. The wrapper simply needs to override the ```step()``` function to do its own custom logic and then call the parent class's ```step()``` function.

## Observation Spaces ##
I've implemented various observation spaces that can be used in the Baloo environment. They are located in the [```observation_spaces```](./utils/observation_spaces.py) module. 

## Action Spaces ##
These are implemented in the [```action_spaces```](./utils/action_spaces.py) module. 

## Environments ##
| **Environment**                      | **Observation Space**                                           | **Action Space**                                     |
| ------------------------------------ | --------------------------------------------------------------- | ---------------------------------------------------- |
| [```baloo_v0```](./envs/baloo_v0.py) | [```StateObservation```](./utils/observation_spaces.py)         | [```NormalizedAction```](./utils/action_spaces.py)   |
| [```baloo_v1```](./envs/baloo_v1.py) | [```StateObservation```](./utils/observation_spaces.py)         | [```IncrementalAction```](./utils/action_spaces.py)  |
| [```baloo_v2```](./envs/baloo_v2.py) | [```StateObservation```](./utils/observation_spaces.py)         | [```IncrementalTorques```](./utils/action_spaces.py) |
| [```baloo_v3```](./envs/baloo_v3.py) | [```RelativeObservation```](./utils/observation_spaces.py)      | [```IncrementalTorques```](./utils/action_spaces.py) |
| [```baloo_v4```](./envs/baloo_v4.py) | [```StateObservationPressure```](./utils/observation_spaces.py) | [```IncrementalTorques```](./utils/action_spaces.py) |
| [```baloo_v5```](./envs/baloo_v5.py) | [```StateObservationPressure```](./utils/observation_spaces.py) | [```NormalizedAction```](./utils/action_spaces.py)   |



## Future Work ##
* Right now, all of the model changes (like for visualization) have to happen in baloo-mujoco-sim. With an upgrade on the version of Mujoco, there is a new model editing API which allows you to make dynamic changes to the model. This would be a good feature to use here to add manipulands and other visualization objects without the need to change the baloo-mujoco-sim code.






 