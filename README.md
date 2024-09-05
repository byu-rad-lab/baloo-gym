# README #

This repo is an RL training environment for Baloo.

## Getting Started ##

Before you can run the application, you need to set up a virtual environment and install the necessary dependencies. Here's how you can do that:

1. First, create a virtual environment. You can do this using the following command:
   ```bash
   python3 -m venv env
   ```
2. Next, activate the virtual environment:
   ```bash
   source env/bin/activate
   ```
3. Install [baloo_mujoco_sim](https://github.com/byu-rad-lab/baloo_mujoco_sim) repository.
4. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```


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
   1. A good place to start for new environments is the baloo_vTEMPLATE.py class you can copy and use as template. It inherites from ```BalooBase``` and shows the necessary functions to create a new environment.
   2. Decide on action_space and implement ```map_action_to_pressure_commands()```.
   3. Decide on observation_space and implement ```get_observation_from_mujoco()```
   4. Design reward function and implement ```calculate_reward()```. The ```calculate_reward()``` function can be overridden with a wrapper since this will likely change frequently.
   5. If you have other things that you want to implement to change the environment, you can implement a wrapper in the ```wrappers``` folder. The wrapper simply needs to override the ```step()``` function to do its own custom logic and then call the parent class's ```step()``` function.






 