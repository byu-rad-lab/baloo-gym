# README #

This repo is an RL training environment for Baloo.

#### Setting up a Virtual Environment and Installing Dependencies ####

Before you can run the application, you need to set up a virtual environment and install the necessary dependencies. Here's how you can do that:

1. First, create a virtual environment. You can do this using the following command:

```bash
python3 -m venv env
```

2. Next, activate the virtual environment:

```bash
source env/bin/activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```


#### GPU Configuration ####

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


#### Assets ####

This code needs an xml file to represent the mujoco model. A static copy of the Baloo xml model file is included in this repo, but you can also generate different versions in the [baloo_mujoco_sim repo](https://bitbucket.org/byu_rad_lab/baloo_mujoco_sim/src/master/). This repo has more details on how to make changes to the model and generate the xml file.

Note that the model uses some [Mujoco Plugins](https://mujoco.readthedocs.io/en/stable/programming/extension.html), so those will need to be built as well. See the baloo_mujoco_sim repo for more details on how to do that. This means that you will need to activate the virtual environment and install the plugin to the virtual environment mujoco installation via MUJOCO_ROOT_DIR variable. 