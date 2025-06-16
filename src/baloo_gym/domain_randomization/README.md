# Domain Randomization
Reinforcement learning policies trained only in simulation are likely to face a performance drop when transferred to the physical world due to the approximations made by simulated environments. This phenomenon is known as the _sim-to-real_ gap. 

Domain randomization is a _sim-to-real_ technique to mitigate the gap between simulation and reality, which randomizes the simulation parameters every training episode so that the policy experiences a variety of simulation scenarios.

# Installation
- Clone the repository
- Install [poetry](https://python-poetry.org/docs/)
- Create a virtual environment
```bash
python3 -m venv venv
```
- Install the requirements
```bash
poetry install
```


## How to use it
- Check the short tutorial in the `test/notebooks/` directory
