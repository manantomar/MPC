### About

This work aims at implementing simple MPC controller for gym's Mujoco models as described in **Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning** 
and build on it by adding LQR based controllers instead of using simple shooting methods. Such controllers are then applied in parallel and the stored trajectories are used to learn a general 
neural network policy.


### Dependencies

This code has been tested on python3 and requires [mujoco_py](https://github.com/openai/mujoco-py) installed.

### How to Run

Please use `python3 main.py` to run. Passing `--load_model` will restore the previously stored policy parameters.

