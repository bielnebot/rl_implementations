# Reinforcement learning implementations

This repository contains the implementation of two RL algorithms: Proximal Policy Optimization and Deep Q Learning both from scratch (with [PyTorch](https://pytorch.org/)) and using the [TorchRL](https://pytorch.org/rl/stable/index.html) library.

The environment is the car racing simulation from OpenAI's Gym package to train autonomous driving.

![trained_agent_demo](docs/PPO_example.gif)
Example of an agent trained with the PPO algorithm.

## Set up
Create a virtual environment:
```console
conda create -n NAME_OF_THE_ENVIRONEMNT python=3.8
conda activate NAME_OF_THE_ENVIRONEMNT
```
Clone the repository:
```console
git clone https://github.com/bielnebot/rl_implementations.git
```
And install the requirements:
```console
pip install -r requirements.txt
```

## Use

| Algorithms | Implementation   |
|------------|------------------|
| PPO, DQN   | PyTorch, TorchRL |
### To train a policy
Choose an algorithm and an implementation from the available ones and run the `main.py` module of its respective directory.
```console
python CHOSEN_ALGORITHM\CHOSEN_IMPLEMENTATION\main.py
```
### To test a pre-trained policy
Choose an algorithm and an implementation from the available ones and run the `test_policy.py` module of its respective directory.
```console
python CHOSEN_ALGORITHM\CHOSEN_IMPLEMENTATION\test_policy.py
```





## To-Do

- [ ] DQN with TorchRL
- [ ] DQN from scratch

## Done
- [x] PPO from scratch
- [x] PPO with TorchRL