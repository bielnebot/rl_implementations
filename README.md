# Reinforcement learning implementations

This repository contains the implementation of two RL algorithms: Proximal Policy Optimization and Deep Q Learning both from scratch (with [PyTorch](https://pytorch.org/)) and using the [TorchRL](https://pytorch.org/rl/stable/index.html) library.

The environment is the car racing simulation from OpenAI's Gym package.

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
### To train a policy
### To test a pre-trained policy





## To-Do
- [ ] PPO with TorchRL
- [ ] DQN with TorchRL
- [ ] DQN from scratch

## Done
- [x] PPO from scratch