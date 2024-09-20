import argparse
import os

import torch
import gym

# Custom modules
from environment import CustomEnv
from neural_networks import CustomNN
from ppo import PPO
from test_policy import test_policy


def train():
    print("Training")
    # Create environment
    env = gym.make('CarRacing-v0')
    env = CustomEnv(env)

    # Check if there are existing checkpoints to continue training
    checkpoint = None
    existing_checkpoints = os.listdir("./checkpoints/PPO_PyTorch/")
    if len(existing_checkpoints) > 0:
        # Find the last one
        checkpoints_numbers = [int(i.split("_")[1][:-4]) for i in existing_checkpoints]
        last_checkpoint = max(checkpoints_numbers)
        # Load it
        checkpoint = torch.load(f"./checkpoints/PPO_PyTorch/checkpoint_{last_checkpoint}.pth")
        print(f"Last checkpoint found: {last_checkpoint}")

    # Create PPO instance
    model = PPO(env, checkpoint)

    print("Training starts")
    model.learn(10_005)
    print("Training finished")


def test():
    # Create environment
    env = gym.make("CarRacing-v0")
    env = CustomEnv(env)

    # Extract policy input and output dimensions
    act_dim = env.action_space.shape[0]

    # Create a policy and load the weights
    policy = CustomNN(act_dim)
    TIME_STEP_TO_TEST = 80000
    policy.load_state_dict(torch.load(f"./checkpoints/PPO_PyTorch/g_{TIME_STEP_TO_TEST}.pth",weights_only=True))

    # Simulate an episode
    test_policy(env,policy)


def main(args):
    # Train or test
    if args.mode == "train":
        train()
    else:
        test()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("mode", type=str, help="train or test")
    # args = parser.parse_args()
    #
    # main(args)
    train()
    # test()