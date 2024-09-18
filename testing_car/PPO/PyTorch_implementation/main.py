import argparse
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

    # Create PPO instance
    model = PPO(env)

    print("Training starts")
    model.learn(200_000_000)
    print("Training finished")


def test():
    # Create environment
    env = gym.make('CarRacing-v0')
    env = CustomEnv(env)

    # Extract policy input and output dimensions
    # obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Create a policy and load the weights
    policy = CustomNN(act_dim)
    TIME_STEP_TO_TEST = 16000
    policy.load_state_dict(torch.load(f"./car_checkpoints/PPO_PyTorch/g_{TIME_STEP_TO_TEST}.pth",weights_only=True))

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