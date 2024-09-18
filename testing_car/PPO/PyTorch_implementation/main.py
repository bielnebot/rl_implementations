import argparse
import torch
from gymnasium.envs.classic_control.pendulum import PendulumEnv

# Custom modules
from environment import CustomEnv
from neural_networks import CustomNN
from ppo import PPO
from test_policy import test_policy


def train(gravity):
    print("Training")
    # Create environment
    env = PendulumEnv(render_mode=None)
    env = CustomEnv(env, forced_gravity=gravity)

    # Create PPO instance
    model = PPO(env)

    print("Training starts")
    model.learn(200_000_000)
    print("Training finished")


def test(gravity):
    # Create environment
    env = PendulumEnv(render_mode="human")
    env = CustomEnv(env, forced_gravity=gravity)

    # Extract policy input and output dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Create a policy and load the weights
    policy = CustomNN(obs_dim, act_dim)
    TIME_STEP_TO_TEST = 328000
    # TIME_STEP_TO_TEST = 779000
    policy.load_state_dict(torch.load(f"./checkpoints/PPO_PyTorch/g_{gravity}_{TIME_STEP_TO_TEST}.pth",weights_only=True))

    # Simulate an episode
    test_policy(env,policy)


def main(args):
    # Train or test
    if args.mode == "train":
        train(gravity=args.gravity)
    else:
        test(gravity=args.gravity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train or test")
    parser.add_argument("gravity", type=int, help="int from 0 to 20")
    args = parser.parse_args()

    main(args)