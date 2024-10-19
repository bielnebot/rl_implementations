import matplotlib.pyplot as plt


def test_single_episode(env,policy):

    rewards = []

    obs = env.reset()
    done = False
    t = 0

    for _ in range(10000):
        env.render()
        t += 1

        # Plot
        # plt.imshow(obs[0])
        # plt.colorbar()
        # plt.show()

        action = policy(obs).detach().numpy()[0]
        print("action=",action)
        obs, rew, done, _ = env.step(action)

        rewards.append(rew)

        # if done:
        #     break

    print(f"episode reward: {sum(rewards)}")
    return rewards


def test_policy(env,policy):
    for _ in range(1):
        test_single_episode(env,policy)


if __name__ == "__main__":
    import os
    import torch
    import gym
    from environment import CustomEnv
    from neural_networks import CustomNN

    def test():
        # Create environment
        env = gym.make("CarRacing-v0")
        env = CustomEnv(env)

        # Extract policy input and output dimensions
        act_dim = env.action_space.shape[0]

        # Load last checkpoint
        existing_checkpoints = os.listdir("./checkpoints/PPO_PyTorch/")
        checkpoints_numbers = [int(i.split("_")[1][:-4]) for i in existing_checkpoints]
        last_checkpoint = max(checkpoints_numbers)
        checkpoint = torch.load(f"./checkpoints/PPO_PyTorch/checkpoint_{last_checkpoint}.pth")
        # checkpoint = torch.load(f"./checkpoints/pre_trained_policies/checkpoint_185000.pth")
        print(f"Last checkpoint found: {last_checkpoint}")

        # Create a policy and update the weights
        policy = CustomNN(act_dim)
        policy.load_state_dict(checkpoint["policy_state_dict"])

        # Simulate an episode
        test_policy(env, policy)

    test()