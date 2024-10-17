import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import Wrapper
from gym.spaces import Box
from gym.spaces.discrete import Discrete


def observation_transformation(image):
    """
    Transforms the default observation into a simpler one
    :param image: the original observation of the environment. A 3D numpy array
    :return: a 2D numpy array with unified green and cropping at the bottom and shape (1, 84, 96)
    """
    # Extract green channel
    image = image[:, :, 1]
    # Crop the bottom of the image
    image = image[:84,:]
    # Set green tonalities equal
    image[image == 204] = 255
    image[image == 230] = 255
    # Normalise
    image = image / 255
    # Reshape to (96,96)
    image = np.reshape(image, (1,) + image.shape)
    return image


class CustomEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = Box(low=0.0, high=1.0, shape=(1, 84, 96))

        self.available_actions = {0: np.array([0.0, 0.0, 0.0]), # do nothing
                                  1: np.array([0.0, 0.5, 0.0]),  # forward
                                  2: np.array([1.0, 0.0, 0.06]),  # right
                                  3: np.array([-1.0, 0.0, 0.06]),  # left
                                  4: np.array([0.0, 0.0, 0.5])}  # break

        self.action_space = Discrete(len(self.available_actions))

    def step(self, action):
        # print("recieved action =",action, type(action))
        action = self.available_actions[action]
        # print("updated action=",action,type(action), action.shape)
        # action = action[0]

        # Rescale gas and brake from [-1,1] to [0,1]
        rescaled_action = np.zeros(3)
        rescaled_action[0] = action[0]
        rescaled_action[1:] = (action[1:] + 1) / 2

        # Step
        observation, reward, done, info = super().step(rescaled_action)
        observation = observation_transformation(observation)

        return observation, reward, done, info

    def reset(self):

        observation = super().reset()
        observation = observation_transformation(observation)

        return observation


def run_episode(policy=None):
    env = gym.make("CarRacing-v0")
    env = CustomEnv(env)

    observation = env.reset()

    for _ in range(1000):
        # Reshape observation to (batch, channel, height, width)
        observation = np.reshape(observation, (1,) + observation.shape)
        env.render()
        # Action
        if policy is not None:
            action = policy(observation)
            action = action.detach().numpy()[0]
        else:
            action = env.action_space.sample()
            action = np.array([0,-0.5,0])
        print("action=",action)
        # Step
        observation, reward, done, info = env.step(action)

        # Plot
        # plt.imshow(observation[0])
        # plt.colorbar()
        # plt.show()

        if done:
            observation = env.reset()

    env.close()


if __name__ == "__main__":
    # Run an episode with random actions
    run_episode()

    # Run an episode with a random policy
    # from neural_networks import CustomNN
    # policy = CustomNN(3)
    # run_episode(policy)
