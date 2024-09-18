import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import Wrapper


def observation_transformation(image):
    """
    Transforms the default observation into a simpler one
    :param image: the original observation of the environment. A 3D numpy array
    :return: a 2D numpy array with unified green and cropping at the bottom
    """
    # Extract green channel
    image = image[:, :, 1]
    # Crop the bottom of the image
    image = image[:84,:]
    # Normalise
    max_value = np.max(image)
    image = image / max_value
    # Set green tonalities equal
    image[image == 204 / max_value] = 1
    image[image == 230 / max_value] = 1
    # Reshape to (96,96)
    image = np.reshape(image, (1,) + image.shape)
    print("image.shape=",image.shape)
    return image


class CustomEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):

        observation, reward, done, info = super().step(action)
        observation = observation_transformation(observation)

        return observation, reward, done, info

    def reset(self):

        observation = super().reset()
        observation = observation_transformation(observation)

        return observation


def run_episode(policy=None):
    env = gym.make('CarRacing-v0')
    env = CustomEnv(env)

    observation = env.reset()

    for _ in range(100):
        observation = np.reshape(observation, (1,) + observation.shape)
        print("nova obs = ",observation, observation.shape)
        env.render()
        # Action
        if policy is not None:
            action = policy(observation)
            action = action.detach().numpy()[0]
        else:
            action = env.action_space.sample()
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
    # run_episode()

    # Run an episode with a random policy
    from neural_networks import CustomNN
    policy = CustomNN()
    run_episode(policy)