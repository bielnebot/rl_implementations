import numpy as np
import torch
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.nn import MSELoss

from torch.utils.tensorboard import SummaryWriter

# Custom modules
from neural_networks import CustomNN


class PPO():
    def __init__(self,env):
        # Environment
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Hyperparameters
        self._init_hyperparameters()

        # Actor and critic
        self.actor = CustomNN(self.obs_dim, self.act_dim)
        self.critic = CustomNN(self.obs_dim, 1) # 1-dimensional state value

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.variances = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.covariance_matrix = torch.diag(self.variances)

    def _init_hyperparameters(self):
        self.time_steps_per_batch = 1000
        self.max_time_steps_per_episode = 205
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005

    def get_action(self, observation):

        # Query the actor for a mean action
        mean = self.actor(observation)

        # Create a distribution
        distribution = MultivariateNormal(mean, self.covariance_matrix)

        # Sample an action and get its log_prob
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.detach().numpy(), log_prob.detach()


    def compute_rewards_to_go(self, batch_rewards):
        """
        Episodes with 1, 2, ..., i, ..., n-1, n time-steps

        1 --> r_1 + γ(r_2 + γ(r_3 + γ(r_4 + γ(r_5 + γ( ... )))))
        1 --> r_2 + γ(r_3 + γ(r_4 + γ(r_5 + γ( ... ))))
        ...
        n-2 --> r_n-2 + γ(r_n-1 + γ(r_n))
        n-1 --> r_n-1 + γ(r_n)
        n --> r_n

        :param batch_rewards: [ [r_1, r_2, r_3, ...], the reward at each dt of the first episode
                                [r_1, r_2, r_3, ...], the reward at each dt of the second episode
                                [...],
                                ...]
        :return: batch_rewards_to_go: len(batch_rewards_to_go) = total time-steps simulated =
        = n_episodes * time-steps per episodes
        = [for each time-step of the simulation, the rewards to go  until the end of its corresponding episode]
        """
        batch_rewards_to_go = []

        for episode_rewards in reversed(batch_rewards):
            discounted_reward = 0
            for reward in reversed(episode_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rewards_to_go.insert(0, discounted_reward)

        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float)

        return  batch_rewards_to_go

    def rollout(self):
        batch_observations = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        # batch_rewards_to_go = []
        batch_lenghts = []

        t = 0

        while t < self.time_steps_per_batch: # for a few time-steps, do...
            # Rewards this episode
            episode_rewards = []

            observation, _ = self.env.reset()
            done = False

            for time_step_i in range(self.max_time_steps_per_episode): # for each episode, do...
                t += 1

                # Collect observation
                batch_observations.append(observation)

                # New action
                action, log_prob = self.get_action(observation)

                # Perform action
                observation, reward, done, _, _ = self.env.step(action)

                # Collect action, reward and log_prob
                batch_actions.append(action)
                episode_rewards.append(reward)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episode rewards and length
            batch_rewards.append(episode_rewards)
            batch_lenghts.append(time_step_i + 1)

        batch_observations = torch.tensor(batch_observations, dtype=torch.float)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)

        return batch_observations, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lenghts, batch_rewards


    def evaluate(self, batch_observations, batch_actions):
        V = self.critic(batch_observations).squeeze()

        mean = self.actor(batch_observations)

        distribution = MultivariateNormal(mean,self.covariance_matrix)

        log_probs = distribution.log_prob(batch_actions)

        return V, log_probs

    def learn(self, total_time_steps):
        writer = SummaryWriter()
        t_so_far = 0

        while t_so_far < total_time_steps:
            batch_observations, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lenghts, batch_rewards = self.rollout()

            # Compute V_{phi, k}
            V, _ = self.evaluate(batch_observations, batch_actions)

            # Compute advantage
            A_k = batch_rewards_to_go - V.detach()
            # Normalise advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Compute V_phi and pi_theta(a_t | s_t)
                V, current_log_probs = self.evaluate(batch_observations, batch_actions)

                # Ratio pi_theta/pi_theta_k
                ratio = torch.exp(current_log_probs - batch_log_probs)

                # Loss
                surr1 = ratio * A_k
                surr2 = torch.clamp(ratio, 1-self.clip, 1 + self.clip) * A_k
                actor_loss = (-torch.min(surr1,surr2)).mean()

                # Calculate gradients + backpropagation
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                critic_loss = MSELoss()(V, batch_rewards_to_go)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            average_reward = np.mean(batch_rewards)
            writer.add_scalar("reward",average_reward,t_so_far)

            t_so_far += np.sum(batch_lenghts)

            if t_so_far % 1000 == 0:
                torch.save(self.actor.state_dict(), f"./checkpoints/PPO_PyTorch/g_{self.env.unwrapped.g}_{t_so_far}.pth")


