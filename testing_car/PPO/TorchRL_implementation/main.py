import argparse

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# Reinforcement learning
import gym
from torchrl.envs.libs import GymWrapper
from torchrl.envs.utils import check_env_specs
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal

# Custom
from environment import CustomEnv
from neural_networks import CustomNN

# Hyperparameters
MAX_TIME_STEPS_PER_EPISODE = 1000
TIME_STEPS_PER_BATCH = 5000
SUB_BATCH_SIZE = 250
TOTAL_TIME_STEPS = 50_000_000
N_EPOCHS = 5

CLIP = 0.2
GAMMA = 0.99
LAMBDA = 0.95
LR = 1e-3

SIZE_OBSERVATIONS = 3
ACTION_BOUNDARY = 1


def train():
    writer = SummaryWriter()
    # layout = {
    #     "ABCDE": {
    #         "loss": ["Multiline", ["loss/total_loss", "loss/entropy_loss", "loss/loss_objective", "loss/loss_critic"]],
    #         "reward": ["Multiline", ["reward/reward", "reward/mean_batch_reward", "reward/cumulative_batch_reward"]]
    #     },
    # }

    # Create the environment
    env = gym.make('CarRacing-v0')
    # env = CustomEnv(env)
    env = GymWrapper(env)
    check_env_specs(env)

    buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=1000),
        sampler=SamplerWithoutReplacement()
    )

    model = TensorDictModule(
        nn.Sequential(
            CustomNN(output_dimension=3*2), # 3 actions * 2 (loc and scale)
            NormalParamExtractor()
        ),
        in_keys=["observation"],
        out_keys=["loc", "scale"]
    )

    actor = ProbabilisticActor(
        model,
        in_keys=["loc","scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.low,
            "max": env.action_spec.space.high
        },
        return_log_prob=True
    )
    # actor.out_keys = ['loc', 'scale', 'action', 'log_prob']

    critic = ValueOperator(
        CustomNN(output_dimension=1),
        in_keys=["observation"]
    )

    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=TIME_STEPS_PER_BATCH,
        total_frames=TOTAL_TIME_STEPS,
        max_frames_per_traj=MAX_TIME_STEPS_PER_EPISODE
    )

    loss_function = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=CLIP,
        normalize_advantage=True
    )

    advantage_function = GAE(
        value_network=critic,
        gamma=GAMMA,
        lmbda=LAMBDA,
        average_gae=True
    )

    optim = torch.optim.Adam(loss_function.parameters(), lr=LR)

    count_reward = 0
    count_loss = 0

    for batch_data in collector:
        for epoch in range(N_EPOCHS):

            advantage_function(batch_data)
            buffer.extend(batch_data.view(-1))

            for i in range(TIME_STEPS_PER_BATCH // SUB_BATCH_SIZE):
                sample = buffer.sample(SUB_BATCH_SIZE)

                loss_vals = loss_function(sample)

                loss_objective = loss_vals["loss_objective"]
                loss_critic = loss_vals["loss_critic"]
                loss_entropy = loss_vals["loss_entropy"]

                loss_val = loss_objective + loss_critic + loss_entropy

                writer.add_scalar("loss/total_loss",loss_val,count_loss)
                writer.add_scalar("loss/loss_objective",loss_objective,count_loss)
                writer.add_scalar("loss/loss_critic",loss_critic,count_loss)
                writer.add_scalar("loss/loss_entropy", loss_entropy, count_loss)
                count_loss += 1

                loss_val.backward()
                optim.step()
                optim.zero_grad()

        mean_reward = batch_data["next", "reward"].mean().item()
        cumulative_reward = batch_data["next", "reward"].sum().item()
        print(f"avg reward: {mean_reward: 4.4f}, cumulative reward: {cumulative_reward: 4.4f}")
        writer.add_scalar("reward/mean_batch_reward",mean_reward,count_reward)
        writer.add_scalar("reward/cumulative_batch_reward",cumulative_reward,count_reward)
        count_reward += 1

        model_id = f"checkpoint_{count_reward}"
        torch.save(
            {"model_state_dict": actor.state_dict(),
             "optimizer_state_dict": optim.state_dict(),
             "critic_state_dict": critic.state_dict()},
            f"./checkpoints/PPO_TorchRL/{model_id}.pt"
        )
    print("Done")


def test():

    # Initialsie policy
    model = TensorDictModule(
        nn.Sequential(
            CustomNN(3*2),
            NormalParamExtractor()
        ),
        in_keys=["observation"],
        out_keys=["loc", "scale"]
    )

    actor = ProbabilisticActor(
        model,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True
    )

    # Load weights
    model_id = f"checkpoint_44"
    model_path = f"./checkpoints/PPO_TorchRL/{model_id}.pt"
    checkpoint_data = torch.load(model_path, weights_only=True)
    actor.load_state_dict(checkpoint_data["model_state_dict"])

    # Simulate an episode
    test_single_episode(gravity,actor)


def main(args):
    if args.mode == "train":
        train()
    else:
        test()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("mode", type=str, help="train or test")
    # parser.add_argument("gravity", type=int, help="int from 0 to 20")
    # args = parser.parse_args()

    # main(args)
    train()