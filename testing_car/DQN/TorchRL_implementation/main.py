import gym
import torch.optim
from torchrl.envs.libs import GymWrapper
from torchrl.envs.utils import check_env_specs
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import EGreedyModule, QValueModule
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.objectives import DQNLoss, SoftUpdate

from environment import CustomEnv
from neural_networks import CustomNN

NUMBER_ACTIONS = 5


def train():
    # Training parameters
    frames_per_batch = 5000
    total_frames = 50_000_000
    minibatch_size = 250
    lr = 1e-3

    # PPO parameters
    epsilon_start = 0.5
    epsilon_end = 0.1
    step_epsilon_decay = 50_000
    double_DQN_epsilon_update = 0.5

    episode_length = 1000

    # Create environment
    env = gym.make('CarRacing-v0')
    env = CustomEnv(env)
    env = GymWrapper(env)
    check_env_specs(env)

    # Initialise Q-function
    net = TensorDictModule(
        CustomNN(output_dimension=NUMBER_ACTIONS),
        in_keys="observation",
        out_keys="action_value"
    )

    qval = QValueModule(spec=env.action_spec, action_space=None)

    exploration_module = EGreedyModule(spec=env.action_spec,
                                       eps_init=epsilon_start,
                                       eps_end=epsilon_end,
                                       annealing_num_steps=step_epsilon_decay)

    # Initialise a stochastic and a deterministic policy
    policy = TensorDictSequential(net,qval)
    stochastic_policy = TensorDictSequential(policy, exploration_module)

    policy(env.reset())

    collector = SyncDataCollector(env,
                                  stochastic_policy,
                                  frames_per_batch=frames_per_batch,
                                  total_frames=total_frames,
                                  max_frames_per_traj=episode_length)

    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(frames_per_batch)
    )

    print("action_spec=",env.action_spec)
    print("policy=",policy)
    loss_function = DQNLoss(value_network=policy,
                            action_space=env.action_spec, # using env.action_spec leads to one-hot
                            delay_value=True) # to use a double DQN

    optim = torch.optim.Adam(policy.parameters(), lr=lr)

    updater = SoftUpdate(loss_function, eps=double_DQN_epsilon_update) # for the double DQN

    counter = 0
    # Training loop
    for batch_data in collector:

        buffer.extend(batch_data)
        for _ in range(frames_per_batch // minibatch_size):
            sample = buffer.sample(minibatch_size)
            print("sample=",sample, sample.shape)
            print("sample['action_value']=", sample["action"], sample["action"].shape)
            print("sample['action_value']=", sample["action_value"], sample["action_value"].shape)
            loss_value = loss_function(sample)
            loss_value = loss_value["loss"]

            # Calculate gradients + backpropagate
            loss_value.backward()
            optim.step()
            optim.zero_grad()

        # Update the epsilon decay at each new batch
        exploration_module.step(batch_data.numel())

        # Save checkpoint
        if counter % 10 == 0:
            # TODO: torch.save()
            pass

        counter += 1

        updater.step()
        collector.update_policy_weights_()


def test():
    pass


def main(args):
    if args.mode == "train":
        train()
    else:
        test()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("mode", type=str, help="train or test")
    # args = parser.parse_args()

    # main(args)
    train()