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

NUMBER_ACTIONS = 3


def train():
    # Training
    frames_per_batch = ...
    total_frames = ...
    minibatch_size = ...
    lr = ...

    # PPO
    epsilon_start = ...
    epsilon_end = ...
    step_epsilon_decay = ...
    double_DQN_epsilon_update = ...

    episode_length = ...
    agent_hz = ...
    simulation_hz = ...

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

    loss_function = DQNLoss(value_network=policy,
                            action_space=env.action_spec,
                            delay_value=True) # to use a double DQN

    optim = torch.optim.Adam(policy.parameters(), lr=lr)

    updater = SoftUpdate(loss_function, eps=double_DQN_epsilon_update) # for the double DQN

    # Training loop
    # TODO


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