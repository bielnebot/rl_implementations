
def test_single_episode(env,policy):

    rewards = []

    obs, _ = env.reset()
    done = False

    t = 0

    for _ in range(150):
        t += 1
        action = policy(obs).detach().numpy()
        obs, rew, done, _, _ = env.step(action)

        rewards.append(rew)

    print(sum(rewards)/len(rewards))
    return rewards


def test_policy(env,policy):
    while True:
        test_single_episode(env,policy)