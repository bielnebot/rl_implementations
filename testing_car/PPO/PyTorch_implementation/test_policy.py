
def test_single_episode(env,policy):

    rewards = []

    obs = env.reset()
    done = False
    t = 0

    for _ in range(150):
        env.render()
        t += 1

        action = policy(obs).detach().numpy()[0]
        obs, rew, done, _ = env.step(action)

        rewards.append(rew)

    print(f"episode reward: {sum(rewards)}")
    return rewards


def test_policy(env,policy):
    while True:
        test_single_episode(env,policy)