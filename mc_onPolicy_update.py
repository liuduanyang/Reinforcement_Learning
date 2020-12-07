import gym
import numpy as np

env = gym.make('Blackjack-v0')

env.reset()

def ob2state(observation):
    return (observation[0], observation[1], int(observation[2]))

''' 同策回合更新策略评估 只涉及评估不涉及策略改进 '''
def evalute_action_monte_carlo(env, policy, episode_num=500000):
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        # 一回合
        state_actions = []
        observation = env.reset()
        while True:
            state = ob2state(observation)  # 指定初始状态
            action = np.random.choice(env.action_space.n, p=policy[state])
            state_actions.append((state, action))  # 轨迹
            observation, reward, done, _ = env.step(action)
            if done:
                break
        g = reward  # 本应逆序求每个轨迹中的每个(q,a)对的收益G，但由于游戏特性 这里不需要
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g-q[state][action]) / c[state][action]

    return q

policy = np.zeros((22,11,2,2))
policy[20:,:,:,0] = 1    # 玩家点数>=20时，不再要牌
policy[:20,:,:,1] = 1   # 玩家点数<20时，再要一张牌
q = evalute_action_monte_carlo(env, policy)  # q是四维数组
v = (q * policy).sum(axis=-1)  # v是三维数组
print(q)



''' 带起始探索的同策回合更新(最优almost 求解) 策略评估+改进 '''
def monte_carlo_with_exploring_start(env, episode_num=500000):
    policy = np.zeros((22, 11, 2 ,2))
    policy[:,:,:,1] = 1.
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        # 一回合
        # 随机选择初始状态和初始动作
        state = (                      # 随机产生初始状态
            np.random.randint(3,22),  # 对12-21更感兴趣 所以忽略3-11
            np.random.randint(1,11),
            np.random.randint(2))
        action = np.random.randint(2)

        state_actions = []
        env.reset()
        while True:
            state_actions.append((state,action))
            observation, reward, done ,_ = env.step(action)
            if done:
                break
            state = ob2state(observation)
            acition = np.random.choice(env.action_space.n, p=policy[state])
        g = reward
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
            
            # 策略改进
            a = q[state].argmax()
            policy[state] = 0.
            policy[state][a] = 1.
    return policy, q

policy, q = monte_carlo_with_exploring_start(env)
v = q.max(axis = -1)
print(q)
print(policy)


''' 基于柔性策略的同策回合更新 策略评估+改进 '''
def monte_carlo_with_soft(env, episode_num=500000, epsilon=0.1):
    policy = np.ones((22,11,2,2)) * 0.5
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        # 一回合
        observation = env.reset()
        state_actions = []
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break
        g = reward
        for state, action in state_actions:
            c[state][action] += 1
            q[state][action] += (g - q[state][action]) / c[state][action]

            # 策略改进
            a = q[state].argmax()
            policy[state] = epsilon / 2.
            policy[state][a] += (1. - epsilon)

    return policy, q

policy, q = monte_carlo_with_soft(env)
v = q.max(axis = -1)
print(q)
print(policy)
