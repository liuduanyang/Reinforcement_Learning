import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('Taxi-v3')
state = env.reset()

# taxirow, taxicol, passloc, destidx = env.unwrapped.decode(state)
# print("出租车位置：{}".format((taxirow, taxicol)))
# print("乘客位置：{}".format(env.unwrapped.locs[passloc]))
# print("目的地位置：{}".format(env.unwrapped.locs[destidx]))
# env.render()
# env.step(1)

''' 定义智能体类 '''
class SARSAAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done, next_action):
        td_target = reward + self.gamma * self.q[next_state, next_action] * (1 - done)
        td_error = td_target - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error


''' 智能体与环境交互 '''
def play_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, done, _  = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation)  # 终止状态此步无意义
        if train:
            agent.learn(observation, action, reward, next_observation, done, next_action)
        if done:
            break
        observation, action = next_observation, next_action
    return episode_reward


# 初始化智能体
agent = SARSAAgent(env)

# 训练智能体
episodes = 5000
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_sarsa(env, agent, train=True, render=False)
    episode_rewards.append(episode_reward)
plt.plot(episode_rewards)
plt.show()

# 测试训练好的策略
agent.epsilon = 0.  # 取消探索，即取消柔性策略，直接选取最优动作
episode_rewards = [play_sarsa(env, agent) for _ in range(100)]
print("平均回合奖励：{} / {} = {}".format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))