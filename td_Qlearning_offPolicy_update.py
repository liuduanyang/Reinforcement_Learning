import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('Taxi-v3')
state = env.reset()

''' 定义智能体类 '''
class QLearningAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, episode=.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.episode = episode
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state):
        if np.random.uniform() > self.episode:   ''' 策略一 采样轨迹使用episode柔性策略 '''
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action
    
    def learn(self, observation, action, reward, next_observation, done):
        td_target = reward + self.gamma * self.q[next_observation].max() * (1. - done)   ''' 策略二 策略更新使用贪婪策略 '''
        td_error = td_target - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error


''' 智能体与环境交互 '''
def paly_qlearning(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, next_observation, done)
        if done:
            break
        observation = next_observation
    return episode_reward


# 初始化智能体
agent = QLearningAgent(env)

# 训练智能体
episodes = 5000
episode_rewards = []
for episode in range(episodes):
    episode_reward = paly_qlearning(env, agent, train=True, render=False)
    episode_rewards.append(episode_reward)
plt.plot(episode_rewards)
plt.show()

# 测试训练好的策略
agent.episode = 0.
episode_rewards = [paly_qlearning(env, agent) for _ in range(100)]
print("平均回合奖励：{} / {} = {}".format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))