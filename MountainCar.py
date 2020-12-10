'''
以MountainCar(小车上山)为例学习如何与gym库中的环境进行交互
Author: Liu
Date: 2020.9.27
'''

import gym
import numpy as np

''' 1. 导入环境 '''
env = gym.make('MountainCar-v0')
print('观测空间 = {}'.format(env.observation_space))   # box(shape)
print('动作空间 = {}'.format(env.action_space))
print('观测空间取值范围 = {} ~ {}'.format(env.observation_space.low,env.observation_space.high))
print('动作空间的取值范围(动作数) = {}'.format(env.action_space.n))


''' 2. 定义与环境交互的智能体类 '''
class BespokeAgent:  # 与环境交互的智能体类
    def __init__(self,env):
        pass
    
    def decide(self,observation):  # 决策
        position,velocity = observation
        lb = min(-0.09*(position+0.25)**2+0.03, 0.3*(position+0.9)**4-0.008)
        ub = -0.07*(position+0.38)**2+0.06
        if(lb<velocity<ub):
            action = 2
        else:
            action = 0
        return action

    def learn(self, *args):  # 学习
        pass

agent = BespokeAgent(env)


''' 3. 智能体类的对象与环境交互 '''
def play_montecarlo(env,agent,render=False,train=False):
    episode_reward = 0.  # 浮点数，记录一个回合总奖励，初始化为0
    observation = env.reset()  # 环境初始化/重置环境开始新回合，返回一个观测
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation,action,reward,done)  # 智能体学习
        if(done):  # 如果回合结束
            break
        observation = next_observation
    return episode_reward


# 交互一个回合
# env.seed(0)  # 没啥用
# episode_reward = play_montecarlo(env,agent)
# env.close()

# 交互一百个回合
episode_rewards = [play_montecarlo(env,agent,render=False) for _ in range(100)]
env.close()
print('平均回合奖励 = {}'.format(np.mean(episode_rewards)))