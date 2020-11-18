"""
使用策略迭代解决 FrozenLake environment 问题
Author: Liu
Date: 2020.11.17
参考：https://github.com/cuhkrlcourse/RLexample/tree/master/MDP
"""
import numpy as np
import gym
from gym import wrappers
from gym.envs.registration import register

def run_episode(env, policy, gamma = 1.0, render = False):
    """ 以当前策略运行一次场景完成时获得的总奖励 """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ 在给定价值函数v中选出新的策略 """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.env.P[s][a]])
        policy[s] = np.argmax(q_sa)  # 选择同一个状态下q值最大的动作a 更新到新的策略中
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    """ 
    Iteratively evaluate the value-function under policy.
    给定策略下，迭代计算每个状态的价值函数
    Alternatively, we could formulate a set of linear equations in iterms of v[s] 
    and solve them to find the value function.
    或者建立线性方程组并求解
    """
    v = np.zeros(env.env.nS) # 所有状态的价值函数的初值赋为0
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]]) # P记录着每个状态下采取各种行动的奖励，以及采取某个行动进入到s_状态的概率
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # 价值函数收敛
            break
    return v

def policy_iteration(env, gamma = 1.0):
    """ 策略迭代算法 """
    policy = np.random.choice(env.env.nA, size=(env.env.nS))  # 用随机数初始化策略列表
    max_iterations = 200000
    gamma = 1.0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy

if __name__ == '__main__':

    env_name  = 'FrozenLake-v0' # 'FrozenLake8x8-v0'
    env = gym.make(env_name)

    optimal_policy = policy_iteration(env, gamma = 1.0)
    scores = evaluate_policy(env, optimal_policy, gamma = 1.0)
    print('Average scores = ', np.mean(scores))