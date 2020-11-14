''' 
策略评估(计算价值函数)过程演示
Author: Liu
Date: 2020.11.14
code copy by: https://zhuanlan.zhihu.com/p/26699028
Environment: https://github.com/dennybritz/reinforcement-learning
'''

import numpy as np
import pprint
import sys
if "../" not in sys.path:
    sys.path.append("../")
from reinforcement_learning_git.lib.envs.gridworld import GridworldEnv # 这个Gridworld环境是构建好的，直接导入
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()
# First, do the policy evaluation 首先进行随机策略的评估
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    # Evaluate a policy given an environment and a full description of the environment's dynamics.环境信息已知
    # Args:
    #     policy:[S,A] shaped matrix representing the policy. 策略就是从状态到行动的映射，随机策略的话这里每个方向都是1/4
    #     env. OpenAI env. env.P represents the transition probabilities of the environment.
    #         env.P[s][a] is a (prob, next_state, reward, done) tuple
    #     thetha: We stop evaluation once our value function change is less than theta for all states.如果前后两次的变化很小，小于这个门槛，那么认为已经收敛了
    # Returns:
    #     Vector of length env.nS representing the value function.返回一个价值函数列表
    V = np.zeros(env.nS)
    #print ("env.nS is", env.nS) 哪里不清楚的就print一下看看
    #print V
    #i = 0
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                #print a, action_prob 这里print出来就是上下左右，概率都是1/4
                # For each action, look at the possible next states..
                for prob, next_state, reward, done in env.P[s][a]:
                    #print env.P[s][a]
                    # Calculate the expected value 计算该策略下的价值函数
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
                    #i = i + 1
            # How much our value function changed (across any states)
            #print i, delta, v, V[s]
            delta = max(delta, np.abs(v - V[s]))  #整体来讲，这个delta是先变大，后来经过不断迭代逐渐变小，理论上趋于零的时候就是收敛的时候
            #delta = np.abs(v - V[s])
            #print v,V[s]
            V[s] = v 
        # Stop evaluating once our value function change is bellow a threshold 
        if delta < theta:
            break
    return np.array(V) # 最终，随机策略下的价值函数出炉

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

print ("Value Function")
print (v)
print ("")
print ('Reshaped Grid Value Function:')
print (v.reshape(env.shape))