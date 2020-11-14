'''
gym的安装与初步使用
Author: Liu
Data: 2020.9.26
'''

import gym
env = gym.make('CartPole-v0')  # 获得倒立摆环境


from gym import envs
env_specs = envs.registry.all()
env_ids = [env_spec.id for env_spec in env_specs]
print(len(env_ids))  # 查看gym库注册的所有环境


# 使用环境对象env
env.reset()  # 初始化环境对象，返回智能体的初始观测(np.array对象)
env.step()  # 用于接收智能体的动作
env.render()  # 图形化当前环境，需要使用env.close() 释放内存资源
env.close()  # 关闭环境
