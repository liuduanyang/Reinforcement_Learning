'''
求解Bellman期望方程 和 Bellman最优方程
Author: Liu
Date: 2020.11.13
'''

import sympy
from sympy import symbols
sympy.init_printing()

print('---  BellMan exception ---')


# 求解Bellman期望方程
v_hungry, v_full = symbols('v_hungry v_full')
q_hungry_eat, q_hungry_none, q_full_eat, q_full_none = symbols('q_hungry_eat q_hungry_none q_full_eat q_full_none')
alpha, beta, x, y, gamma = symbols('alpha beta x y gamma')
system = sympy.Matrix((
    (1, 0, x-1, -x, 0, 0, 0),
    (0, 1, 0, 0, -y, y-1, 0),
    (-gamma, 0, 1, 0, 0, 0, -2),
    ((alpha-1)*gamma, -alpha*gamma, 0, 1, 0, 0, 4*alpha-3),
    (-beta*gamma, (beta-1)*gamma, 0, 0, 1, 0, -4*beta+2),
    (0, -gamma, 0, 0, 0, 1, 1)
))

re = sympy.solve_linear_system(system,v_hungry,v_full,q_hungry_none,q_hungry_eat,q_full_none,q_full_eat)
for i in re:
    print(re[i])


print('---  BellMan optimization ---')


# 求解Bellman最优方程
sympy.init_printing()
alpha, beta, gamma = symbols('alpha beta gamma')
v_hungry, v_full = symbols('v_hungry v_full')
q_hungry_eat, q_hungry_none,q_full_eat, q_full_none = symbols('q_hungry_eat, q_hungry_none,q_full_eat, q_full_none')
xy_tuples = ((0,0), (1,0), (0,1), (1,1))

for x,y in xy_tuples:  # 分类讨论
    system = sympy.Matrix((
        (1,0,x-1,-x,0,0,0),
        (0,1,0,0,-y,y-1,0),
        (-gamma,0,1,0,0,0,-2),
        ((alpha-1)*gamma,-alpha*gamma,0,1,0,0,4*alpha-3),
        (-beta*gamma,(beta-1)*gamma,0,0,1,0,-4*beta+2),
        (0,-gamma,0,0,0,1,1)
    ))
    result = sympy.solve_linear_system(system, v_hungry, v_full, q_hungry_eat, q_hungry_none, q_full_eat, q_full_none)

    msgx = 'v(饿) = q(饿，{}吃)'.format('' if x else '不')
    msgy = 'v(饱) = q(饿，{}吃)'.format('不' if y else '')
    print('==== {}, {} ==== x = {}, y = {} ===='.format(msgx, msgy, x ,y))

    for i in re:
        print(re[i])