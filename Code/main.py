# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 11:05:19 2019

@author: Laila Niazy 03660940
"""
from Analyse_Maze import Analyse_Maze
from mdp import MarkovDecisionProcess
import plotting as pl
import matplotlib.pyplot as plt
import os
import sys


#create directory to save figures
if not(os.path.exists('Images')):
    os.mkdir('Images')
    
"""Add the path of the maze"""
if len(sys.argv)>1:
    path=str(sys.argv[1])
else:
    path='maze.txt'
    
maze = Analyse_Maze(path)
#get the reward and transtion dictionary
Reward_1 = maze.reward(method=1)
Reward_2 = maze.reward(method=2)
Transitions = maze.look_up()
#initialize epsilon
epsilon = 0.001
#to suppress the plots
plt.ioff()

#######################################
# cost function 1 , gamma=0.99
#######################################
gamma = .99
#Initialize the MarkovDecisionProcess object for method 1 of the reward
mdp1_a = MarkovDecisionProcess(transition=Transitions, reward=Reward_1, method = 1, gamma=gamma, epsilon =epsilon)
""" value iteration with method 1"""
V1_a, error_v1_a = mdp1_a.value_iteration(maze.maze)
pi_v1_a = mdp1_a.best_policy(V1_a)
pl.heatmap(V1_a, pi_v1_a, maze.height, maze.width,'VI',gamma,1)
pl.plot_error(error_v1_a,'VI',gamma,1)
""" policy iteration with method 1"""
error_p1_a, pi_p1_a, U1_a = mdp1_a.policy_iteration(maze.maze)
pl.heatmap(U1_a, pi_p1_a, maze.height, maze.width,'PI',gamma,1)
pl.plot_error(error_p1_a,'PI',gamma,1)

#######################################
# cost function 2 , gamma=0.99
#######################################
gamma = .99
#Initialize the MarkovDecisionProcess object for method 2 of the reward
mdp2_a = MarkovDecisionProcess(transition=Transitions, reward=Reward_2, method = 2,gamma=gamma, epsilon =epsilon)
""" value iteration with method 2"""
V2_a, error_v2_a = mdp2_a.value_iteration(maze.maze)
pi_v2_a = mdp2_a.best_policy(V2_a)
pl.heatmap(V2_a, pi_v2_a, maze.height, maze.width,'VI',gamma,2)
pl.plot_error(error_v2_a,'VI',gamma,2)

""" policy iteration with method 2"""
error_p2_a, pi_p2_a, U2_a = mdp2_a.policy_iteration(maze.maze)
pl.heatmap(U2_a, pi_p2_a, maze.height, maze.width,'PI',gamma,2)
pl.plot_error(error_p2_a,'PI',gamma,2)


#######################################
# cost function 1 , gamma=0.9
#######################################
gamma = .9
#Initialize the MarkovDecisionProcess object for method 1 of the reward
mdp1_b = MarkovDecisionProcess(transition=Transitions, reward=Reward_1, method = 1, gamma=gamma, epsilon =epsilon)
""" value iteration with method 1"""
V1_b, error_v1_b = mdp1_b.value_iteration(maze.maze)
pi_v1_b = mdp1_b.best_policy(V1_b)
pl.heatmap(V1_b, pi_v1_b, maze.height, maze.width,'VI',gamma,1)
pl.plot_error(error_v1_b,'VI',gamma,1)
""" policy iteration with method 1"""
error_p1_b, pi_p1_b, U1_b = mdp1_b.policy_iteration(maze.maze)
pl.heatmap(U1_b, pi_p1_b, maze.height, maze.width,'PI',gamma,1)
pl.plot_error(error_p1_b,'PI',gamma,1)

#######################################
# cost function 2 , gamma=0.9
#######################################
gamma = .9
#Initialize the MarkovDecisionProcess object for method 2 of the reward
mdp2_b = MarkovDecisionProcess(transition=Transitions, reward=Reward_2, method = 2,gamma=gamma, epsilon =epsilon)
""" value iteration with method 2"""
V2_b, error_v2_b = mdp2_b.value_iteration(maze.maze)
pi_v2_b = mdp2_b.best_policy(V2_b)
pl.heatmap(V2_b, pi_v2_b, maze.height, maze.width,'VI',gamma,2)
pl.plot_error(error_v2_b,'VI',gamma,2)

""" policy iteration with method 2"""
error_p2_b, pi_p2_b, U2_b = mdp2_b.policy_iteration(maze.maze)
pl.heatmap(U2_b, pi_p2_b, maze.height, maze.width,'PI',gamma,2)
pl.plot_error(error_p2_b,'PI',gamma,2)

#######################################
# cost function 1 , gamma=0.5
#######################################
gamma = .5
#Initialize the MarkovDecisionProcess object for method 1 of the reward
mdp1_c = MarkovDecisionProcess(transition=Transitions, reward=Reward_1, method = 1, gamma=gamma, epsilon =epsilon)
""" value iteration with method 1"""
V1_c, error_v1_c = mdp1_c.value_iteration(maze.maze)
pi_v1_c = mdp1_c.best_policy(V1_c)
pl.heatmap(V1_c, pi_v1_c, maze.height, maze.width,'VI',gamma,1)
pl.plot_error(error_v1_c,'VI',gamma,1)
""" policy iteration with method 1"""
error_p1_c, pi_p1_c, U1_c = mdp1_c.policy_iteration(maze.maze)
pl.heatmap(U1_c, pi_p1_c, maze.height, maze.width,'PI',gamma,1)
pl.plot_error(error_p1_c,'PI',gamma,1)

#######################################
# cost function 2 , gamma=0.5
#######################################
gamma = .5
#Initialize the MarkovDecisionProcess object for method 2 of the reward
mdp2_c = MarkovDecisionProcess(transition=Transitions, reward=Reward_2, method = 2,gamma=gamma, epsilon =epsilon)
""" value iteration with method 2"""
V2_c, error_v2_c = mdp2_c.value_iteration(maze.maze)
pi_v2_c = mdp2_c.best_policy(V2_c)
pl.heatmap(V2_c, pi_v2_c, maze.height, maze.width,'VI',gamma,2)
pl.plot_error(error_v2_c,'VI',gamma,2)

""" policy iteration with method 2"""
error_p2_c, pi_p2_c, U2_c = mdp2_c.policy_iteration(maze.maze)
pl.heatmap(U2_c, pi_p2_c, maze.height, maze.width,'PI',gamma,2)
pl.plot_error(error_p2_c,'PI',gamma,2)

#######################################
# cost function 1 , gamma=0.01
#######################################
gamma = .01
#Initialize the MarkovDecisionProcess object for method 1 of the reward
mdp1_d = MarkovDecisionProcess(transition=Transitions, reward=Reward_1, method = 1, gamma=gamma, epsilon =epsilon)
""" value iteration with method 1"""
V1_d, error_v1_d = mdp1_d.value_iteration(maze.maze)
pi_v1_d = mdp1_d.best_policy(V1_d)
pl.heatmap(V1_d, pi_v1_d, maze.height, maze.width,'VI',gamma,1)
pl.plot_error(error_v1_d,'VI',gamma,1)
""" policy iteration with method 1"""
error_p1_d, pi_p1_d, U1_d = mdp1_d.policy_iteration(maze.maze)
pl.heatmap(U1_d, pi_p1_d, maze.height, maze.width,'PI',gamma,1)
pl.plot_error(error_p1_d,'PI',gamma,1)

#######################################
# cost function 2 , gamma=0.01
#######################################
gamma = .01
#Initialize the MarkovDecisionProcess object for method 2 of the reward
mdp2_d = MarkovDecisionProcess(transition=Transitions, reward=Reward_2, method = 2,gamma=gamma, epsilon =epsilon)
""" value iteration with method 2"""
V2_d, error_v2_d = mdp2_d.value_iteration(maze.maze)
pi_v2_d = mdp2_d.best_policy(V2_d)
pl.heatmap(V2_d, pi_v2_d, maze.height, maze.width,'VI',gamma,2)
pl.plot_error(error_v2_d,'VI',gamma,2)

""" policy iteration with method 2"""
error_p2_d, pi_p2_d, U2_d = mdp2_d.policy_iteration(maze.maze)
pl.heatmap(U2_d, pi_p2_d, maze.height, maze.width,'PI',gamma,2)
pl.plot_error(error_p2_d,'PI',gamma,2)

#######################################
# cost function 1 , gamma=0.6
#######################################
gamma = .6
#Initialize the MarkovDecisionProcess object for method 1 of the reward
mdp1_e = MarkovDecisionProcess(transition=Transitions, reward=Reward_1, method = 1, gamma=gamma, epsilon =epsilon)
""" value iteration with method 1"""
V1_e, error_v1_e = mdp1_e.value_iteration(maze.maze)
pi_v1_e = mdp1_e.best_policy(V1_e)
pl.heatmap(V1_e, pi_v1_e, maze.height, maze.width,'VI',gamma,1)
pl.plot_error(error_v1_e,'VI',gamma,1)
""" policy iteration with method 1"""
error_p1_e, pi_p1_e, U1_e = mdp1_e.policy_iteration(maze.maze)
pl.heatmap(U1_e, pi_p1_e, maze.height, maze.width,'PI',gamma,1)
pl.plot_error(error_p1_e,'PI',gamma,1)

#######################################
# cost function 2 , gamma=0.6
#######################################
gamma = .6
#Initialize the MarkovDecisionProcess object for method 2 of the reward
mdp2_e = MarkovDecisionProcess(transition=Transitions, reward=Reward_2, method = 2,gamma=gamma, epsilon =epsilon)
""" value iteration with method 2"""
V2_e, error_v2_e = mdp2_e.value_iteration(maze.maze)
pi_v2_e = mdp2_e.best_policy(V2_e)
pl.heatmap(V2_e, pi_v2_e, maze.height, maze.width,'VI',gamma,2)
pl.plot_error(error_v2_e,'VI',gamma,2)

""" policy iteration with method 2"""
error_p2_e, pi_p2_e, U2_e = mdp2_e.policy_iteration(maze.maze)
pl.heatmap(U2_e, pi_p2_e, maze.height, maze.width,'PI',gamma,2)
pl.plot_error(error_p2_e,'PI',gamma,2)

#######################################
# cost function 1 , gamma=0.7
#######################################
gamma = .7
#Initialize the MarkovDecisionProcess object for method 1 of the reward
mdp1_f = MarkovDecisionProcess(transition=Transitions, reward=Reward_1, method = 1, gamma=gamma, epsilon =epsilon)
""" value iteration with method 1"""
V1_f, error_v1_f = mdp1_f.value_iteration(maze.maze)
pi_v1_f = mdp1_f.best_policy(V1_f)
pl.heatmap(V1_f, pi_v1_f, maze.height, maze.width,'VI',gamma,1)
pl.plot_error(error_v1_f,'VI',gamma,1)
""" policy iteration with method 1"""
error_p1_f, pi_p1_f, U1_f = mdp1_f.policy_iteration(maze.maze)
pl.heatmap(U1_f, pi_p1_f, maze.height, maze.width,'PI',gamma,1)
pl.plot_error(error_p1_f,'PI',gamma,1)

#######################################
# cost function 2 , gamma=0.7
#######################################
gamma = .7
#Initialize the MarkovDecisionProcess object for method 2 of the reward
mdp2_f = MarkovDecisionProcess(transition=Transitions, reward=Reward_2, method = 2,gamma=gamma, epsilon =epsilon)
""" value iteration with method 2"""
V2_f, error_v2_f = mdp2_f.value_iteration(maze.maze)
pi_v2_f = mdp2_f.best_policy(V2_f)
pl.heatmap(V2_f, pi_v2_f, maze.height, maze.width,'VI',gamma,2)
pl.plot_error(error_v2_f,'VI',gamma,2)

""" policy iteration with method 2"""
error_p2_f, pi_p2_f, U2_f = mdp2_f.policy_iteration(maze.maze)
pl.heatmap(U2_f, pi_p2_f, maze.height, maze.width,'PI',gamma,2)
pl.plot_error(error_p2_f,'PI',gamma,2)

#######################################
# cost function 1 , gamma=0.8
#######################################
gamma = .8
#Initialize the MarkovDecisionProcess object for method 1 of the reward
mdp1_g = MarkovDecisionProcess(transition=Transitions, reward=Reward_1, method = 1, gamma=gamma, epsilon =epsilon)
""" value iteration with method 1"""
V1_g, error_v1_g = mdp1_g.value_iteration(maze.maze)
pi_v1_g = mdp1_g.best_policy(V1_g)
pl.heatmap(V1_g, pi_v1_g, maze.height, maze.width,'VI',gamma,1)
pl.plot_error(error_v1_g,'VI',gamma,1)
""" policy iteration with method 1"""
error_p1_g, pi_p1_g, U1_g = mdp1_g.policy_iteration(maze.maze)
pl.heatmap(U1_g, pi_p1_g, maze.height, maze.width,'PI',gamma,1)
pl.plot_error(error_p1_g,'PI',gamma,1)

#######################################
# cost function 2 , gamma=0.8
#######################################
gamma = .8
#Initialize the MarkovDecisionProcess object for method 2 of the reward
mdp2_g = MarkovDecisionProcess(transition=Transitions, reward=Reward_2, method = 2,gamma=gamma, epsilon =epsilon)
""" value iteration with method 2"""
V2_g, error_v2_g = mdp2_g.value_iteration(maze.maze)
pi_v2_g = mdp2_g.best_policy(V2_g)
pl.heatmap(V2_g, pi_v2_g, maze.height, maze.width,'VI',gamma,2)
pl.plot_error(error_v2_g,'VI',gamma,2)

""" policy iteration with method 2"""
error_p2_g, pi_p2_g, U2_g = mdp2_g.policy_iteration(maze.maze)
pl.heatmap(U2_g, pi_p2_g, maze.height, maze.width,'PI',gamma,2)
pl.plot_error(error_p2_g,'PI',gamma,2)

#######################################
# cost function 1 , gamma=0.3
#######################################
gamma = .3
#Initialize the MarkovDecisionProcess object for method 1 of the reward
mdp1_h = MarkovDecisionProcess(transition=Transitions, reward=Reward_1, method = 1, gamma=gamma, epsilon =epsilon)
""" value iteration with method 1"""
V1_h, error_v1_h = mdp1_h.value_iteration(maze.maze)
pi_v1_h = mdp1_h.best_policy(V1_h)
pl.heatmap(V1_h, pi_v1_h, maze.height, maze.width,'VI',gamma,1)
pl.plot_error(error_v1_h,'VI',gamma,1)
""" policy iteration with method 1"""
error_p1_h, pi_p1_h, U1_h = mdp1_h.policy_iteration(maze.maze)
pl.heatmap(U1_h, pi_p1_h, maze.height, maze.width,'PI',gamma,1)
pl.plot_error(error_p1_h,'PI',gamma,1)

#######################################
# cost function 1 , gamma=0.3
#######################################
gamma = .3
#Initialize the MarkovDecisionProcess object for method 2 of the reward
mdp2_h = MarkovDecisionProcess(transition=Transitions, reward=Reward_2, method = 2,gamma=gamma, epsilon =epsilon)
""" value iteration with method 2"""
V2_h, error_v2_h = mdp2_h.value_iteration(maze.maze)
pi_v2_h = mdp2_h.best_policy(V2_h)
pl.heatmap(V2_h, pi_v2_h, maze.height, maze.width,'VI',gamma,2)
pl.plot_error(error_v2_h,'VI',gamma,2)

""" policy iteration with method 2"""
error_p2_h, pi_p2_h, U2_h = mdp2_h.policy_iteration(maze.maze)
pl.heatmap(U2_h, pi_p2_h, maze.height, maze.width,'PI',gamma,2)
pl.plot_error(error_p2_h,'PI',gamma,2)




