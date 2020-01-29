# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 21:11:14 2019

@author: Laila Niazy 03660940
"""
import numpy as np
import matplotlib.pyplot as plt

def pi_to_arrow(pi):
    '''getting the setting the coordinates of the states
    and setting them for the arrows (policies) to plot the arrows in the heatmap'''
    arrows = []
    for s in pi:
        i,j = np.ravel(s)
        if pi[s] == 'up':
          arrows.append([i+0.25,j,-0.25,0])
        elif pi[s] == 'down':
            arrows.append([i-0.25,j,0.25,0])
        elif pi[s] == 'left':
            arrows.append([i,j+0.25,0,-0.25]) 
        elif pi[s] == 'right':
            arrows.append([i,j-0.25,0,0.25])
        else:
            continue
    return arrows
    
def V_to_matrix(V,h,w):
    '''form the dict V to a numpy matrix for plotting'''
    matrix = np.empty([h,w])
    for s in V:
        matrix[s] = V[s]
    return matrix
    
def V_to_plot(V,pi,h,w):
    '''form the dict V to a numpy matrix for plotting'''
    matrix = np.zeros([h,w])
    for s in V:
        if s in pi.keys():
           matrix[s] = V[s]
        else:
            matrix[s] = None
            continue
    return matrix
    
def heatmap(V, pi, h, w, Algorithm, gamma, method):
    matrix = V_to_plot(V,pi,h,w)
    arrows= pi_to_arrow(pi)
    plt.figure()
    for (x,y,dx,dy) in arrows:
       plt.arrow(y,x,dy,dx,head_width=0.1, head_length=0.1,head_starts_at_zero=True)
    heatmap = plt.imshow(matrix)
    plt.colorbar()
    plt.title('{} with gamma={} and method {}'.format(Algorithm,str(gamma),str(method)))
    plt.savefig('Images/Heatmap_{}_gamma={}_method{}.png'.format(Algorithm,str(gamma),str(method)))
    
    return heatmap
    
def plot_error(error,Algorithm, gamma, method):
    plt.figure()
    error_plot = plt.plot(error)
    plt.title('{} with gamma = {} and method {}'.format(Algorithm,str(gamma), str(method)))
    plt.xlabel('Number of iterations')
    plt.ylabel('Error')
    plt.savefig('Images/Error_{}_gamma={}_method{}.png'.format(Algorithm,str(gamma), str(method)))
    return error_plot
