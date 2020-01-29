# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 20:53:43 2019

@author: Laila Niazy 03660940

A class for the maze with different methods
"""

import numpy as np
     
class Analyse_Maze:
    
    def __init__(self, path):
        #initializing different values
        self.maze, self.width, self.height = self.generateMaze(path)
        self.initial_state = self.find_specific_states("S")
        self.Trap = self.find_specific_states('T')
        self.Goal = self.find_specific_states('G')
        self.p = 0.1
        self.action = np.array(['up', 'down', 'left', 'right', 'idle'])
        self.prob = {self.action[0]:[self.p,np.subtract(1,np.multiply(2,self.p)), self.p],
                     self.action[1]:[self.p,np.subtract(1,np.multiply(2,self.p)), self.p],
                     self.action[2]:[self.p,np.subtract(1,np.multiply(2,self.p)), self.p],
                     self.action[3]:[self.p,np.subtract(1,np.multiply(2,self.p)), self.p],
                     self.action[4]:[self.p,np.subtract(1,np.multiply(2,self.p)), self.p]}
                       
    def generateMaze(self, path):
        """Return maze from textfile as a nested list with the width and height of the maze"""
        text = open(path, "r")
        self.maze = []
        for line in text:
            if line[0] == "#":
                continue
            else: 
                char = [x for x in line if x != "\n"]  
                self.maze.append(char)
        text.close()
        return(self.maze, len(self.maze[1]), len(self.maze))
        
    def find_specific_states(self, case):
        
        '''getting the coordinates for the start, goal and trap state'''
        location = [(i,j) for i,m in enumerate(self.maze) for j,s in enumerate(m) if s == case]

        return location[0]
        
    def checkwall(self, s):
        '''checking if this state is a wall or outside the maze'''
        if self.maze[s[0]][s[1]] == '1' :
            return(False)   
        else:
            return(True)
        
    def allowed_actions(self, i, j, is_wall=False):
        '''Based on the current states, which actions are allowed and to which 
        successor states will they lead
        Return a list of allowed actions for this state (i,j)'''
        allowed_actions = []
        #mapping the different successor states to actions
        station_to_action = {(i-1,j):self.action[0], 
                             (i+1,j):self.action[1],
                             (i,j-1):self.action[2], 
                             (i,j+1):self.action[3]} 
        #iterate through the dict 
        for s in station_to_action:
            #s is the successor states
            #check if the successor states is a wall or outside the maze
            if  s[0] < 0 or s[1] > (self.width-1) or s[1] < 0 or s[0] > (self.height-1):
                continue
            elif self.checkwall(np.ravel(s)) == is_wall:
                continue
            else:
                #see if the current state is the goal because then the action 
                #ideal will be taken into account
                if (i,j) == self.Goal:
                    allowed_actions.append(self.action[4])
                else: 
                    pass
                allowed_actions.append(station_to_action[s])
              
        return allowed_actions
        
#        
    def transition_propability(self, allowed_actions, i, j, is_wall=False):
        """Returns a dict with the transition probabilities for each action as 
        a key: we get a tuple with a probability and successor state"""
        # the states that correspond to action a.
        states = {self.action[0]:[(i-1,j-1),(i-1,j),(i-1,j+1)],
                  self.action[1]:[(i+1,j-1),(i+1,j),(i+1,j+1)],
                  self.action[2]:[(i-1,j-1),(i,j-1),(i+1,j-1)],
                  self.action[3]:[(i-1,j+1),(i,j+1),(i+1,j+1)],
                  self.action[4]:[(i-1,j),(i,j),(i,j+1)]}
        #initialize a dict for the transition probabilities   
        trans_prob = {}
        #iterating through the allowed actions
        for a in allowed_actions:
            #for each action a, we get the possible successor states from the dict 'states'
            coordinates = states[a]
            #make a list out of the possible successor states
            cor = list(coordinates)
            #get the possible probabilities for the corresponding action
            #the probabilies are in a dict and were initialized in the beginning
            pr = list(self.prob[a])
            #setting dummy variable
            x = 3
            #iterating through the list of possible successor states
            for k, c in enumerate(cor):
                #check if any of the succcessor states is outside maze
                if  c[0] < 0 or c[1] > (self.width-1) or c[1] < 0 or c[0] > (self.height-1):
                    if k == 2:
                           pr[k-1] = pr[k-1] + pr[k]
                           pr[k] = 0.0
                    else:
                    #if we are in any other iteration and there is a wall then add p
                    #to the second entry in the probability list
                       pr[k+1] = pr[k+1]+ pr[k]
                       pr[k] = 0.0
                    #save indices that is not in maze to del later on
                    x = k
                    continue
                # or is a wall 
                elif self.checkwall(c) == is_wall:
                   #if we are in the third iteration and there is a wall then add p
                   #to the second entry in the probability list
                   if k == 2:
                       pr[k-1] = pr[k-1] + pr[k]
                       pr[k] = 0.0
                   else:
                   #if we are in any other iteration and there is a wall then add p
                   #to the second entry in the probability list
                       pr[k+1] = pr[k+1]+ pr[k]
                       pr[k] = 0.0
                else:
                  # if the state is not a wall, leave the probability as initiated
                   continue
            #the final dict with the change probabilities
            #del the coordinate and probability where the coordinate is outside the maze
            if x == 3:
                trans_prob[a] = [(pr[y],cor[y]) for y in range(len(pr))]
                continue
            else: 
                indices = [y for y in range(len(pr)) if y != x]
                trans_prob[a] = [(pr[y],cor[y]) for y in indices]
            

        return trans_prob
        
    def look_up(self, is_wall=False):
        """Returns a dict with the following form {state:{action:(prob,next_state),....}}
        its a nested dictionary"""
        Transitions = {}
        #iterate through all states in the maze
        for i in range(len(self.maze)):
            for j in range(len(self.maze[1])):
                #check if the current state is a wall
                if self.checkwall((i,j)) == is_wall:
                    continue
                else:
                    #get the list of allowed states
                    actions = self.allowed_actions(i,j)
                    if actions is None:
                        continue
                    else:
                        #get the dictionary with transition prob and put them 
                        # as values in the dictionary of the final Transitions dict
                        trans_prob = self.transition_propability(actions,i,j)
                        Transitions[i,j] = trans_prob
        
        return Transitions
        
    def reward(self, method=1, is_wall=False):
        '''calculating the reward using one of the two methods'''
        """Return a nested dict with the following form
        {state:{(next_state,action): reward....}}"""
        reward= {}
        #iterate through the maze
        for i in range(len(self.maze)):
            for j in range(len(self.maze[1])):
                #check if any state is a wall
                if self.checkwall((i,j)) == is_wall:
                    continue
                else:
                    #get list of allowed actions
                    allowed_actions = self.allowed_actions(i, j)
                    #mapping actions to successor states
                    states = {self.action[0]:[(i-1,j-1),(i-1,j),(i-1,j+1)],
                      self.action[1]:[(i+1,j-1),(i+1,j),(i+1,j+1)],
                      self.action[2]:[(i-1,j-1),(i,j-1),(i+1,j-1)],
                      self.action[3]:[(i-1,j+1),(i,j+1),(i+1,j+1)],
                      self.action[4]:[(i-1,j),(i,j),(i,j+1)]}
                    current_state = (i,j)
                    #iterating through allowed action
                    for a in allowed_actions:
                        #getting the possible successor states from the dict states
                        coordinates = states[a]
                        cor = list(coordinates)
                        for c in cor:
                            if  c[0] < 0 or c[1] > (self.width-1) or c[1] < 0 or c[0] > (self.height-1):
                                continue
                            else:
                                if method == 1:
                                    #checking if the current state is not the goal state and if the successor state is the goal
                                    if not np.any(np.subtract(current_state, self.Goal)) and np.any(np.subtract(c, self.Goal)):
                                       #check if dict has a key of the current state
                                       #if yes them append the reward, action and successor state
                                       #if not generate a new key
                                       if (i,j) in reward:
                                           reward[(i,j)][(c,a)] = 1.0
                                       else:
                                           reward[(i,j)] = {(c,a):1.0}
                                    #checking if the current state and the next state are the goal
                                    elif not np.any(np.subtract(current_state, self.Goal)) and not np.any(np.subtract(c, self.Goal)):
                                       if (i,j) in reward:
                                           reward[(i,j)][(c,a)] = 1.0
                                       else:
                                           reward[(i,j)] = {(c,a):1.0}  
                                    #checking if the current state is the trap
                                    elif not np.any(np.subtract(current_state, self.Trap)) and np.any(np.subtract(c, self.Trap)):
                                        if (i,j)  in reward:
                                            reward[(i,j)][(c,a)] = -50.0
                                        else:
                                           reward[(i,j)] = {(c,a):-50.0}
                                    else:
                                        if (i,j)  in reward:
                                           reward[(i,j)][(c,a)] = 0.0
                                        else:
                                           reward[(i,j)] = {(c,a):0.0}     
                                elif method == 2:
                                    #checking if the current state and the next state are the goal
                                    if not np.any(np.subtract(current_state, self.Goal)) and not np.any(np.subtract(c, self.Goal)):
                                        if (i,j)  in reward:
                                           reward[(i,j)][(c,a)] = 0.0
                                        else:
                                           reward[(i,j)] = {(c,a):0.0}
                                    #checking if the current state is the trap and the successor state is anything else
                                    elif not np.any(np.subtract(current_state, self.Trap)) and np.any(np.subtract(c, self.Trap)):
                                        if (i,j)  in reward:
                                           reward[(i,j)][(c,a)] = -50.0
                                        else:
                                           reward[(i,j)] = {(c,a):-50.0}
                                    else:
                                        if (i,j)  in reward:
                                           reward[(i,j)][(c,a)] = -1.0
                                        else:
                                           reward[(i,j)] = {(c,a):-1.0}
                                else:
                                    print('No Method with that number only 1 or 2')
        return reward

