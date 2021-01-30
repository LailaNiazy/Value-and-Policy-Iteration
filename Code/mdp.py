
import numpy as np
import pickle
import plotting as pl

    
class MarkovDecisionProcess:

    """A Markov Decision Process, defined by an states, actions, transition model and reward function."""

    def __init__(self,  transition={}, reward={}, method = 1, gamma=.99, epsilon = 0.001):
        ###########collect all nodes from the transition models
        self.states = transition.keys()
        #initialize transition
        self.transition = transition
        #initialize reward
        self.reward = reward
        #initialize gamma with gamma being the discount factor
        self.gamma = gamma
        #initialize reward method
        self.method = method
        #the maximum error allowed in the utility of any state
        self.epsilon = epsilon

    def R(self, state, next_state_action):
        """return reward for this state."""
        return self.reward[state][next_state_action]

    def actions(self, state):
        """return set of actions that can be performed in this state"""
        return self.transition[state].keys()

    def T(self, state, action):
        """for a state and an action, return a list of (probability, result-state) pairs."""
        return self.transition[state][action]
        


    def value_iteration(self,maze):
        """
        Solving the MDP by value iteration.
        returns utility values for states after convergence
        """
        #depending on the reward method, a different ground truth will be loaded to calculate the error
        if self.method == 1:
           with open("Ground_Truth_V1", "rb") as f:
               V_optimal = pickle.load(f)
        else:
           with open("Ground_Truth_V2", "rb") as f:
               V_optimal = pickle.load(f) 
        #initialize error to plot
        error = []
        #initialize value of all the states to 0 (this is k=0 case)
        l = [(p1, p2) for p1 in range(len(maze)) for p2 in range(len(maze[1]))]
        V1 = {s: 0.0 for s in l}
        iterations = 0
        while True:
            iterations += 1
            V = V1.copy()
            delta = 0.0
            for s in self.states:
                #Bellman update, update the utility values
                V1[s] = max([sum([p*(self.R(s,(s1,a)) + self.gamma * V[s1]) for (p, s1) in self.T(s, a)]) for a in self.actions(s)])
                #calculate maximum difference in value
                delta = max(delta, abs(V1[s] - V[s]))
            V_now = pl.V_to_matrix(V1,len(maze),len(maze[0]))
            error.append(np.linalg.norm(V_optimal-V_now))
            #check for convergence, if values converged then return V
            if delta < self.epsilon* (1.0 - self.gamma) / self.gamma:
                return V, error
   #     

    def best_policy(self,V):
        """
        Given an MDP and a utility values V, determine the best policy as a mapping from state to action.
        returns policies which is dictionary of the form {state1: action1, state2: action2}
        """
        pi = {}
        for s in self.states:
            pi[s] = max(self.actions(s), key=lambda a: self.expected_utility(a, s, V))
        return pi

    
    def expected_utility(self, a, s, V):
        """returns the expected utility of doing a in state s, according to the MDP and V."""
        return sum([p*(self.R(s,(s1,a)) + self.gamma * V[s1]) for (p, s1) in self.T(s, a)])
    
    
    def policy_iteration(self,maze):
        "Solve an MDP by policy iteration"
        #depending on the reward method, a different ground truth will be loaded to calculate the error
        if self.method == 1:
           with open("Ground_Truth_U1", "rb") as f:
               U_optimal = pickle.load(f)
        else:
           with open("Ground_Truth_U2", "rb") as f:
               U_optimal = pickle.load(f)    
        #initialize the policy randomly from the allowed actions of each states
        pi = dict([(s, np.random.choice(self.actions(s))) for s in self.states])
        iterations_policy = 0
        error = []
        while True:
            iterations_policy += 1
            U = self.policy_evaluation(pi, maze)
            unchanged = True
            for s in self.states:
                a = []
                for e in self.actions(s):
                    a.append(self.expected_utility(e,s,U))
                a = np.argmax(a)
                action = self.actions(s)[a]
                if action != pi[s]:
                    pi[s] = action
                    unchanged = False
            U_now = pl.V_to_matrix(U,len(maze),len(maze[0]))
            error.append(np.linalg.norm(U_optimal-U_now))
            if unchanged:
                return error, pi, U

    def policy_evaluation(self,pi, maze):
        """Return an updated utility mapping U from each state in the MDP to its
        utility, using an approximation (modified policy iteration)."""
        l = [(p1, p2) for p1 in range(len(maze)) for p2 in range(len(maze[1]))]
        U1 = {s: 0.0 for s in l}
        while True:
            U = U1.copy()
            delta = 0.0
            for s in self.states:
               #Bellman update, update the utility values
                U1[s] = sum([p*(self.R(s,(s1,pi[s])) + self.gamma * U[s1]) for (p, s1) in self.T(s, pi[s])])
                #calculate maximum difference in value
                delta = max(delta, np.abs(U1[s] - U[s]))
            #check for convergence, if values converged then return U
            if delta <  self.epsilon* (1.0 - self.gamma) / self.gamma:
                return U
    


   







    
