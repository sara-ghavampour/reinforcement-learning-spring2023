import gymnasium as gym
import numpy as np
from math import *
from policy_iteration import greedy_bellman_updateRule


def bellman_optimality_UPdateRule(s,nA,P,value_function,gamma):
    
    s_actions=P[s]
    v_set=[]
    
    for a in range(nA):
        v = 0
        for action in P[s][a]:
            p = action[0]
            s_prime = action[1]
            r = action[2]
            done = action[3]
            v+= p*(r+gamma*value_function[s_prime])

        v_set.append(v)
    return max(v_set)   


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-4):
    '''
    parameters:
        P: transition probability matrix
        nS: number of states
        nA: number of actions
        gamma: discount factor
        tol: tolerance for convergence
    returns:
        value_function: value function for each state
        policy: policy for each state
    '''
    # initialize value function and policy
    print('value_iteration')
    value_function = np.zeros(nS)
    policy = np.ones((nS, nA),dtype=int) /  nA

    # Implement value iteration here #
    delta = inf
    while delta > tol:
        delta = 0
        for s in range(nS):
            v = value_function[s].copy()
            value_function[s] = bellman_optimality_UPdateRule(s,nA,P,value_function,gamma)
            delta = max(delta,abs(v-value_function[s]))

    for s in range(nS):
        policy = greedy_bellman_updateRule(s,policy,P,value_function,nA,gamma)

    return value_function, policy

if __name__ == "__main__":

    # create FrozenLake environment note that we are using a deterministic environment change is_slippery to True to use a stochastic environment
    env = gym.make("FrozenLake-v1", is_slippery=False)
    # reset environment to start state
    env.reset()
    # run value iteration algorithm
    value_function, policy = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-4)