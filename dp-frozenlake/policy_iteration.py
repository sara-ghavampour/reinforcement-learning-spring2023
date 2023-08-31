import gymnasium as gym
import numpy as np
from math import *



def bellman_eq(s,nA,policy,P,value_function,gamma):
    
    s_actions=P[s]
    v = 0
    for a in range(nA):
        for action in P[s][a]:
            p = action[0]
            s_prime = action[1]
            r = action[2]
            done = action[3]
            v+= policy[s, a] *p*(r+gamma*value_function[s_prime])

    return v    

def greedy_bellman_updateRule(s,policy,P,value_function,nA,gamma):
    s_actions=P[s]
    
    action_value_function = []
    for a in s_actions.keys():
        v = 0
        for action in P[s][a]:
            p = action[0]
            s_prime = action[1]
            r = action[2]
            done = action[3]
            v+= p*(r+gamma*value_function[s_prime])
        
        action_value_function.append(v)


    policy[s, :] = 0
    policy[s, np.argmax(action_value_function)] = 1
    return policy


def policy_evaluation(value_function,policy,P, nS, nA, gamma=0.9, tol=1e-4):
    policy_iter_flag=True
    delta = inf
    while delta > tol:
        delta = 0
        for s in range(nS):
            v = value_function[s].copy()
            value_function[s] = bellman_eq(s,nA,policy,P,value_function,gamma)
            delta = max(delta,abs(v-value_function[s]))


    return   value_function       
    

def policy_improvment(policy,P,value_function, nS, nA, gamma=0.9):
    # print('policy_improvment')
    policy_stable = True
    for s in range(nS):
        old_action = policy[s].copy()
        policy= greedy_bellman_updateRule(s,policy,P,value_function,nA,gamma)
        if(not np.array_equal(old_action ,policy[s])):policy_stable = False
   
    return policy,policy_stable

def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-4):

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
    print('policy_iteration')
    value_function = np.zeros(nS)
    #policy = np.zeros(nS, dtype=int)
    policy = np.ones((nS, nA),dtype=int) /  nA
    policy_stable = False

    # Implement policy iteration here #
    while policy_stable==False:
        # print('1: ',value_function)
        value_function = policy_evaluation(value_function,policy,P, nS, nA, gamma=0.9, tol=1e-4)
        # print('2: ',value_function)
        policy,policy_stable = policy_improvment(policy,P,value_function, nS, nA, gamma=0.9)

    return  value_function, policy  

    
    
    #return value_function, policy

if __name__ == "__main__":
    
    # create FrozenLake environment note that we are using a deterministic environment change is_slippery to True to use a stochastic environment
    env = gym.make("FrozenLake-v1", is_slippery=False)
    # reset environment to start state
    env.reset()
    # run policy iteration algorithm
    value_function, policy = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-4)
    



    
    
    





