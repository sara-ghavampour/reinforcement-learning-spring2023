import numpy as np
from tqdm import tqdm
from math import *
import random

import gymnasium as gym
from tqdm import trange


     

def generate_episode(env,policy):
    episode=[]
    done = False 
    cummulative_r =0 
    state, _ = env.reset()
    while not done:
        # action = env.action_space.sample()
        action = np.random.choice(np.arange(env.action_space.n), p=policy[state]) 
        next_state, reward, terminated, truncated, info = env.step(action)
        cummulative_r += reward
        episode.append((state,action,reward))
        done = terminated or  truncated 
        state = next_state


    return episode ,cummulative_r
    
def q_learning(env,episodes_n,action_space_n,state_space_n,alpha=0.1,epsilon=0.1,gamma=0.9):
    q  = np.zeros((state_space_n,action_space_n))
    all_episodes_cummulative_r = []
    for ep in tqdm(range(episodes_n)):
        state, _ = env.reset()
        done = False
        cummulative_r =0 
        while not done: # loop for each step of episode 
            if random.random() < epsilon: # choose a from q (eps greedy)
                    action = env.action_space.sample()
            else:
                    action = np.argmax(q[state])

            next_state, reward, terminated, truncated, info = env.step(action)
            # done = terminated or truncated
            cummulative_r+=reward

            target = reward+(gamma*np.max(q[next_state,:]))
            q[state,action]=q[state,action]+alpha*(target-q[state,action])
            state = next_state
            done = terminated or truncated


        all_episodes_cummulative_r.append(cummulative_r)    

    return q,all_episodes_cummulative_r    
            
# def q_learning(env:gym.Env, num_episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards = []
    for _ in trange(num_episodes):
        cum_reward = 0
        state, _ = env.reset()
        done = False
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            cum_reward+=reward
            
            Q_s = q_table[state, action]
            max_Q_s_prime = np.max(q_table[next_state])

            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * max_Q_s_prime - Q_s)
            state = next_state
        rewards.append(cum_reward)
    
    return q_table, rewards
                
    


def onPolicy_first_visit_MC(env,state,episodes_n,action_space_n,state_space_n,alpha=0.1,epsilon=0.1,gamma=0.9):
    q  = np.zeros((state_space_n,action_space_n))
    all_episodes_cummulative_r = []
    policy  = np.ones((state_space_n,action_space_n))/action_space_n
    returns = np.zeros((state_space_n,action_space_n))
    returns_counter = np.zeros((state_space_n,action_space_n))

    for ep in tqdm(range(episodes_n)):
        seen_steps=[]
        episode,cummulative_r=generate_episode(env,policy)
        g =0

        for (state,action,reward) in episode:
            g = (gamma*g) + reward
            if (state,action) not in seen_steps:
                seen_steps.append((state,action))
                returns[state,action]+=g
                returns_counter[state,action]+=1
                q[state,action]=returns[state,action] / returns_counter[state,action]
                a_star  = np.argmax( q[state,:])

                for a in range(env.action_space.n):
                    if a==a_star:
                        policy[state,a]=1-epsilon+(epsilon/action_space_n)
                    else :
                        policy[state,a]=epsilon/action_space_n

        all_episodes_cummulative_r.append(cummulative_r)    
                 

    return policy , q,  all_episodes_cummulative_r           


def sarsa(env,episodes_n,action_space_n,state_space_n,alpha=0.1,epsilon=0.1,gamma=0.9):
    q  = np.zeros((state_space_n,action_space_n))
    all_episodes_cummulative_r = []
    for ep in tqdm(range(episodes_n)):
        state, _ = env.reset()
        done = False
        cummulative_r =0 
        # choose action
        if random.random() < epsilon:
                    action = env.action_space.sample()
        else:
                    action = np.argmax(q[state])
        # loop for each step of ep 
        while not done:

            next_state, reward, terminated, truncated, info = env.step(action)

            if random.random() < epsilon:
                    action_prime = env.action_space.sample()
            else:
                    action_prime = np.argmax(q[next_state])

            # done = terminated or truncated
            cummulative_r+=reward

            target = reward+(gamma*(q[next_state,action_prime]))
            q[state,action]+=alpha*(target-q[state,action])
            state = next_state
            action = action_prime
            done = terminated or truncated


        all_episodes_cummulative_r.append(cummulative_r)    

    return q,all_episodes_cummulative_r
                



