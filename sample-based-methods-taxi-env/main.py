import gymnasium as gym
import numpy as np
from wrapper import *
import matplotlib.pyplot as plt
import random
# from IPython.display import clear_output
from time import sleep
from matplotlib import animation
from learning import onPolicy_first_visit_MC ,sarsa , q_learning


def plot(rewards,title):
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Cummulative Reward')
    plt.title(title)
    title = title+'.png'
    plt.savefig(title)
    plt.show()

def start_game(gamma):
    env = gym.make('Taxi-v3', render_mode='rgb_array')
    env= taxi_wrapper(env=env)
    file_name = 'run'
    # env = record_video.RecordVideo(env, video_folder='runs',name_prefix=file_name)
    
    state, info = env.reset()
    # print('info_reset',info)
    # print('state: ',state)
    
   
  
    done = False 
    cumulative_reward = 0
    time_steps = 0
    policy,q,all_episodes_cummulative_r=onPolicy_first_visit_MC(env,state,10000,env.action_space.n,env.observation_space.n,alpha=0.1,epsilon=0.1,gamma=gamma)
    # q,all_episodes_cummulative_r=sarsa(env,10000,env.action_space.n,env.observation_space.n,alpha=0.1,epsilon=0.1,gamma=gamma)
    # q,all_episodes_cummulative_r =q_learning(env,10000,env.action_space.n,env.observation_space.n,alpha=0.1,epsilon=0.1,gamma=gamma)
    # env = record_video.RecordVideo(env, video_folder='runs',name_prefix=file_name)

    state, info = env.reset()

    while not done:
        # action = env.action_space.sample()
        action = np.argmax(q[state,:])
        # print('astion: ',action)
        next_state, reward, terminated, truncated, info = env.step(action)
        # print('observation, reward, terminated, truncated, info: ',next_state, reward, terminated, truncated, info)
        # test_s = env.s
        # print('test_s: ',test_s)
        done = terminated or  truncated
        cumulative_reward +=reward
        state = next_state
        time_steps+=1

    env.render()
    env.close()    
    return file_name,time_steps,cumulative_reward , all_episodes_cummulative_r

if __name__ == "__main__":  
    gamma=0.9
    title ='onPolicy_first_visit_MC - gamma='+str(gamma)
    _,time_steps,cumulative_reward , all_episodes_cummulative_r=start_game(gamma)
    # print('time_steps,all_episodes_cummulative_r: ',time_steps,' , ',all_episodes_cummulative_r)
    # print(all_episodes_cummulative_r)
    plot(all_episodes_cummulative_r,title=title)