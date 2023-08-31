# import gym 
import gymnasium as gym
from gymnasium.wrappers import record_video
from policy_iteration import policy_iteration
from value_iteration import value_iteration
import numpy as np

class RewardWrapper(gym.Wrapper):
    def __init__(self, env,reward_step=0.0, reward_hole=0.0):
        super().__init__(env)
        self.reward_step=reward_step
        self.reward_hole=reward_hole
        self.hole_set = [5,7,11,12]
    
    def step(self, action):
        observation, reward,terminated, truncated, _ = self.env.step(action)
        if observation == 15: reward= 1.0
        elif   observation in self.hole_set: reward= self.reward_hole
        else: reward = self.reward_step
        
        return observation, reward,terminated, truncated, _

def start(policy_iteration_flag,default_configuration=False,is_slippery=False,gamma=0.9,reward_step=0.0, reward_hole=0.0):

    env = gym.make("FrozenLake-v1", is_slippery=is_slippery,render_mode="rgb_array")
    if not default_configuration: # default_configuration false --> custom rewards
        env= RewardWrapper(env=env,reward_step=reward_step, reward_hole=reward_hole)
    
    if policy_iteration_flag:
            file_name = 'policy_iteration ,is_slippery:'+str(is_slippery) +',gamma:  '+str(gamma)+' ,reward_step: '+str(reward_step)+' ,reward_hole: '+str(reward_hole)
    else:
            file_name = 'value_iteration ,is_slippery:'+str(is_slippery) +',gamma:  '+str(gamma)+' ,reward_step: '+str(reward_step)+' ,reward_hole: '+str(reward_hole)

    env = record_video.RecordVideo(env, video_folder='runs',name_prefix=file_name)
    state,_=env.reset()
    done = False
    cumulative_reward = 0
    if policy_iteration_flag:
        value_function,policy=policy_iteration(P=env.P, nS=env.observation_space.n, nA=env.action_space.n, gamma=0.9, tol=1e-4)   
    else:
        value_function,policy=value_iteration(P=env.P, nS=env.observation_space.n, nA=env.action_space.n, gamma=0.9, tol=1e-4)
    #print(value_function.reshape((4,4)))
    time_steps = 0
    while not done:
        action = np.argmax(policy[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or  truncated
        cumulative_reward +=reward
        state = next_state
        time_steps+=1

    #print('cumulative_reward : ',cumulative_reward)

    
    env.render()
    env.close()
    return file_name,time_steps,cumulative_reward

if __name__ == "__main__":    
    # start(policy_iteration_flag=True,default_configuration=False,is_slippery=True,gamma=0.9,reward_step=0.0, reward_hole=0.0)
    runs = []
    policy_iteration_flag_set=[True,False]
    is_slippery_set=[False,True]
    gamma_set=[0.0,0.9,1.0]
    reward_step_set=[0.0,-0.05]
    reward_hole_set=[0.0,-2.0]

    for policy_iteration_flag in policy_iteration_flag_set:
        for is_slippery in is_slippery_set:
            for gamma in gamma_set:
                for reward_step in reward_step_set:
                    for reward_hole in reward_hole_set:
                        file_name,time_steps,cumulative_reward = start(policy_iteration_flag=policy_iteration_flag
                                                                    ,default_configuration=False,is_slippery=is_slippery,gamma=gamma
                                                                    ,reward_step=reward_step, reward_hole=reward_hole)

                        runs.append((file_name,time_steps,cumulative_reward))


    for idx,run in enumerate(runs):
        print(idx,' ',run[1],' ',run[2])
        





    