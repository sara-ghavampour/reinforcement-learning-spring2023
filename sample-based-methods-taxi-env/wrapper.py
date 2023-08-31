from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium.core import Env
from gymnasium.spaces import Discrete
from gymnasium.wrappers import record_video
import numpy as np


# 0: Move south (down)

# 1: Move north (up)

# 2: Move east (right)

# 3: Move west (left)

# 4: Pickup passenger

# 5: Drop off passenger

class taxi_wrapper(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        # print('hi')
        self.env.action_space = Discrete(10)
        self.transition_matrix = self.env.P
        print(self.env.action_space)

    
    
    def give_open_path(self,curr_state,first_half,second_half):
        intermediate_state_a1 = self.transition_matrix[curr_state][first_half][0][1]            
        intermediate_state_a2 = self.transition_matrix[curr_state][second_half][0][1] 
        diagnal_move_possible = False 

        if intermediate_state_a1 != curr_state:
            if self.transition_matrix[intermediate_state_a1][second_half][0][1] != intermediate_state_a1 :
                diagnal_move_possible = True
                return diagnal_move_possible,first_half,second_half
            
        if intermediate_state_a2 != curr_state:
            if self.transition_matrix[intermediate_state_a2][first_half][0][1] != intermediate_state_a2 :
                diagnal_move_possible = True
                return diagnal_move_possible,second_half,first_half
            
        return diagnal_move_possible,second_half,first_half     #???
    
    
    
    
    def step(self, action):
        curr_state = self.env.s
        

        if action < 6:
            observation, reward, terminated, truncated, info = self.env.step(action)

        else:
            if action==6: # north east = (right+up)
                first_half = 1
                second_half = 2

            elif action==7: # north west = (left+up)
                first_half = 1
                second_half =3  

            elif action==8: # south east = (right+down)
                first_half = 0
                second_half = 2   

            elif action==9: # south west = (left+down)
                first_half = 0
                second_half = 3  

            diagnal_move_possible,first_half,second_half =  self.give_open_path(curr_state,first_half,second_half)   
            if diagnal_move_possible:
                observation, reward, terminated, truncated, info = self.env.step(first_half)
                observation, reward, terminated, truncated, info = self.env.step(second_half)

            else : # hold still
                observation, reward, terminated, truncated, info = curr_state, -1 , False , False ,{}


             
        return observation, reward, terminated, truncated, info
    

    
    


