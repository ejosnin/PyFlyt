#TBS: decision transformer https://github.com/nikhilbarhate99/min-decision-transformer
#look at https://lorenzopieri.com/rl_transformers/ and https://www.reddit.com/r/reinforcementlearning/comments/rpdi1h/decision_transformers_to_replace_conventional_rl/


import pybullet_data
from pybullet_utils import bullet_client
from PyFlyt.core.abstractions import DroneClass, WindFieldClass
from PyFlyt.core.drones import Fixedwing, QuadX, Rocket
#import PyFlyt.gym_envs
import cv2
import pygame
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler
import pygame
import time

import numpy as np

from PyFlyt.gym_envs.fixedwing_envs.fixedwing_waypoints_env import FixedwingWaypointsEnv
#from custompyflyt.lib_quad import QuadXWaypointsEnv
#from custompyflyt.lib_wing import FixedwingWaypointsEnv







#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% init of env
drone_type = 'fixedwing'
agent_hz = 20
dt = 1/agent_hz
use_rc_controller = True
num_targets = 20
goal_reach_distance = 10





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% define env
if use_rc_controller:
    pygame.init()
    controller = pygame.joystick.Joystick(0)
    num_axes = controller.get_numaxes()
    num_buttons = controller.get_numbuttons()
    #initialize gimbals
    pygame.event.get()
    roll_cmd_init = controller.get_axis(0)
    pitch_cmd_init = controller.get_axis(1)
    throtle_cmd_init = 0 #controller.get_axis(2)
    yaw_cmd_init = controller.get_axis(3)  


render_mode = 'rgb_array'


if drone_type=='fixedwing':
    base_mode = 0
    env = FixedwingWaypointsEnv(render_mode=render_mode,flight_dome_size=500,
                            render_resolution= (128*2,128*2),
                            #use_yaw_targets=False,
                            angle_representation = "euler",
                            #start_pos=np.array([[0.0, 0.0, 10.0]]),
                         max_duration_seconds= 10, agent_hz= agent_hz,
                         num_targets= num_targets, goal_reach_distance= goal_reach_distance,)

else: 
    raise BaseException('no case found')




#%%%%%%%%%%%%%%%%%%%%%%%%% simulation loop

in_control = 0
n_epi = 0
while n_epi < 2000:
    t_disp = time.time()
    step_time_list = []
    state_, info = env.reset()



    done = False
    trunc = False
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% loop
    while (not done) and (not trunc):
        
        t_start_cycle = time.time()

            
        if use_rc_controller:
            pygame.event.get()


        if drone_type=='fixedwing':
            roll_cmd = 0.4*controller.get_axis(0) + 0.6*controller.get_axis(0)**3
            pitch_cmd = controller.get_axis(1)
            throtle_cmd = controller.get_axis(2)
            yaw_cmd = controller.get_axis(3)
            
            action = np.array([
                min(1, max(roll_cmd -roll_cmd_init,-1)),
                min(1, max(pitch_cmd - pitch_cmd_init,-1)),
                min(1, max(yaw_cmd - yaw_cmd_init,-1)),
                min(1, max(throtle_cmd,-1))])
            action_ = np.array([action[0], action[1], action[2], action[3]])
        else:
            roll_cmd = 0.7*controller.get_axis(0) + 0.3*controller.get_axis(0)**3
            pitch_cmd = controller.get_axis(1)
            throtle_cmd = controller.get_axis(2)
            yaw_cmd = controller.get_axis(3)
        
            action = np.array([
                min(1, max(roll_cmd -roll_cmd_init,-1)),
                min(1, max(pitch_cmd - pitch_cmd_init,-1)),
                min(1, max(- (yaw_cmd - yaw_cmd_init),-1)),
                min(1, max(throtle_cmd,-1))])
            action_ = action

        next_state_, reward, done, trunc, info = env.step(action_)


        time.sleep(max(0, 1/agent_hz-(time.time()-t_disp)))
        
        RGBA_img = env.env.drones[0].rgbaImg
        RGBA_img[:,:, [2, 0]] = RGBA_img[:,:, [0, 2]]

        img = RGBA_img[:,:,:3]

        cv2.imshow('test', img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            


            
    
        t_end_cycle = time.time()        
        step_time_list += [t_end_cycle - t_start_cycle]
        if len(step_time_list) == agent_hz:

            print('time for 1s calculation = {}'.format(np.array(step_time_list).mean()*agent_hz))
            step_time_list = []  
        
    



