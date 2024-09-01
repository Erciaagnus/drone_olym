#!/usr/bin/env python3
#
# Maintainer : JeongHyeok Lim
# E-mail : henricus0973@korea.ac.kr
import math
import numpy as np
import os
import sys
import time
import argparse

current_file_path = os.path.dirname(__file__)
gym_setting_path = os.path.join(current_file_path, '../../gym_setting/mdp')
sys.path.append(os.path.abspath(gym_setting_path))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from mdp import States, Actions, Surveillance_Actions, Rewards, StateTransitionProbability, Policy, MarkovDecisionProcess

# This is the solution using Heuristic Method
import smach
from smach import State as SmachState
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3, PoseWithCovarianceStamped
from mavros_msgs.msg import ActuatorControl, AttitudeTarget, Thrust
from mavros_msgs.msg import State as MavrosState
from mavros_msgs.srv import SetMode, SetModeRequest, CommandTOL, CommandBool, CommandBoolRequest, CommandBoolResponse, StreamRate, StreamRateRequest
from sensor_msgs.msg import Imu, BatteryState
from std_msgs.msg import Header, Float64
from std_srvs.srv import Empty, EmptyRequest
#import mavros.setpoint
from scipy import sparse as sp
from scipy.stats.qmc import MultivariateNormalQMC
from scipy.optimize import linear_sum_assignment
import rospy
import gym
from gym import Env
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from math import sin, cos, pi
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy import arctan2, array
from typing import Optional
from matplotlib import pyplot as plt
import random
import pandas as pdf
from simulation.src.envs.mumt_v1 import MUMT_v1
import csv
class Heuristic():
    def __init__(self):
        #env=gym.make('SUST_v3-v0')
        self.d_min = 30 # minimum turning radius
        self.d = 40 # keeping Distance
        self.r_cs = 10 # Charging Radius
        self.C_rate = 2
        self.D_rate = 0.41 # 1/2.442 (Battery Runtime)
        self.Q = 22_000 #[mAh] Battery Capacity
        self.v = 17 # [m/s]
    def make_cost_matrix(self, obs, m , n):
        # Create a list of keys for all UAV-target pairs
            keys = [f"uav{uav_id+1}_target{target_id+1}" for uav_id in range(m) for target_id in range(n)]
        # Extract the relevant observations and convert to a Numpy array
            observations = np.array([obs[key][0] for key in keys]) # obs[0] distance
        # Reshape the array into the cost matrix
            cost_matrix = observations.reshape(m, n)
            return cost_matrix
    def hungarian_assignment(self, cost_matrix):
            uav_idx, target_idx = linear_sum_assignment(cost_matrix)
            return uav_idx, target_idx
    def uav1_target1_heuristic(self, battery, age, b1, b2, a1):
            if battery > b1:
                action = 1
            elif battery >b2:
                if age == 0 or age > a1:
                    action = 1
                else:
                    action = 0
            else:
                action = 0
            return action
    def uav1_target1_heuristic2(self, battery, r_c, r_t, eps1, eps2):
        '''
                calculate the battery needed to return/surveil.
                 d_min : minimum turning rate
                d : minimum keeping distance radius
        '''
        beta = np.arctan((r_t-self.d)/(self.d_min))
        r_charge = self.d_min*2*(np.pi - beta) + r_c + eps1
        r_target = 2*np.pi*(self.d_min+self.d) + 2*(r_t) + eps2
        battery_to_charge = r_charge /self.v*(self.D_rate*self.Q/3600)
        battery_to_target = r_target / self.v*(self.D_rate*self.Q/3600)
        if battery < battery_to_charge:
            action = 0
        elif battery> battery_to_target:
            action = 1
        # else:
        #     action = -1 ???
        return action
    # 3번 사용
    def get_action_from_pairs(self, UAV_idx, Target_idx, m, n, obs, eps1, eps2):
        battery = obs["battery"]
        action = np.zeros(m, dtype=int)
        for uav_idx, target_idx in zip(UAV_idx, Target_idx):
            # action[uav_idx] = uav1_target1_heuristic(battery[uav_idx], age[target_idx], b1, b2, a1)*(target_idx+1)
            r_c = obs[f"uav{uav_idx+1}_charge_station"][0]
            r_t = obs[f"uav{uav_idx+1}_target{target_idx+1}"][0]
            u1t1_heuristic_action = self.uav1_target1_heuristic2(battery[uav_idx], r_c, r_t, eps1, eps2)
            action[uav_idx] = -1 if u1t1_heuristic_action == -1 else u1t1_heuristic_action*(target_idx+1)
        if m > n: # in case of m > n: unselected uav stay charge even full battery
            unselected_uav_idx = np.setdiff1d(np.arange(m), UAV_idx) # returns the unique values in array1 that are not present in array2
            action[unselected_uav_idx] = 0
        return action
    # def get_action_from_pairs(self, UAV_idx, Target_idx, battery, age, m, n, b1, b2, a1):
    #         # Initialize
    #         action = np.zeros(m, dtype =int)
    #         for uav_idx, target_idx in zip(UAV_idx, Target_idx):
    #             action[uav_idx] = self.uav1_target1_heuristic(battery[uav_idx], age[target_idx], b1, b2, a1)*(target_idx + 1)
    #         if m>n:
    #             unselected_uav_idx = np.setdiff1d(np.arrange(m), UAV_idx)
    #             action[unselected_uav_idx] = 0
    #         return action
    # Get action 1번
    def r_t_hungarian(self, obs, m, n, eps1=0, eps2=0):
        uav_idx, target_idx = self.hungarian_assignment(self.make_cost_matrix(obs, m, n))
        action = self.get_action_from_pairs(uav_idx, target_idx, m, n, obs, eps1, eps2)
        return action

        #TODO(6) Criteria for Age

    # Get action 2번
    def high_age_first(self, obs, m, b3=1000):
        bat = obs['battery']
        age = obs['age']
        uav_list = [uav_idx for uav_idx in range(m)]
        # 여기서 action이란 무엇일까?
        action = np.zeros(m, dtype = int)
        for uav_idx in range(m):
            if bat[uav_idx] < b3:
                uav_list.remove(uav_idx)
        sorted_age_indices = np.argsort(age)[::-1]
        for target_idx in sorted_age_indices:
            closest_uav_idx = None
            closest_distance = float('inf')
            if uav_list == []:
                pass
            else:
                for uav_idx in uav_list:
                    distance = obs[f"uav{uav_idx + 1}_target{target_idx+1}"][0]
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_uav_idx = uav_idx
                action[closest_uav_idx] = target_idx + 1
                uav_list.remove(closest_uav_idx)
        return action


class Simulation():
    def __init__(self):
        self.m = 3 # UAV 개수 -- > 바꾸기.. 나중에 한 번에 시뮬레이션 하고 Graph 그리는 툴 필요
        self.n = 5 # FIXED!!
        #Parameter
        self.eps1 = 21 #Parameter 1
        self.eps2 = 120 #Parameter 2
        self.uav_positions = []
        self.target_positions = []
    def save_positions_to_csv(self):
        with open('uav_positions.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y"])
            writer.writerows(self.uav_positions)
        with open('target_positions.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y"])
            writer.writerows(self.target_positions)

    def run(self):
        self.target_pose = ([[800, 1430], [-1180, 700], [1200, 1300], [-1100, -1200], [800, -1120]], [0]*self.n)
        env = MUMT_v1(m=self.m, n=self.n)
        start_time = time.time()
        obs, _ = env.reset(uav_pose = None, target_pose = self.target_pose, batteries=[22000, 22000, 22000])
        heuristic = Heuristic()
        truncated =False
        total_time_step = 0
        step = 0
        episode_reward = 0
        self.target_positions = self.target_pose[0]
        while truncated == False:
            step += 1
            #action = heuristic.high_age_first(obs, self.m)
            action = heuristic.r_t_hungarian(obs, self.m, self.n, eps1 = self.eps1, eps2 = self.eps2)
            obs, reward, _, truncated, _ = env.step(action)
            episode_reward += reward
            bat = obs['battery']
            age = obs['age']
            print(f"step: {step} | battery: {bat} | reward: {episode_reward}")
            print("##### ITRERATION IS FINISHED #####")
            #print(f'###################### action is {action}')
            uav_positions = env.uavs[0].state[:2]
            self.uav_positions.append(uav_positions)
            for i in range(self.m):
                for j in range(1, self.n+1):
                    if action[i] == 0:
                        print(f"UAV{i+1} GO TO CHARGE STATION")
                        get_observation = obs[f'uav{i+1}_charge_station']
                        print(f"Observation from {i+1} is {get_observation}")
                    if action[i] == j:
                        print(f"UAV{i+1} GO TO TARGET {j}:: {self.target_positions[j-1]}")
                        get_observation = obs[f'uav{i+1}_target{j}']
                        print(f"Observation from {i+1} is {get_observation}")
            #env.publish_rviz_poses()
            rospy.sleep(0.1)
        total_time_step += step
        reward_per_step = episode_reward / total_time_step
        end_time = time.time()
        excution_time = end_time - start_time
        print(f"Reward per step :: {reward_per_step}, Excution Time :: {excution_time}")
if __name__ == '__main__':
    simul = Simulation()
    simul.run()