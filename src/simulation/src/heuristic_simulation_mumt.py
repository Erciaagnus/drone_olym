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
import mavros.setpoint
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
        self.d_min = 4.5 # minimum turning radius
        self.d = 10
        self.r_cs = 3
    def make_cost_matrix(self, obs, m , n):
            keys = [f"uav{uav_id+1}_target{target_id+1}" for uav_id in range(m) for target_id in range(n)]
            observations = np.array([obs[key][0] for key in keys]) # obs[0] distance
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
    def uav1_target1_heuristic2(self, battery, age, r_c, r_t, eps):
        beta = np.arctan((r_t-self.d)/(self.d_min))
        r_charge = self.d_min*2*(np.pi - beta) + r_t - self.d + eps
        r_target = r_t + np.pi*self.d_min + eps*2
        if battery < r_charge:
            action = 0
                # charging   20: 30(closest r_t) - d(10)
        elif (r_c < self.r_cs and 20 < age) and battery < r_target:
            action = 0
        else:
            action = 1
        # print(action, end='')
        # print(f'b: {battery}, a: {action}')
        return action
    # 3번 사용
    def get_action_from_pairs(self, UAV_idx, Target_idx, m, n, b1, b2, a1, obs, eps):
        battery = obs["battery"]
        age = obs["age"]

        action = np.zeros(m, dtype=int)
        for uav_idx, target_idx in zip(UAV_idx, Target_idx):
            # action[uav_idx] = uav1_target1_heuristic(battery[uav_idx], age[target_idx], b1, b2, a1)*(target_idx+1)
            r_c = obs[f"uav{uav_idx+1}_charge_station"][0]
            r_t = obs[f"uav{uav_idx+1}_target{target_idx+1}"][0]
            action[uav_idx] = self.uav1_target1_heuristic2(battery[uav_idx], age[target_idx], r_c, r_t, eps)*(target_idx+1)

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
    def r_t_hungarian(self, obs, m, n, b1=2000, b2=1000, a1=1200):
            bat = obs['battery']
            age = obs['age']
            uav_idx, target_idx = self.hungarian_assignment(self.make_cost_matrix(obs, m, n)) # COST MATRIX 사용 1번, 2번 사용
            action = self.get_action_from_pairs(uav_idx, target_idx, bat, age, m, n, b1, b2, a1)
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
        self.target_pose = ([[100, 150], [-200, 300], [700, 800], [-300, -800], [300, -920]], [0]*self.n)
        env = MUMT_v1(m=self.m, n=self.n)
        obs, _ = env.reset(uav_pose = [[10, 0, 0], [0, 10, 0], [-10, 0, 0]], target_pose = self.target_pose, batteries=[22000, 22000, 22000])
        total_reward = 0
        heuristic = Heuristic()
        truncated =False
        step = 0
        self.target_positions = self.target_pose[0]
        while truncated == False:
            step += 1
            action = heuristic.high_age_first(obs, self.m)
            obs, reward, _, truncated, _ = env.step(action)
            total_reward += reward
            bat = obs['battery']
            age = obs['age']
            print(f"step: {step} | battery: {bat} | reward: {total_reward}")
            print("##### ITRERATION IS FINISHED #####")
            print(f'###################### action is {action}')
            uav_positions = env.uavs[0].state[:2]
            self.uav_positions.append(uav_positions)
            for i in range(self.m):
                for j in range(1, self.n+1):
                    if action[i] == 0:
                        print(f"UAV{i+1} GO TO CHARGE STATION")
                        get_observation = obs[f'uav{i+1}_charge_station']
                        print(f"Observation from {i+1} is {get_observation}")
                    if action[i] == j:
                        print(f"UAV{i+1} GO TO TARGET {j}")
                        get_observation = obs[f'uav{i+1}_target{j}']
                        print(f"Observation from {i+1} is {get_observation}")
            #env.publish_rviz_poses()
            rospy.sleep(0.1)
if __name__ == '__main__':
    simul = Simulation()
    simul.run()