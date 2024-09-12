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
import threading
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
from tabulate import tabulate
HIGHER_LEVEL_FREQUENCY = 1
class Heuristic():
    def __init__(self):
        #env=gym.make('SUST_v3-v0')
        self.d_min = 10 # minimum turning radius
        self.d = 40 # keeping Distance
        self.r_cs = 10 # Charging Radius
        self.C_rate = 2
        self.D_rate = 0.41 # 1/2.442 (Battery Runtime)
        self.Q = 22_000 #[mAh] Battery Capacity
        self.v = 17 # [m/s]
    def make_cost_matrix(self, obs, m , n, w_d = 0.05, w_age= 0.5):
        # Create a list of keys for all UAV-target pairs
            keys = [f"uav{uav_id+1}_target{target_id+1}" for uav_id in range(m) for target_id in range(n)]
        # Extract the relevant observations and convert to a Numpy array
            distances = np.array([obs[key][0] for key in keys]) # obs[0] distance
        # Reshape the array into the cost matrix
            distances_matrix = distances.reshape(m, n)
            #print(f"distance_matrix{distances_matrix}")
            ages = np.array([obs["age"]])
            age_matrix = np.tile(ages, (m,1))
            #print(f"age_matrix{age_matrix}")
            cost_matrix = w_d*distances_matrix + w_age*age_matrix
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
    def uav1_target1_heuristic2(self, battery, r_c, r_t, eps1, eps2, duration_time):
        '''
                calculate the battery needed to return/surveil.
                 d_min : minimum turning rate
                d : minimum keeping distance radius
        '''
        beta = np.arctan((r_t-self.d)/(self.d_min))
        r_charge = self.d_min*2*(np.pi - beta) + r_c + eps1
        r_target = 2*np.pi*(self.d_min+self.d) + 2*(r_t) + eps2
        battery_to_charge = r_charge /(self.v*0.6)*(self.D_rate*self.Q/3600) # 1초당이야. -> 근데 우리는 step별 real time으로 할거라서.
        #print("Charging Distance", r_c)
        #print("Battery to Charge Station", battery_to_charge)
        battery_to_target = r_target / (self.v*0.6)*(self.D_rate*self.Q/3600)
        if battery < battery_to_charge:
            action = 0
        elif battery> battery_to_target:
            action = 1
        else:
            action = -1 # 애매할 때,
        return action
    # 3번 사용
    def get_action_from_pairs(self, UAV_idx, Target_idx, m, n, obs, eps1, eps2, duration_time):
        #print(f"PAIRS : [UAV{UAV_idx}:Target{Target_idx}]")
        battery = obs["battery"]
        action = np.zeros(m, dtype=int)
        for uav_idx, target_idx in zip(UAV_idx, Target_idx):
            # action[uav_idx] = uav1_target1_heuristic(battery[uav_idx], age[target_idx], b1, b2, a1)*(target_idx+1)
            #print(f"UAV {uav_idx} distance from CHARGE STATION",  obs[f"uav{uav_idx+1}_charge_station"][0])
            r_c = obs[f"uav{uav_idx+1}_charge_station"][0]
            r_t = obs[f"uav{uav_idx+1}_target{target_idx+1}"][0]
            u1t1_heuristic_action = self.uav1_target1_heuristic2(battery[uav_idx], r_c, r_t, eps1, eps2, duration_time)
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
    def r_t_hungarian(self, obs, m, n, eps1, eps2, duration_time):
        uav_idx, target_idx = self.hungarian_assignment(self.make_cost_matrix(obs, m, n))
        action = self.get_action_from_pairs(uav_idx, target_idx, m, n, obs, eps1, eps2, duration_time)
        #print(f"actions{action}")
        return action

        #TODO(6) Criteria for Age


class Simulation():
    def __init__(self):
        self.m = 3 # UAV 개수 -- > 바꾸기.. 나중에 한 번에 시뮬레이션 하고 Graph 그리는 툴 필요
        self.n = 5 # FIXED!!
        #Parameter
        self.eps1 = 21 #Parameter 1
        self.eps2 = 120 #Parameter 2
        self.uav_positions = []
        self.target_positions = []
    # def save_positions_to_csv(self):
    #     with open('uav_positions.csv', mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["x", "y"])
    #         writer.writerows(self.uav_positions)
    #     with open('target_positions.csv', mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["x", "y"])
    #         writer.writerows(self.target_positions)
    def print_uav_info(self, uav_id, location, observation, bat):
        if location == "charge_station":
            destination = "CHARGE STATION"
            position = "[0, 0]"
            #print("SUCCESSFUL")
        else:
            destination = f"TARGET {location}"
            #position = self.target_positions[location-1]
            position = f"[{self.target_positions[location-1][0]:.2f}, {self.target_positions[location-1][1]:.2f}]"
        distance = float(f"{observation[0]:.3f}") #round(observation[0], 3)
        angle = float(f"{observation[1]:.3f}")#round(observation[1], 3)
        #print(f"test ######### {distance}, {angle}")
        return [uav_id, destination, position, distance, angle, bat]

    def run(self):
        #self.target_pose = ([[800, 1430], [-1180, 700], [1200, 1300], [-1100, -1200], [800, -1120]], [0]*self.n)
        env = MUMT_v1(m=self.m, n=self.n, seed=1)
        #start_time = time.time()
        obs, _ = env.reset(uav_pose = None, target_pose = None, batteries=[22000, 22000, 22000])
        # self.target_pose = ([[env.targets[i].state[0], env.targets[i].state[1]]],[env.targets[i].initial_beta] for i in range(self.n))
        self.target_pose = ([[env.targets[i].state[0], env.targets[i].state[1]] for i in range(self.n)],
                    [env.targets[i].initial_beta for i in range(self.n)])
        heuristic = Heuristic()
        truncated =False
        total_time_step = 0
        step = 0
        episode_reward = 0
        self.target_positions = self.target_pose[0]
        total_time = 0
        rate = rospy.Rate(20)
        threads = []
        # for uav_idx in range(len(env.uavs)):
        #     t = threading.Thread(target=env.publish_attitude, args=(uav_idx,))
        #     t.daemon = True
        #     t.start()
        while truncated == False:
            start_time = env.clock
            step += 1
            total_time += env.duration_time
            #action = heuristic.high_age_first(obs, self.m)
            action = heuristic.r_t_hungarian(obs, self.m, self.n, eps1 = self.eps1, eps2 = self.eps2, duration_time = env.duration_time)
            obs, reward, _, truncated, _ = env.step(action)
            episode_reward += reward
            bat = obs["battery"]
            age = obs['age']
            #float(f"{observation[1]:.3f}
            print(f"step: {step} | total time {total_time} | age : {age}| reward: {episode_reward:.3f}")
            #print("##### ITRERATION IS FINISHED #####")
            #print(f'###################### action is {action}')
            uav_positions = env.uavs[0].state[:2]
            # self.uav_positions.append(uav_positions)
            table_data = []
            added_uavs = set()  # 이미 표에 추가된 UAV를 기록하는 집합
            for i in range(self.m):
                if action[i] == -1:
                    print(f"UAV{i} OOPS, USING PREVIOUS VALUE")
                    #print("#### ", env.uavs[i].previous_lower_action)
                    action[i] = env.uavs[i].previous_lower_action
                    #print(f"UAV{i} is action {action[i]}")
                if action[i] == 0:
                        # print(f"UAV{i+1} GO TO CHARGE STATION")
                        get_observation = obs[f'uav{i+1}_charge_station']
                        if i+1 not in added_uavs:
                            table_data.append(self.print_uav_info(i+1, "charge_station", get_observation, bat[i]))
                            added_uavs.add(i+1)
                else:
                    j = action[i]
                    # print(f"[UAV{i+1} : TARGET {j}] :: {self.target_positions[j-1]}")
                    get_observation = obs[f'uav{i+1}_target{j}']
                    table_data.append(self.print_uav_info(i+1, j, get_observation, bat[i]))
                #env.publish_rviz_poses()
            print(tabulate(table_data, headers=["UAV ID", "Destination", "Target Position", "Distance", "Angle", "Battery"], tablefmt="pretty"))
            rate.sleep()
        #end_time = time.time()
        total_time_step += step
        reward_per_step = episode_reward / total_time_step
        end_time = env.clock
        excution_time = end_time - start_time # Total Step
        reward_per_real_step = episode_reward / excution_time
        print(f"Reward per step :: {reward_per_step} || Excution Time(Total Step Time) :: {excution_time} || Simulator Reward per Step :: {reward_per_real_step}")
if __name__ == '__main__':
    simul = Simulation()
    simul.run()