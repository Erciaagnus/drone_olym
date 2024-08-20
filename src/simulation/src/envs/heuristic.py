#!/usr/bin/env python3
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
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from mavros_msgs.msg import ActuatorControl, AttitudeTarget, Thrust, State
from mavros_msgs.srv import SetMode, SetModeRequest, CommandBool, CommandBoolRequest, CommandBoolResponse, StreamRate, StreamRateRequest
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
#from simulation.src.envs.sust_v3 import SUST_v3

class Heuristic():
    def __init__(self):
        #TODO(1) Assignment UAVs to Targets to minimize total sum of r(cost matrix)
        # env = gym.make('SUST_v3-v0')
        pass
    def make_cost_matrix(self, obs, m, n):
            keys = [f"uav{uav_id+1}_target{target_id+1}" for uav_id in range(m) for target_id in range(n)]
            observations = np.array([obs[key][0] for key in keys]) # obs[0] distance
            cost_matrix = observations.reshape(m, n)
            return cost_matrix
        #TODO(2) For each Assignment UAV-Target pair - Based on Battery

    def hungarian_assignment(self, cost_matrix):
            uav_idx, target_idx = linear_sum_assignment(cost_matrix)
            return uav_idx, target_idx

        # 1개 UAV 1개 Target일 때만
        #TODO(3) For each Assignment UAV-Target pair - Based on Battery
        # 만든 pair(각각에 대해) 충전/감시 여부 판단하기
    def uav1_target1_heuristic(self, battery, age, b1, b2, a1):
            if battery > b1:
                action =1
            elif battery >b2:
                if age == 0 or age > a1:
                    action = 1
                else:
                    action = 0
            else:
                action = 0
            return action

        #TODO(4) Choosing Unselected UAVs Action
    def get_action_from_pairs(self, UAV_idx, Target_idx, battery, age, m, n, b1, b2, a1):
            # Initialize
            action = np.zeros(m, dtype =int)
            for uav_idx, target_idx in zip(UAV_idx, Target_idx):
                action[uav_idx] = self.uav1_target1_heuristic(battery[uav_idx], age[target_idx], b1, b2, a1)*(target_idx + 1)
            if m>n:
                unselected_uav_idx = np.setdiff1d(np.arrange(m), UAV_idx)
                action[unselected_uav_idx] = 0
            return action

        #TODO(5) Action Return Function : Order of Pair using Hungarian
    def r_t_hungarian(self, obs, m, n, b1=2000, b2=1000, a1=800):
            bat = obs['battery']
            age = obs['age']
            uav_idx, target_idx = self.hungarian_assignment(self.make_cost_matrix(obs, m, n))
            action = self.get_action_from_pairs(uav_idx, target_idx, bat, age, m, n, b1, b2, a1)
            return action

        #TODO(6) Criteria for Age
    def high_age_first(self, obs, m, b3=1000):
        bat = obs['battery']
        age = obs['age']
        uav_list = [uav_idx for uav_idx in range(m)]
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

    def arm_drone(self, arm):
        rospy.wait_for_service('/mavros/cmd/arming')
        try:
            arm_srv= rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
            response = arm_srv(arm)
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def set_mode(self, mode):
        rospy.wait_for_service('/mavros/set_mode')
        try:
            set_mode_srv = rospy.ServiceProxy('/mavros/set_mode', SetMode)
            response = set_mode_srv(0, mode)
            return response.mode_sent
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
    def controlUAV(self, state):
        print(state)
    # def run(self):
    #     self.arm_drone()
    #     self.set_mode("OFFBOARD")
    #     start_time = time.time()
    #     total_time_step = 0
    #     step = 0
    #     truncated = False
    #     args = argparse.Namespace(
    #         b1=2000,
    #         b2=1000,
    #         b3=1000,
    #         a1=1000,
    #         m=1,
    #         n=1,
    #         policy='rt',
    #         seed =0,
    #         target='static',
    #         targetv=0.5
    #     )
    #     env = gym.make('SUST_v3-v0', m = args.m, n=args.n)
    #     obs, _ = env.reset(seed=args.seed, target_type=args.target, sigma_rayleigh=args.targetv)
    #     episode_reward = 0
    #     while truncated == False:
    #         step += 1
    #         if args.policy == 'age':
    #             action = self.high_age_first(obs, 1, 1)
    #             print("Step_{step}_action_{action}")
    #         else:
    #             action = self.r_t_hungarian(obs, 1, 1, b1=args.b1, b2=args.b2, a1=args.a1)
    #             print("Step_{step}_action_{action}")
    #         obs, reward, _, truncated, _ =env.step(action)
    #         episode_reward += reward
    #     self.controlUAV(obs)
    #     total_time_step += step
    #     reward_per_step = episode_reward / total_time_step
    #     end_time = time.time()
    #     execution_time = end_time - start_time

# if __name__ == '__main__':
    # heuristic = Heuristic()
    # heuristic.run()