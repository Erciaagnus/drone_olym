#!/usr/bin/env python3
import math
import numpy as np
import os
import sys
import time
import argparse

current_file_path = os.path.dirname(__file__)
gym_setting_path = os.path.join(current_file_path, '../gym_setting/mdp')
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
import pandas as pd
import gymPX4.envs.MUMT_v5

#import rendering
class HeuristicGazebo():
    def __init__(self):
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_cb, queue_size=10)
        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.imu_cb, queue_size=10)
        self.local_pos_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.lp_cb, queue_size=10)
        self.local_vel_sub = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self.lv_cb, queue_size=10)
        self.act_control_sub = rospy.Subscriber("/mavros/act_control/act_control_pub", ActuatorControl, self.act_cb, queue_size=10)
        self.global_alt_sub = rospy.Subscriber("/mavros/global_position/rel_alt", Float64, self.ra_cb, queue_size=10)
        self.global_pos_sub = rospy.Subscriber("/mavros/global_position/gp_vel", TwistStamped, self.gv_cb, queue_size=10)
        self.battery_state_sub = rospy.Subscriber("/mavros/battery", BatteryState, self.bs_cb)

       ## ROS Publisher
        self.local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local",PoseStamped,queue_size=10)
        self.mocap_pos_pub = rospy.Publisher("/mavros/mocap/pose",PoseStamped,queue_size=10)
        self.acutator_control_pub = rospy.Publisher("/mavros/actuator_control",ActuatorControl,queue_size=10)
        self.setpoint_raw_pub = rospy.Publisher("/mavros/setpoint_raw/attitude",AttitudeTarget,queue_size=10)
        self.thrust_pub = rospy.Publisher("/mavros/setpoint_attitude/thrust",Thrust,queue_size=10)

    ## CALLBACK Function ##
    def bs_cb(self, data):
        self.battery_state = data
    def ra_cb(self, data):
        self.relative_altitude = data
    def state_cb(self, data):
        self.state = data
    def lp_cb(self, data):
        self.local_position = data
    def lv_cb(self, data):
        self.local_velocity = data
    def imu_cb(self, data):
        self.imu_data = data
    def act_cb(self, data):
        self.act_controls = data
    def gv_cb(self, data):
        self.global_velocity = data
    #TODO(0) Initialize Cost Matrix
    def make_cost_matrix(self, obs, m, n):
        # Create a list of keys for all UAV-target pairs, m uav, n targets
        keys = [f"uav{uav_id+1}_target{target_id+1}" for uav_id in range(m) for target_id in range(n)]
        # Extract the relevant observations and convert to a Numpy array
        observations = np.array([obs[key][0] for key in keys])

        # Reshape the array into the cost matrix
        cost_matrix = observations.reshape(m,n) # m x n matrix
        return cost_matrix
    #TODO(1) Assignment UAVs to Targets to minimize total sum of r(cost_matrix)
    # testing env: heuristic policy
    def hungarian_assignment(self, cost_matrix): # 여러 UAV와 여러 타겟 사이 최적 매칭을 수행하기 위해 사용
        uav_idx, target_idx = linear_sum_assignment(cost_matrix)
        return uav_idx, target_idx
    #TODO(2) For each Assignment UAV-Target pair - Based on Battery
    def uav1_target1_heuristic(self, battery, age, b1, b2, a1): # 배터리 상태와 타겟 나이를 기반으로 UAV 행동 결정
        if battery > b1:
            action = 1
        elif battery > b2:
            # previous action = obs[previous action]
            if age == 0 or age > a1:
                action =1
            else:
                action =0
        else:
            action =0
        return action

    #TODO(3) Choosing Unselected UAVs Action
    def get_action_from_pairs(self, UAV_idx, Target_idx, battery, age, m, n, b1, b2, a1):
        action = np.zeros(m, dtype = int)
        for uav_idx, target_idx in zip(UAV_idx, Target_idx):
            action[uav_idx] = self.uav1_target1_heuristic(battery[uav_idx], age[target_idx], b1, b2, a1)*(target_idx + 1)

        if m > n: # in case of m>n (uav > targets) : unselected uav stay charge even full battery
            unselected_uav_idx = np.setdiff1d(np.arrange(m), UAV_idx)
            action[unselected_uav_idx] = 0
        return action
    #TODO(4) Action Return Function : Order of Pair using Hungarian
    def r_t_hungarian(self, obs, m, n, b1=2000, b2=1000, a1= 800):
        bat = obs['battery'] # using keys
        age = obs['age'] # using age
        uav_idx, target_idx = self.hungarian_assignment(self.make_cost_matrix(obs, m, n))
        action = self.get_action_from_pairs(uav_idx, target_idx, bat, age, m, n, b1, b2, a1)
        return action
    #TODO(5) Action Return Function : Order of Age
    def high_age_first(self, obs, m, b3=1000): # Age 높은 순서대로 UAV 할당하는 알고리즘
        bat= obs['battery']
        age = obs['age']
        uav_list = [uav_idx for uav_idx in range(m)]
        action = np.zeros(m, dtype=int)
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
                    # Find the closest UAV to the target
                    distance = obs[f"uav{uav_idx+1}_target{target_idx+1}"][0]
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_uav_idx = uav_idx
                action[closest_uav_idx] = target_idx+1
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
    def run(self):
        #env=gym.make('MUMT_v5', m=1, n=1)
        start_time=time.time()
        total_time_step = 0
        step = 0
        truncated =False
        # parser = argparse.ArgumentParser()
        # parser.add_argument(
        #     "-b1",
        #     type=int,
        #     required=False,
        #     default=2000,
        #     help="battery threshold 1",
        # )
        # parser.add_argument(
        #     "-b2",
        #     type=int,
        #     required=False,
        #     default=1000,
        #     help="battery threshold 2",
        # )
        # parser.add_argument(
        #     "-b3",
        #     type=int,
        #     required=False,
        #     default=1000,
        #     help="battery threshold 3",
        # )
        # parser.add_argument(
        #     "-a1",
        #     type=int,
        #     required=False,
        #     default=1000,
        #     help="age threshold 1",
        # )
        # parser.add_argument(
        #     "-m",
        #     type=int,
        #     required=True,
        #     help="number of UAVs",
        # )
        # parser.add_argument(
        #     "-n",
        #     type=int,
        #     required=True,
        #     help="number of Targets",
        # )
        # parser.add_argument(
        #     "-p",
        #     "--policy",
        #     type=str,
        #     required=False,
        #     default="rt",
        #     help="whether r_t_hungarian(rt) or high_age_first(age)",
        # )
        # parser.add_argument(
        #     "--seed",
        #     "-s",
        #     type=int,
        #     required=False,
        #     default=0,
        #     help="setting seed",
        # )
        # parser.add_argument(
        #     "--target",
        #     "-t",
        #     type=str,
        #     required=False,
        #     default='static',
        #     help="static or rayleigh or deterministic",
        # )
        # parser.add_argument(
        #     "--targetv",
        #     "-tv",
        #     type=float,
        #     required=False,
        #     default=0.5,
        #     help="sigma for rayleigh distribution of target speed",
        # )
        # args = parser.parse_args()
        args = argparse.Namespace(
            b1=2000,
            b2=1000,
            b3=1000,
            a1=1000,
            m=1,
            n=1,
            policy= 'rt',
            seed = 0,
            target='static',
            targetv=0.5
        )
        env = gym.make('MUMT_v5-v0', m=args.m, n=args.n)

        start_time = time.time()
        total_time_step = 0
        step = 0
        truncated = False
        obs, _ = env.reset(seed=args.seed, target_type=args.target, sigma_rayleigh=args.targetv)
        episode_reward = 0
        while truncated == False:
            step += 1
            if args.policy == 'age':
                #action = self.high_age_first(obs, args.m, args.b3)
                action = self.high_age_first(obs, 1, 1)

            else:
                #action = self.r_t_hungarian(obs, args.m, args.n, b1=args.b1, b2=args.b2, a1=args.a1)
                action = self.r_t_hungarian(obs, 1, 1, b1=args.b1, b2=args.b2, a1=args.a1)

            obs, reward, _, truncated, _ = env.step(action)
            episode_reward += reward
            # bat = obs['battery']
            # age = obs['age']
            # print(f'step: {step} | battery: {bat} | reward: {reward}') #, end=' |')
            # print(f'action: {action}')#, end=' |')
            # env.render(action, mode='rgb_array')
        total_time_step += step
        # print(f'{i}: {episode_reward}')
        reward_per_step = episode_reward / total_time_step
        # print(f'average age per time step: {-reward_per_step}')
        end_time = time.time()
        execution_time = end_time - start_time
        # print(f'mean modified age[sec]: {-reward_per_step*0.05}')
        # env.plot_trajectory(policy='Heuristic')

        # File path for the CSV
        if args.target == 'static':
            file_path = f'CSV/experiment_data_{args.target}_target.csv'
        else:
            file_path = f'CSV/experiment_data_{args.target}_target_{args.targetv}.csv'

        # Check if file exists and load it, otherwise create a new DataFrame
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
        else:
            columns = ['m', 'n', 'seed', 'policy', 'average_age/timestep','total_wall_clock_time']
            data = pd.DataFrame(columns=columns)

        policy = 'Heuristic' #args.policy #'heuristic'
        experiment_data = pd.DataFrame({
            'm': [args.m],
            'n': [args.n],
            'seed': [args.seed],
            'policy': [policy],
            'average_age/timestep': [-reward_per_step],
            'total_wall_clock_time': [execution_time],
        })
        data = pd.concat([data, experiment_data], ignore_index=True)
        print('data', data)

        # Save the DataFrame
        data.to_csv(file_path, index=False)

if __name__ == "__main__":
    method = HeuristicGazebo()
    method.run()