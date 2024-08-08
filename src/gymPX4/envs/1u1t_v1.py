#!/usr/bin/env python3
import math
import numpy as np
import os
import sys
import time
import argparse
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + os.path.sep + "gym")
current_file_path = os.path.dirname(__file__)
gym_setting_path = os.path.join(current_file_path, '../gym_setting/mdp')
sys.path.append(os.path.abspath(gym_setting_path))

from mdp import States, Actions, Surveillance_Actions, Rewards, StateTransitionProbability, Policy, MarkovDecisionProcess

# This is the solution using Heuristic Method: 1u1t method
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

def wrap(theta):
    if theta>pi:
        theta -= 2*pi
    elif theta < -pi:
        theta += 2*pi
    return theta
class sust_v1(Env):
    metadata = {"render_modes": [ "human", "rgb_array"], "render_fps":30}
    # Define UAV Agent ( Child Class )
    # Three Method
    '''
    Optional[str]: "human" or "rgb"
    r : distance
    dt: time step?
    m: number of UAVS -> 1
    n: Number of Targets -> 1
    r_c : Chargable distance
    seed : Randomness -> 동일 시드에 대해 동일 실험값 : 보통 강화학습에 많이 사용
    '''
    def __init__(self, render_mode : Optional[str]=None, r_max=80, r_min=0, dt=0.05, d=10.0, l=3, m=1, n=1, r_c=3, max_step=6000, seed=None):
        super().__init__()
        ## SUBSCRIBER from MAVROS
        self.battery_state = rospy.Subscriber("/mavros/")
        self.position_state = rospy.Subscriber("/mavros")
        self.velocity_state = rospy.Subscriber("/mavros")
        self.heading_angle = rospy.Subscriber("/mavros/")
        self.render_mode = render_mode # We use Human
        self.seed=seed # Randomness
        obs_space= {} # Obsrvation Space Initialization
        for uav_id in range(1, m+1):
            for target_id in range(1, n+1):
                key = f"uav{uav_id}_target{target_id}" # 타겟과 UAV 관계 딕셔너리 키로 사용
                '''
                관측 space 생성 특정 UAV-Target 간 관계 생성 후, Box를 이용해 연속적인 값 제한 생성
                low bound와 high bound 생성 -> 그 범위 안에서 관측값 제한 생성
                [거리, 각도], 데이터 타입은 실수
                key=는 딕셔너리
                관측 공간은
                [특정 관계 쌍]=[거리, 각도(heading angle)]형태로 구성되어 있음
                '''
                obs_space[key] = Box(low=np.float32([r_min, -np.pi]),
                                        high=np.float32([r_max, np.pi]),
                                        dtype=np.float32)

                # Add Observation Space for Battery and Age
                # battery에 대한 딕셔너리에 대해 UAV 개수 만큼 행렬 생성
                # [0]이 m개(UAV 개수)
                # age에 대해선 "pair" 개수 여야 하고 pair 개수는 target 수와 동이랗므로 n개 생성
                obs_space["battery"] = Box(low=np.float32([0]*m),
                                        high=np.float32([3000]*m),
                                        dtype=np.float32)
                obs_space["age"] = Box(low=np.float32([0]*n),
                               high=np.float32([1000]*n),
                               dtype=np.float32)
                self.observation_space = Dict(obs_space)
                '''
                {
                "uav1_target1": Box([  0.        , -3.1415927], [100.        ,  3.1415927], (2,), float32),
                "uav1_target2": Box([  0.        , -3.1415927], [100.        ,  3.1415927], (2,), float32),
                "uav2_target1": Box([  0.        , -3.1415927], [100.        ,  3.1415927], (2,), float32),
                "uav2_target2": Box([  0.        , -3.1415927], [100.        ,  3.1415927], (2,), float32),
                "uav3_target1": Box([  0.        , -3.1415927], [100.        ,  3.1415927], (2,), float32),
                "uav3_target2": Box([  0.        , -3.1415927], [100.        ,  3.1415927], (2,), float32),
                }
                "battery": Box([0. 0. 0.], [3000. 3000. 3000.], (3,), float32)
                "age": Box([0. 0.], [1000. 1000.], (2,), float32)
                이런 형태를 가진다.
                '''
                self.action_space = MultiDiscrete([n+1]*m, seed=self.seed)
                # 0이 충전을 뜻하기 때문에 Target 개수 n보다 1개 더 많은 공간을 생성, 그리고 각 UAV의 행동은 UAV 개수에 의존 따라서 m을 곱함
                self.dt =dt
                self.discount = 0.99 # Discount Factor
                self.r_max = r_max
                self.d = d
                self.l = l
                self.m = m
                self.uavs = [] # uav dictionary 생성
                # 아래는 중요하지 않음
                self.uav_color = [(random.randrange(0, 11) / 10, random.randrange(0, 11) / 10, random.randrange(0, 11) / 10) for _ in range(m)]
                self.n=n
                self.targets=[]
                self.trajectory_data=[]
                self.r_c = r_c # Charge Station Radius
                self.step_count = None
                self.num2str = {0: "charge", 1: "target_1"}
                self.max_step = max_step
                self.viewer = None
                self.episode_counter =0
                self.frame_counter =0
                self.save_frames = False
                # # For DP
                # self.n_r =800
                # self.n_alpha = 300
                # self.n_u =2
                
                # self.states = States()
                # self.actions = Actions()
                self.uav_trajectory_data = [[] for _ in range(self.m)]
                self.target_trajectory_data = [[] for _ in range(self.n)]
            # Episode가 끝나고 Reset? 초기 관측을 반환하는 건데, 새로운 에피소드가 시작될 때마다 호출

    def reset(self, uav_pose=None, target_pos=None, batteries=None, ages=None, target_type= 'static',
                      sigma_rayleigh=0.5, seed: Optional[int]=None, options: Optional[dict]=None):
        self.uavs = []
        self.targets =[]
        self.uav_trajectory_data = [[] for _ in range(self.m)]
        self.target_trajectory_data = [[] for _ in range(self.n)]
        np.random.seed(seed)
        self.seed = seed
        self.episode_counter +=1
        self.step_count = 0
        # 무시해도 됨
        if self.save_frames:
            os.makedirs(
                os.path.join(self.SAVE_FRAMES_PATH, f"{self.episode_counter:03d}"),
                exist_ok=True,
            )
            self.frame_counter = 0
        # 우리는 여기서 uav의 Pos를 받아올 수 있으니 그걸로 대체하기
        if uav_pose is None:
            uav_r = np.random.uniform(0,40,self.m)
            uav_beta = np.random.uniform(-pi, pi, self.m) # m개 uav 임의 생성
            uav_theta = np.random.uniform(-pi, pi, self.m) # m개 생성
            uav_x = uav_r*np.cos(uav_beta)
            uav_y = uav_r*np.sin(uav_beta) # uav_beta: uav의 방위각
            uav_states = np.vstack([uav_x, uav_y, uav_theta]).T # mx3행
        else:
            uav_states = uav_pose
        if batteries in None:
            batteries = np.random.radint(1500, 3000, self.m)
        else:
            batteries=batteries
        
        # Create UAV Instance
        for i in range(self.m): # 모든 UAV에 대해서
            # 지금 UAV, batteries 따로 있는데 이거 묶어서 한 번에 정의해주기
            self.uavs.append(self.UAV(state=uav_states[i], battery=batteries[i]))
        for uav_idx, uav in enumerate(self.uavs):
            uav_x, uav_y, uav_theta = uav.state # 이 UAV 정보를 어디서 받아와? UAV정의를 내가 한 적이...
            uav_battery_level = uav.battery
            self.uav_trajectory_data[uav_idx].append((uav_x, uav_y, uav_battery_level, uav_theta))
        
        if target_pose is None:
            target1_r=np.random.uniform(30,35,self.n)
            target1_beta = np.random.uniform(-np.pi, np.pi, self.n)
            target_states = np.array([target1_r*np.cos(target1_beta), target_r*np.sin(targe1_beta)]).T
            ages = [0]*self.n
        else:
            target_states, ages = target_pose # target_pose 형태는 2개를 return?
        
        # Create Target instances
        for i in range(self.n):
            self.targets.append(self.Target(stat=target_states[i], age=ages[i],
                                            initial_beta=target1_beta[i], initial_r=target1_r[i],
                                            target_type=target_type, sigma_rayleigh=sigma_rayleigh,
                                            m=self.m, seed=self.seed,))
        for target_idx, target in enumerate(self.targets):
            target_x, target_y = target.state
            self.target_trajectory_data[target_idx].append((target_x, target_y))
        return self.dict_observation, {}
    def q_init(self): # Value Function
        self.n_alpha = 10
        self.target_discretized_r_space = np.array([0,4,6,8,10,12,14,16,20,30,40,60,80])
        self.charge_discretized_r_space = np.array([0,2,3,5,6,10,20,30,40])
        self.target_alpha_space = np.linspace(-np.pi, np.pi - np.pi / self.n_alpha, self.n_alpha, dtype=np.float32)
        self.charge_alpha_space = np.linspace(-np.pi, np.pi - np.pi / self.n_alpha, self.n_alpha, dtype=np.float32)
        self.battery_space = np.concatenate([np.arange(0, 500, 100), np.arange(500, 3100, 500)])
        self.age_space = np.arange(0, 1001, 100) #changeage
        ## 사전에 계산된 정책과 가치 함수를 업로드함.
        self.UAV1Target1_result00 = np.load(current_file_path + f"/1U1T_s6_age1000:100_gamma_{self.discount}_dt_{self.dt}_{'val'}_iter.npz")
        self.UAV1Target1_straightened_policy00 = self.UAV1Target1_result00["policy"]
        self.UAV1Target1_values00 = self.UAV1Target1_result00["values"]
