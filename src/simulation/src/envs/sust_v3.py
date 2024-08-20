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
gym_setting_path = os.path.join(current_file_path, '../../../gym_setting/mdp')
sys.path.append(os.path.abspath(gym_setting_path))
#print(sys.path)
#from heuristic import Heuristic
from mdp import States, Actions, Surveillance_Actions, Rewards, StateTransitionProbability

from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from mavros_msgs.msg import ActuatorControl, AttitudeTarget, Thrust, State
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
import pandas as pd
from tf.transformations import euler_from_quaternion

def wrap(theta):
    if theta > pi:
        theta -= 2*pi
    elif theta < -pi:
        theta += 2*pi
    return theta

class SUST_v3(Env):
    methadata = {"render_modes": [ "human", "rgb_array"], "render_fps": 30}
    class UAV:
        def __init__(self, state, v=1.0, battery = None):
            self.v = v
            self.dt = 0.05
            self.state = state
            self.battery = battery
            self.charging = 0
        def copy(self):
            return SUST_v3.UAV(state=self.state.copy(), v=self.v, battery=self.battery)
        def move(self, action):
            dtheta = action*self.dt
            _lambda = dtheta / 2
            if _lambda == 0.0:
                self.state[0] += self.v * self.dt * cos(self.state[-1])
                self.state[1] += self.v * self.dt * sin(self.state[-1])
            else:
                ds = self.v * self.dt * sin(_lambda) / _lambda
                self.state[0] += ds*cos(self.state[-1] + _lambda)
                self.state[1] += ds*sin(self.state[-1] + _lambda)
                self.state[2] += dtheta
                self.state[2] = wrap(self.state[2])
        @property
        def obs(self):
            x, y = self.state[:2]
            r = np.sqrt(x**2 + y**2)
            alpha = wrap(np.arctan2(y, x) - wrap(self.state[-1]) - pi)
            beta = arctan2(y, x)
            return array([r, alpha, beta], dtype = np.float32)

    class Target:
        _id_counter = 0
        def __init__(self, state, age=0, initial_beta = 0, initial_r = 30, target_type = 'static', sigma_rayleigh = 0.5, m=None, seed = None ):
            self.dt = 0.05
            self.state = state
            self.surveillance = None
            self.age = age
            self.initial_beta = initial_beta
            self.initial_r = initial_r
            self.target_type = target_type
            self.sigma_rayleigh = sigma_rayleigh
            self.m = m
            self.seed = seed
            self.target_v = 0.25
            self.time_elapsed = 0
            self.positions = []
            type(self)._id_counter += 1
            self.id = type(self)._id_counter
            self.step_idx = 0
            self.angle_radians = self.target_v * self.dt / self.initial_r
            self.rotation_matrix = np.array([
                [np.cos(self.angle_radians), -np.sin(self.angle_radians)],
                [np.sin(self.angle_radians), np.cos(self.angle_radians)]
            ])
        def copy(self):
            return SUST_v3(state = self.state.copy(), age=self.age, initial_beta = self.initial_beta, target_type = self.target_type, sigma_rayleigh = self.sigma_rayleigh)
        def cal_age(self):
            if self.surveillance == 0:
                self.age = min(1000, self.age + 1)
            else:
                self.age = 0
        def update_position(self):
            if self.target_type == 'load':
                #Target Trajectory Formation 확인 필요
                # 따로 타겟 경로 지정해줄 경우 [[x, y]] 형태임
                try:
                    trajectory_array = np.load()
                except Exception as e : print(e)
                if trajectory_array.ndim > 2:
                    self.state = trajectory_array[self.id][self.step_idx]
                else:
                    self.state = trajectory_array[self.step_idx]
                self.step_idx += 1
            if self.target_type == 'static':
                print("Target position is fixed : STATIC")
        @property
        def obs(self):
            x, y =self.state
            r = np.sqrt(x**2 + y**2)
            beta = np.arctan2(y,x)
            return np.array([r, beta])

    def __init__(self, render_mode: Optional[str]=None, r_max=80, r_min=0, dt=0.05, d=10.0, l=3, m=1, n=2, r_c=3, max_step=6000, seed=None):
        super().__init__()
        self.render_mode = render_mode
        self.seed = seed
        self.discount = 0.999 #Discount Factor
        self.r_max = r_max
        self.d = d
        self.l = l
        self.m = m # UAV
        self.n = n # Target
        self.r_c = r_c
        self.step_count = None
        self.episode_number = 0
        self.frame_number = 0
        self.n_r = 800
        self.n_alpha = 360
        self.n_u = 2 #2 #21 # Action Space Size
        self.max_step = max_step
        self.episode_counter = 0
        self.frame_counter = 0
        # self.battery_state = 10 ㄴ어나ㅣㅁㅇㄴ;ㅣㄴ이니

        obs_space = {}
        '''
        1. uav_obs = [x, y, heading angle]
        2. Target_obs = [x, y, beta]
        3. Total obs = [alpha_t, alpha_c, battery, age]
        '''
        for uav_id in range(1, m+1):
            for target_id in range(1, n+1):
                key = f"uav_id{uav_id}_target{target_id}"
                obs_space[key] = Box(low=np.float32([r_min, -np.pi]),
                                        high=np.float32([r_max, np.pi]),
                                        dtype=np.float32)
        for uav_id in range(1, m+1):
            key = f"uav{uav_id}_charge_station"
            obs_space[key] = Box(low=np.float32([r_min, -np.pi]),
                                 high = np.float32([r_max, np.pi]),
                                 dtype=np.float32)
        obs_space["battery"] = Box(low=np.float32([0]*m),
                                   high = np.float32([3000]*m),
                                   dtype=np.float32)
        obs_space["age"] = Box(low=np.float32([0]*n),
                               high=np.float32([1000]*n),
                               dtype=np.float32)
        ## Observations SPACE 정의
        self.observation_space = Dict(obs_space)
        ## ACTION SPACE 정의
        self.action_space = MultiDiscrete([n + 1]*m, seed=self.seed) # Action 0: charge, 1: 1target 2: 2target..
        print("action_space", self.action_space)
        self.uavs=[]
        self.targets =[]
        self.num2str = {0: "charge", 1: "target1"}
        self.uav_trajectory_data = [[] for _ in range(self.m)] # array 초기화
        self.target_trajectory_data = [[] for _ in range(self.n)] # array 초기화


        # UPLOAD DKC and TOD Policy Learning Results.
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        self.distance_keeping_results = np.load(os.path.join(current_file_path, "v1_80_2a_dkc_val_iter.npz"))
        self.distance_keeping_straightened_policy00 = self.distance_keeping_results["policy"]
        self.time_optimal_straightened_policy00 = np.load(os.path.join(current_file_path, "v1_terminal_40+40_2a_toc_policy_fp64.npy"))
        # current_file_path = os.path.dirname(os.path.abspath(__file__))
        # self.distance_keeping_result00 = np.load(current_file_path+ os.path.sep + "v1_80_2a_dkc_val_iter.npz")
        # self.distance_keeping_straightened_policy00 = self.distance_keeping_result00["policy"] # .data
        # self.time_optimal_straightened_policy00 = np.load(current_file_path+ os.path.sep + "v1_terminal_40+40_2a_toc_policy_fp64.npy")

        # Define States
        '''
        States Form
        '''
        self.states = States(
            np.linspace(0.0, 80.0, self.n_r, dtype=np.float32), # 0~80까지 2개 생성
            np.linspace(
                -np.pi,
                np.pi - np.pi /self.n_alpha,
                self.n_alpha,
                dtype=np.float32
            ), # n_alpha개 생성
            cycles = [np.inf, np.pi*2],
            n_alpha=self.n_alpha
        )
        '''
        Actions Form
        '''
        # Define Action Spaces
        self.actions = Actions(
            np.linspace(-1.0 / 4.5, 1.0 / 4.5, self.n_u, dtype=np.float32).reshape(
                (-1, 1)
            ) # Tangentional / max_steering angle (4.5) # Yaw니까
        ) # Shape = (2, 1)

    def reset(
        self, uav_pose = None, target_pose=None, batteries=None, ages=None,
        target_type = 'static', sigma_rayleigh = 0.5, seed: Optional[int] = None, options : Optional[dict] = None,
    ): # Sigma rayleigh => 일단 random 아니면 의미가 흠
        self.uavs = []
        self.targets=[]
        self.uav_trajectory_data = [[] for _ in range(self.m)]
        self.target_trajectory_data = [[] for _ in range(self.n)]
        np.random.seed(seed)
        self.seed = seed
        self.episode_counter += 1
        self.step_count = 0
        if uav_pose is None: # 아니면 uav_pose 위치 정의해줘야 함. uav 소환위치면 되겠지 [[0,0]]
            # 원점 좌표로 잡지
            uav_r = np.random.uniform(0, 40, self.m)  # D=40
            uav_beta = np.random.uniform(-pi, pi, self.m)
            uav_theta = np.random.uniform(-pi, pi, self.m)
            # Create the state arrays
            uav_x = uav_r * np.cos(uav_beta)
            uav_y = uav_r * np.sin(uav_beta)
            # uav_theta = 0 # Orientation
            # uav_x = 0
            # uav_y = 0
            uav_states = np.vstack([uav_x, uav_y, uav_theta]).T
        else:
            uav_states = uav_pose # uav_pose는 [x, y, theta] 형태여야 함. reset에 줄 때 uav_pose 정의할 것
        if batteries is None:
            batteries = np.random.randint(1500, 3000, self.m)
            #batteries = self.battery_state #원래는 임의값
        else:
            batteries = batteries #함수에 넣어준 값 사용하기
        for i in range(self.m):
            # uav 어레이 만들기
            self.uavs.append(self.UAV(state = uav_states[i], battery = batteries[i])) # i행이 UAV니까
        for uav_idx, uav in enumerate(self.uavs): # 행렬 번호를 idx로 그리고 uav를 내용으로?
            uav_x, uav_y, uav_theta = uav.state # uav는 위의 iteration내용 i번째 UAV(UAV agent)
            uav_battery_level = uav.battery # self.uav아니고?
            self.uav_trajectory_data[uav_idx].append((uav_x))

        if target_pose is None:
            print("Please Set the Target Pose")
            # 아래는 원래 지정값인데, 환경 테스트 용으로 넣겠음.
            target1_r = np.random.uniform(30, 35, self.n)
            target1_beta = np.random.uniform(-np.pi, np.pi, self.n)
            target_states = np.array([target1_r*np.cos(target1_beta), target1_r*np.sin(target1_beta)]).T
            ages = [0]*self.n
            #극좌표계 형태.
        else:
            '''
            # 예시 데이터
            target_states = np.array([[10.0, 5.0], [15.0, 10.0]])  # 두 개의 타겟의 초기 위치
            ages = [0, 0]  # 두 개의 타겟의 초기 나이

            # target_pose는 다음과 같은 형태
            target_pose = (target_states, ages)
            '''
            target_states, ages = target_pose # Target pose정의는 [target_states, ages] 형태?
            target1_r = np.sqrt(np.array([target[0]**2 + target[1]**2 for target in target_states]))
            target1_beta = np.arctan2(np.array([target[1] for target in target_states]), np.array([target[0] for target in target_states]))
        # 위에서 구한, target_states, ages
        # Taregt_beta는 어디서 정의함? initial_r은 어디서 정의함? target_type=static? 나머지는?
        for i in range(self.n): # Target 개수 만큼 self.target에서 에이전트 관리
            # 이 데이터는 결국 n개의 행을 가진 targets agent 행렬임
            self.targets.append(self.Target(state=target_states[i], age=ages[i],
                                            initial_beta=target1_beta[i], initial_r=target1_r[i],
                                            target_type=target_type, sigma_rayleigh=sigma_rayleigh,
                                            m=self.m, seed=self.seed,))
        for target_idx, target in enumerate(self.targets):
            target_x, target_y = target.state
            self.target_trajectory_data[target_idx].append((target_x, target_y))
        return self.dict_observation, {}

# 여기서 action list에 대해서 받음 action [ [1] [2] [3] [4]] etc
    def step(self, action):
        terminal = False
        truncated = False
        action = np.squeeze(action)
        reward = 0
        if action.ndim == 0:
            print("Action dimension # is zero")
            action = np.expand_dims(action, axis = 0)
        for uav_idx, uav_action in enumerate(action): # 인덱스 번호(UAV), Target 번호(uav_action)
            print('uav_idx in control_uav:', uav_idx, 'uav_action in control', uav_action)
            self.control_uav(uav_idx, uav_action) # uav_idx, uav_action 문제가 생김
        surveillance_matrix = np.zeros((self.m, self.n)) # mxn matrix

        # SURVEILLANCE
        for uav_idx in range(self.m):
            for target_idx in range(self.n):
                surveillance_matrix[uav_idx, target_idx] = self.cal_surveillance(uav_idx, target_idx)
        surveillance = np.any(surveillance_matrix, axis=0).astype(int)

        for target_idx in range(self.n):
            # Target의 Age와 각 target의 surveillance설정
            self.targets[target_idx].surveillance = surveillance[target_idx]
            self.targets[target_idx].cal_age()
            reward += -self.targets[target_idx].age
        reward = reward / self.n # Average Reward of All targets


        for uav_idx, uav in enumerate(self.uavs):  # Replace 'self.uavs' with how you access your UAVs
            uav_x, uav_y, uav_theta = uav.state  # Replace with actual position attributes
            uav_battery_level = uav.battery  # Replace with actual battery attribute
            self.uav_trajectory_data[uav_idx].append((uav_x, uav_y, uav_battery_level, uav_theta))

        for target_idx, target in enumerate(self.targets):
            target.update_position()

            target_x, target_y = target.state
            self.target_trajectory_data[target_idx].append((target_x, target_y))
        self.step_count += 1
        if self.step_count >= self.max_step:
            truncated = True
            self.trajectory_data.append(list(target.positions for target in self.targets))
            self.save_trajectories()
        return self.dict_observation, reward, terminal, truncated, {}


    ## UTILS FUNCTION
    def toc_get_action(self, state):
        # S: 각 요소에 대한 인덱스 배열, P: 각 요소에 대한 가중치 배열
        #print('##### STATE in ComputeBaryCentric Function Find! ERROR!!',state)
        S, P = self.states.computeBarycentric(state)
        action = sum(p * self.actions[int(self.time_optimal_straightened_policy00[s])] for s, p in zip(S, P))
        print("GETTING TOC POLICY IS SUCCESSFUL!, Return dTHETA", action)
        return action

    def dkc_get_action(self, state): # state에 거리와 알파값이 들어감??
        print("####STATE####", state) # 1st iter ####STATE#### [36.05551     0.98279375]
                                      # 2nd iter ####STATE3### [[36.02757   ] [ 0.97283304]]

        S, P = self.states.computeBarycentric(state)
        print(f"S: {S}, P : {P}")
        #Length of distance_keeping_straightened_policy00: 288000
        #print(f"Length of distance_keeping_straightened_policy00: {len(self.distance_keeping_straightened_policy00)}")
        try:
            # S: [129835 129836 130195 130196], P : [0.01677369 0.8787953  0.00195596 0.10247502]
            #optimal_index = [int(self.distance_keeping_straightened_policy00[s]) for s, p in zip(S,P)]
            #print("##optimal_index##", optimal_index)
            ##optimal_index## [20, 20, 20, 20]
            #print('###### SHAPE ########', np.shape(self.actions))
            action = sum(p * self.actions[int(self.distance_keeping_straightened_policy00[s])] for s, p in zip(S, P))
            print("GETTING DKC POLICY IS SUCCESSFUL!, Return dTHETA", action)
        except IndexError as e:
            print(f"IndexError: {e}, S: {S}, distance_keeping_straight")
            raise e
        return action

    def rel_observation(self, uav_idx, target_idx): # of target relative to uav
        uav_x, uav_y, theta = self.uavs[uav_idx].state
        target_x, target_y = self.targets[target_idx].state
        x = target_x - uav_x # target x, uav_x 상대 x 좌표
        y = target_y - uav_y # target y, uav y 상대 y 좌표
        r = np.sqrt(x**2 + y**2) # 상대 거리
        beta = arctan2(y, x)
        alpha = wrap(beta - wrap(theta))
        return array([r, alpha, beta],dtype=np.float32)

    def cal_surveillance(self, uav_idx, target_idx):
        if self.uavs[uav_idx].battery <= 0:
            return 0
        else: # UAV alive
            if (
                self.d - self.l < self.rel_observation(uav_idx, target_idx)[0] < self.d + self.l
                and self.uavs[uav_idx].charging != 1 # uav 1 is not charging(on the way to charge is ok)
            ):
                return 1 # uav1 is surveilling target 1
            else:
                return 0

    # 배터리 상태에 따라 Action값 반환.
    def control_uav(self, uav_idx, action):
        self.uavs[uav_idx].charging = 0
        if self.uavs[uav_idx].battery <= 0:
            pass
        else:
            if action == 0: # Charge
                # TODO(1) : Create Message Files
                if (self.uavs[uav_idx].obs[0] < self.r_c):
                    self.uavs[uav_idx].charging = 1 # charging 비율 선정하기
                    self.uavs[uav_idx].battery = min(self.uavs[uav_idx].battery + 10, 3000)
                else:  # not able to land on charge station(too far)
                    self.uavs[uav_idx].battery -= 1
                    w1_action = self.toc_get_action(self.uavs[uav_idx].obs[:2])
                    self.uavs[uav_idx].move(w1_action)
            else:  # surveil target1 # Target 번호
                self.uavs[uav_idx].battery -= 1 # Battery 감소 비율 정해서 수정할 것. [Parameter]
                print("TEST####",self.rel_observation(uav_idx, action-1)[:2]) #TEST#### [36.05551     0.98279375]
                # Action ERROR
                w1_action = self.dkc_get_action(self.rel_observation(uav_idx, action-1)[:2]) # 거리 & 알파값을 반환할 것
                print("w1_action_uav_idx", uav_idx, "w1_action", w1_action)
                # uav 상태 업데이트.
                self.uavs[uav_idx].move(w1_action)
        # UAV의 다음단계 추가.. 리턴? : 내가 추가한 것
        #return self.uavs[uav_idx].move(w1_action)

    @property
    def dict_observation(self):
        # 각 목표-UAV 간의 거리 및 각도 등의 상대적 관측값을 모조리 저장함. 그러면 어떤 pair를 쓸지는 어떻게 아나?
        dictionary_obs = {}
        # Add observations for UAV-target pairs according to the rule
        for uav_id in range(self.m):
            for target_id in range(self.n):
                key = f"uav{uav_id+1}_target{target_id+1}"
                # 여기서는 uav 번호에 따른 target 번호 간의 distance, angle 을 반환함.
                # 그런데 몇 번째랑 해야 하는지 어떻게 알아?
                dictionary_obs[key] = self.rel_observation(uav_id, target_id)[:2]

        # Add observations for each UAV-charging station
        for uav_id in range(self.m):
            # 이건  uav_charging에 따른 obs 결과: 거리, 각도
            key = f"uav{uav_id+1}_charge_station"
            dictionary_obs[key] = self.uavs[uav_id].obs[:2]

        # Add observation for battery levels and ages of targets
        dictionary_obs["battery"] = np.float32([self.uavs[uav_id].battery for uav_id in range(self.m)])
        dictionary_obs["age"] = np.float32([self.targets[target_id].age for target_id in range(self.n)])

        return dictionary_obs

# if __name__ == '__main__':
    # env=SUST_v3()
    # heuristic = Heuristic()
    # # Reset함수의 반환값은 self.dictionary_obs, {} 두 개인데.../
    # m=4
    # n=2
    # uav_env = SUST_v3(m=m, n=n)
    # state_sample = uav_env.observation_space.sample()
    # action_sample = uav_env.action_space.sample()
    # print("state_sample", state_sample)
    # print("Action_sample", action_sample)
    # repitition = 10
    # avg_reward = 0
    # for i in range(repitition):
    #     step = 0
    #     truncated = False
    #     obs, _ = uav_env.reset(seed=i)
    #     total_reward = 0
    #     while truncated == False:
    #         step += 1
    #         action = heuristic.high_age_first(obs, m)
    #         obs, reward, _, truncated, _ = env.step(action)
    #         total_reward += reward
    #         bat = obs['battery']
    #         age = obs['age']
    #         print(f'step: {step} | battery: {bat} | reward: {reward}')
    #         print(f"Action{action}, Observation:{obs}")
    #     avg_reward += total_reward
    # avg_reward /= repitition
    # print(f'average reward: {avg_reward}')



