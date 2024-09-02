#!/usr/bin/env python3
import math
import numpy as np
import os
import sys
import time
import argparse
import pickle
import h5py
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + os.path.sep + "gym")
current_file_path = os.path.dirname(__file__)
gym_setting_path = os.path.join(current_file_path, '../../../gym_setting/mdp')
sys.path.append(os.path.abspath(gym_setting_path))
from mdp import States, Actions, Surveillance_Actions, Rewards, StateTransitionProbability

import random
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from mavros_msgs.msg import ActuatorControl, AttitudeTarget, Thrust, State
from mavros_msgs.srv import SetMode, SetModeRequest, CommandTOL, CommandBool, CommandBoolRequest, CommandBoolResponse, StreamRate, StreamRateRequest
from sensor_msgs.msg import Imu, BatteryState
from std_msgs.msg import Header, Float64
from std_srvs.srv import Empty, EmptyRequest
#import mavros.setpoint
from scipy import sparse as sp
from scipy.stats.qmc import MultivariateNormalQMC
from scipy.optimize import linear_sum_assignment
import rospy
from gym import Env
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from typing import Optional
from math import sin, cos, pi
from numpy import arctan2, array
import re
from mavros_msgs.srv import ParamSet, ParamSetRequest, ParamSetResponse
from mavros_msgs.msg import ParamValue
from object import UAV, Target
import threading
import rospkg
import os
HIGHER_LEVEL_FREQUENCY = 0.1
LOWER_LEVEL_FREQUENCY = 10
def wrap(theta):
    if theta > math.pi:
        theta -= 2*math.pi
    elif theta < -math.pi:
        theta += 2*math.pi
    return theta
class MUMT_v1(Env):
    def __init__(self, r_max=5000, r_min=10, dt=0.05, d=40.0, l=3, m=3, n=5, r_c=10, max_step=72*360, seed=None):
        #TODO(1): Parameters
        self.Q = 22_000 #[mAh] battery capacity
        self.C_rate = 2
        self.D_rate = 0.41 # 1/2.442 (battery runtime)
        self.v = 17 #17
        self.d = d #keeping distance [ m ] = 40 m
        self.d_min = 30
        self.omega_max = self.v / self.d_min
        self.r_max = r_max

        #TODO(2): Check the MAVROS PX4 Communication
        self.init_env()
        self.uavs = [UAV(ns=f"/uav{i}", state=np.zeros(3), battery=22000) for i in range(m)]
        self.dt = dt # 20Hz

        # SUBSCRIBER
        rospack = rospkg.RosPack()
        self.package_path = rospack.get_path('simulation')
        self.attitude_target = AttitudeTarget()
        #TODO(3) GYM SPACE and Parameter
        self.m = m # UAV
        self.n = n # Target
        self.r_c = r_c # Charging Radius
        self.step_count = None
        self.episode_number = 0
        self.max_step = max_step # Max Step
        self.seed = seed

        # Initialization for Dynamic Programming
        self.n_r = round(self.r_max/self.v*10) # 원래는 800이었음.
        self.n_alpha = 360
        self.seed = seed
        self.n_u = 2 # 액션 스페이스 개수
        self.l = l
        self.lock = threading.Lock()
        self.w1_action = [None]*self.m

        #TODO(4) Define Observation and Action Space
        obs_space = {}
        for uav_id in range(1, m+1):
            for target_id in range(1, n+1):
                key = f"uav_id{uav_id}_target{target_id}"
                obs_space[key] = Box(low=np.float32([0, -np.pi]),
                                     high= np.float32([r_max, np.pi]),
                                     dtype = np.float32)
        for uav_id in range(1, m+1):
            key = f"uav{uav_id}_charge_station"
            obs_space[key] = Box(low=np.float32([0, -np.pi]),
                                 high = np.float32([r_max, np.pi]),
                                 dtype=np.float32)
        obs_space[f"battery{uav_id}"] = Box(low=np.float32([0]*m),
                                   high = np.float32([self.Q]*m),
                                   dtype=np.float32)
        obs_space["age"] = Box(low=np.float32([0]*n),
                               high=np.float32([Target.max_age]*n),
                               dtype=np.float32)
        self.observation_space = Dict(obs_space)
        self.action_space = MultiDiscrete([n+1]*m, seed=self.seed)
        #self.uavs=[] 위에서 uavs_namespace로 이미 정함. 위에서 state=뭐시기라 정한 것 좀 고쳐야 할 듯.
        self.targets = []
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        #self.distance_keeping_result00 = np.load(current_file_path+ os.path.sep + "dkc_real_dt_0.05_2a_sig0_val_iter.npz")

        self.distance_keeping_results = np.load(os.path.join(current_file_path, "dkc_real_dt_0.05_2a_sig0_val_iter.npz"))
        self.distance_keeping_straightened_policy00 = self.distance_keeping_results["policy"]

        #self.time_optimal_straightened_policy00 = np.load(current_file_path+ os.path.sep + "lengthened_toc_real_dt_0.05_2a_sig0_val_iter.npy")
        self.time_optimal_straightened_policy00 = np.load(os.path.join(current_file_path, "lengthened_toc_real_dt_0.05_2a_sig0_val_iter.npy"))

        '''
        States Form
        '''
        # 0.0 ~ self.dkc_env.r_max, self.n_r
        self.states = States(
            np.linspace(0.0, self.r_max, self.n_r, dtype=np.float32), # 0~80까지 2개 생성
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
        # Define Action Spaces (Range, Here we set 2 actions)
        self.actions = Actions(
            np.linspace(-self.omega_max, self.omega_max, self.n_u, dtype=np.float32).reshape(
                (-1, 1)
            ) # Tangentional / max_steering angle (4.5) # Yaw니까
        ) # Shape = (2, 1)
        self.initial_altitude = 10 # Set Altitude

    def init_env(self):
        print('--connecting to mavros')
        rospy.init_node('gym_px4_mavros', anonymous=True)
        print('connected')

#TODO(4) : Reset Function [GYM Environment]
#####################################################################################
    def reset(self, uav_pose=None, target_pose=None, batteries=None, ages=None,
              target_type='static', sigma_rayleigh=0.5, seed: Optional[int] = None, options: Optional[dict] = None):
        np.random.seed(seed)
        self.seed = seed
        self.step_count = 0
        for uav in self.uavs:
            uav.battery = random.randint(11000, 22000) if batteries is None else batteries[self.uavs.index(uav)]
        self.targets = []
        self.episode_number += 1
        self.uav_trajectory_data =[[] for _ in range(self.m)]

        for uav in self.uavs:
            while uav.local_position_boolean is None:
                print(f"Waiting for {uav.ns} current position to be updated")
                rospy.sleep(0.1)
            print(f"{uav.ns} current position updated successfully")
        if uav_pose is None:
            for uav in self.uavs:
                uav_x = uav.pose.pose.position.x # Current Position x
                uav_y = uav.pose.pose.position.y # Current Position y
                yaw = uav.heading_data.data * pi / 180 # Degree to Radian
                #orientation = uav.attitude_target.orientation
                #_, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
                uav_theta = yaw
                uav_states = np.vstack([uav_x, uav_y, uav_theta]).T
                print(f"Initial Position of {uav.ns} is {uav_states}")
        else:
            uav_states = uav_pose # 3xm matrix

        if batteries is None:
            batteries = np.random.randint(11000, 22000, self.m)
        else:
            batteries = batteries
        for uav_idx, uav in enumerate(self.uavs):
            uav_x, uav_y, uav_theta = uav.state
            uav_battery_level = uav.battery
            self.uav_trajectory_data[uav_idx].append((uav_x, uav_y, uav_battery_level, uav_theta))

        # TARGET
        if target_pose is None:
            print("Please Set the Target Pose")
            target1_r = np.random.uniform(30, 35, self.n)
            target1_beta = np.random.uniform(-np.pi, np.pi, self.n)
            target_states = np.array([target1_r*np.cos(target1_beta), target1_r*np.sin(target1_beta)]).T
            ages = [0]*self.n
            #극좌표계 형태.
        else:
            target_states, ages = target_pose # Target pose정의는 [target_states, ages] 형태?
            target1_r = np.sqrt(np.array([target[0]**2 + target[1]**2 for target in target_states]))
            target1_beta = np.arctan2(np.array([target[1] for target in target_states]), np.array([target[0] for target in target_states]))
        # 위에서 구한, target_states, ages
        for i in range(self.n): # Target 개수 만큼 self.target에서 에이전트 관
            # 이 데이터는 결국 n개의 행을 가진 targets agent 행렬임
            self.targets.append(Target(state=target_states[i], age=ages[i],
                                            initial_beta=target1_beta[i], initial_r=target1_r[i],
                                            target_type=target_type, sigma_rayleigh=sigma_rayleigh,
                                            m=self.m, seed=self.seed,))

        self.initial_altitude = 10
        rate = rospy.Rate(20)
        threads = []
        for uav in self.uavs:
            t = threading.Thread(target=self.takeoff_uav, args=(uav,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return self.dict_observation, {}
    #TODO(6) : STEP FUNCTION [GYM ENVRIONMENT]
#######################################################################################
    def step(self, action):
        try:
            start_time = time.time()
            terminal = False
            truncated = False
            action = np.squeeze(action)
            reward = 0
            if action.ndim == 0:
                action = np.expand_dims(action, axis = 0)
            for _ in range(LOWER_LEVEL_FREQUENCY):
                threads = []
                #print("Action is", action)
                for uav_idx, uav_action in enumerate(action): #i번째 UAV, j번쨰 Target
                    thread = threading.Thread(target=self.control_uav_thread, args=(uav_idx, uav_action))
                    thread.start()
                    threads.append(thread)
                for thread in threads:
                    thread.join()

            #TODO(7) : 목표 대상 감시 여부(진행) 확인.
            surveillance_matrix = np.zeros((self.m, self.n)) # mxn Correspondence
            for uav_idx in range(self.m):
                for target_idx in range(self.n):
                    surveillance_matrix[uav_idx, target_idx] = self.cal_surveillance(uav_idx, target_idx)
            surveillance = np.any(surveillance_matrix, axis=0).astype(int)
            #print("Surveillnace matrix ::: ", surveillance)

            for uav_idx, uav in enumerate(self.uavs):
                uav_x, uav_y, uav_theta = uav.state
                uav_battery_level = uav.battery
                self.uav_trajectory_data[uav_idx].append((uav_x, uav_y, uav_battery_level, uav_theta))
            for target_idx in range(self.n):

                self.targets[target_idx].surveillance = surveillance[target_idx]
                self.targets[target_idx].cal_age()
                reward += -self.targets[target_idx].age
            reward = reward / self.n # Average Reward of All targets

            self.step_count += 1
            end_time = time.time()
            print(f"Step Duration :: ", end_time-start_time)
            if self.step_count >= self.max_step:
                truncated = True
            return self.dict_observation, reward, terminal, truncated, {}

        except KeyboardInterrupt:
            self.save_trajectories()
        finally:
            self.save_trajectories()

####################################################################################
    #TODO(7) : Control UAV Function (Orientation)
    def control_uav_thread(self, uav_idx, action):
        #action is 0 or 1
        #with self.lock:
        self.uavs[uav_idx].charging = 0
        if self.uavs[uav_idx].battery <=0:
            pass
        else:
            if action == -1:
                action = self.uavs[uav_idx].previous_action
            elif action == 0:
                print("Go to the Charging Station")
                self.action_is_charge(uav_idx)
            else:
                if self.uavs[uav_idx].previous_action == 0 and self.uavs[uav_idx].is_landed == True:
                    self.takeoff_uav(self.uavs[uav_idx])
                    self.uavs[uav_idx].is_landed = False
                    print("Charging is Finished, Take off Again")
                    self.uavs[uav_idx].previous_action = 1
                else:
                    self.uavs[uav_idx].battery = max(0, self.uavs[uav_idx].battery - self.D_rate*self.Q/3600/LOWER_LEVEL_FREQUENCY*HIGHER_LEVEL_FREQUENCY)
                    self.w1_action[uav_idx] = self.dkc_get_action(self.rel_observation(uav_idx, action-1)[:2]) # 거리 & 알파값을 반환할 것
                    #print(f"uav_{uav_idx+1} Relative Value to Target is :::{self.rel_observation(uav_idx, action-1)[:2]}, Action is {self.w1_action[uav_idx]}")
                    self.publish_attitude(self.uavs[uav_idx], self.w1_action[uav_idx])
                    self.uavs[uav_idx].move() #그런데 w1_action은 x축 방향 각도
                    self.uavs[uav_idx].previous_action = 1

    def landing(self, uav_idx):
        print("LANDING...")
        uav=self.uavs[uav_idx]
        set_mode_service = uav.set_mode_client
        landing_mode_request=uav.landing_mode_request
        landing_mode_request.custom_mode = 'LAND'
        try:
            response = set_mode_service(landing_mode_request)
            if response.mode_sent:
                rospy.loginfo(f"{uav.ns}: LAND mode set successfully")
            else:
                rospy.logwarn(f"{uav.ns}: Failed to set LAND mode")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call to set LAND mode failed: {e}")
        land_service = uav.land_service
        landing_request = uav.landing_request
        try:
            response = land_service(landing_request)
            if response.success:
                rospy.loginfo(f"{uav.ns}: Landing initiated successfully")
            else:
                rospy.logwarn(f"{uav.ns}: Failed to initiate landing")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call to land failed: {e}")

        # UAV의 현재 고도를 확인하고, 지면에 착륙할 때까지 대기
        while True:
            current_altitude = uav.local_position.pose.position.z
            if current_altitude <= 0.1:
                rospy.loginfo(f"{uav.ns} has successfully landed.")
                uav.is_landed = True
                break
            rospy.sleep(0.1)

    def action_is_charge(self, uav_idx):
        if (self.uavs[uav_idx].obs[0]<self.r_c):
            self.uavs[uav_idx].charging = 1
            if self.uavs[uav_idx].is_landed == False:
                self.landing(uav_idx)
            else:
                print(f"charging...current battery of UAV {uav_idx} is {self.uavs[uav_idx].battery}")
                self.uavs[uav_idx].battery = min(self.Q, self.uavs[uav_idx].battery + self.C_rate*self.Q/3600/LOWER_LEVEL_FREQUENCY*HIGHER_LEVEL_FREQUENCY) # 20 Hz
        else:
            self.uavs[uav_idx].battery = max(0, self.uavs[uav_idx].battery - self.D_rate*self.Q/3600/LOWER_LEVEL_FREQUENCY*HIGHER_LEVEL_FREQUENCY) # 20 Hz
            self.w1_action[uav_idx]=self.toc_get_action(self.uavs[uav_idx].obs[:2])
            self.uavs[uav_idx].move()
            self.publish_attitude(self.uavs[uav_idx], self.w1_action[uav_idx])
        self.uavs[uav_idx].previous_action = 0

    #TODO(8) : UTILS Function
    # def save_trajectories(self):
    #     directory = os.path.join(self.package_path, 'traj')
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     filename = f'{directory}/uav_trajectory_m_{self.m}_seed_{self.seed}.pkl'
    #     with open(filename, 'wb') as file:
    #         pickle.dump(self.uav_trajectory_data, file)
    #     #print(f"Trajecotry saved in {filename}")
    def save_trajectories(self):
        directory = os.path.join(self.package_path, 'traj') 
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = f'{directory}/uav_trajectory_m_{self.m}_seed_{self.seed}.h5'  # HDF5 파일로 저장

        with h5py.File(filename, 'w') as file:
            for i, uav_data in enumerate(self.uav_trajectory_data):
                file.create_dataset(f'uav_{i}', data=np.array(uav_data))


    def publish_attitude(self, uav, yaw_rate):
        #print(f"X_veloctiy :: {self.v*sin(uav.state[2])}, Y_velocity :: {self.v*cos(uav.state[2])}")
        uav.vel_target.twist.linear.x = self.v*sin(uav.state[2])
        uav.vel_target.twist.linear.y = self.v*cos(uav.state[2])
        uav.vel_target.twist.linear.z = 0
        uav.vel_target.twist.angular.z = yaw_rate
        uav.local_vel_pub.publish(uav.vel_target)
        rospy.sleep(0.1)


    def takeoff_uav(self, uav):
        uav.pose.pose.position.x = uav.local_position.pose.position.x
        uav.pose.pose.position.y = uav.local_position.pose.position.y
        uav.pose.pose.position.z = self.initial_altitude
        for i in range(100):
            if rospy.is_shutdown():
                return
            uav.local_pos_pub.publish(uav.pose)
            rospy.sleep(0.1)
        uav.offboard()
        while not uav.offboard():
            rospy.logwarn(f"{uav.ns}: Failed to set OFFBOARD mode. Try Again")
            uav.offboard()
            continue

        while not uav.current_state.armed:
            try:
                uav.arming()
                rospy.loginfo(f"UAV ID: {uav.uav_id}, Namespace: {uav.ns}, Armed: {uav.current_state.armed}, Mode: {uav.current_state.mode}")
                if uav.current_state.armed:
                    rospy.loginfo(f"{uav.ns}: UAV successfully armed")
                    break
                else:
                    rospy.logwarn(f"{uav.ns}: Failed to arm UAV. Retrying...")
                    uav.arming()
                    rospy.sleep(0.1)
            except Exception as e:
                rospy.logerr(f"Failed to arm {uav.ns}: {e}")
            rospy.sleep(0.5)

        print(f"All UAVs are armed, starting takeoff sequence for {uav.ns}...")
        initial_position_x = uav.local_position.pose.position.x
        initial_position_y = uav.local_position.pose.position.y
        while True:
            uav.pose.pose.position.x = initial_position_x
            uav.pose.pose.position.y = initial_position_y
            uav.pose.pose.position.z = self.initial_altitude
            uav.local_pos_pub.publish(uav.pose)
            rospy.sleep(0.1)

            current_position_x = uav.local_position.pose.position.x
            current_position_y = uav.local_position.pose.position.y
            current_position_z = uav.local_position.pose.position.z

            distance_to_goal = np.sqrt((uav.pose.pose.position.x - current_position_x) ** 2 +
                                    (uav.pose.pose.position.y - current_position_y) ** 2 +
                                    (self.initial_altitude - current_position_z) ** 2)
            if distance_to_goal < 0.1:
                print(f"INITIAL TAKE OFF of {uav.ns} is SUCCESSFUL")
                rospy.sleep(1)
                break
        rospy.sleep(0.1)

    def toc_get_action(self, state):
        # S: 각 요소에 대한 인덱스 배열, P: 각 요소에 대한 가중치 배열
        S, P = self.states.computeBarycentric(state)
        action = sum(p * self.actions[int(self.time_optimal_straightened_policy00[s])] for s, p in zip(S, P))
        return action

    def dkc_get_action(self, state): # state에 거리와 알파값이 들어감??
        S, P = self.states.computeBarycentric(state)
        try:
            action = sum(p * self.actions[int(self.distance_keeping_straightened_policy00[s])] for s, p in zip(S, P))
        except IndexError as e:
            print(f"IndexError: {e}, S: {S}, distance_keeping_straight")
            raise e
        return action

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

    def convert_angle_from_euler(self, euler):
        theta = pi/2 - euler
        if theta > pi:
            theta -= 2*pi
        elif theta < - pi:
            euler += 2*pi
        return theta

    def rel_observation(self, uav_idx, target_idx): # of target relative to uav
        '''
        yaw: +y basis, CW
        angle(arctan2) : +x, CCW
        '''
        uav_x, uav_y, euler_theta = self.uavs[uav_idx].state # 현재 uav state 정보를 받아야 하는 거 아닌가?
        target_x, target_y = self.targets[target_idx].state # state update도 좋지만.
        #print(f"uav{uav_idx+1}_position : uav_x, uav_y :: target{target_idx+1}_position: [{target_x}, {target_y}]")
        x = uav_x - target_x# target x, uav_x 상대 x 좌표 본래 target_x - uav_x
        y = uav_y - target_y # target y, uav y 상대 y 좌표 본래 target_y - uav_y
        r = np.sqrt(x**2 + y**2) # 상대 거리
        beta = arctan2(y, x) # x축 기준
        theta = self.convert_angle_from_euler(euler_theta)
        alpha = wrap(beta - theta - math.pi) # theta는 orientation [-pi, pi]
        return array([r, alpha, beta],dtype=np.float32)

    @property
    def dict_observation(self):
        dictionary_obs = {}
        for uav_id in range(self.m):
            for target_id in range(self.n):
                key = f"uav{uav_id+1}_target{target_id+1}"
                dictionary_obs[key] = self.rel_observation(uav_id, target_id)[:2]

        # Add observations for each UAV-charging station
        for uav_id in range(self.m):
            key = f"uav{uav_id+1}_charge_station"
            dictionary_obs[key] = self.uavs[uav_id].obs[:2]

        dictionary_obs[f"battery{uav_id}"] = np.float32([self.uavs[uav_id].battery for uav_id in range(self.m)])
        dictionary_obs["age"] = np.float32([self.targets[target_id].age for target_id in range(self.n)])

        return dictionary_obs
if __name__ == '__main__':#
    main=MUMT_v1()
    main.reset()
