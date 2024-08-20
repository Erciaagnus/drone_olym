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
from mdp import States, Actions, Surveillance_Actions, Rewards, StateTransitionProbability

import random
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
def wrap(theta):
    if theta > math.pi:
        theta -= 2*math.pi
    elif theta < -math.pi:
        theta += 2*math.pi
    return theta
class MUMT_v1(Env):
    def __init__(self, r_max=80, r_min=0, dt=0.05, d=10.0, l=3, m=3, n=5, r_c=10, max_step=6000, seed=None):
        #TODO(2): Check the MAVROS PX4 Communication
        self.init_env()
        self.uavs = [UAV(ns=f"/uav{i}", state=np.zeros(3), battery=22000) for i in range(m)]
        self.attitude_target=  AttitudeTarget()
        #self.ns = ns #uav0, uav1, uav2 형식
        self.local_position_boolean = None
        self.current_state = State()
        self.local_position = PoseStamped()
        self.current_state = State()
        self.offb_set_mode = SetModeRequest()
        self.arm_cmd = CommandBoolRequest()
        self.arm_cmd.value = True
        self.imu_data = Imu()
        self.local_velocity = TwistStamped()
        self.global_velocity = TwistStamped()
        self.dt = dt
        #self.gps_data = NavSatFix()
            # SUBSCRIBER
        self.state_sub = rospy.Subscriber(f"/uav0/mavros/state", State, self.state_cb, queue_size=10)
        self.attitude_sub = rospy.Subscriber(f"/uav0/mavros/setpoint_raw/attitude", AttitudeTarget, self.update_orientation, queue_size=10)
        self.pose_sub = rospy.Subscriber(f"/uav0/mavros/local_position/pose", PoseStamped, self.update_pose, queue_size=10)
        # self.imu_sub = rospy.Subscriber(f"/uav0/mavros/imu/data", Imu, self.imu_cb, queue_size=10)
        # self.gps_sub = rospy.Subscriber(f"/uav0/mavros/global_position/global", NavSatFix, self.gps_cb, queue_size=10)
            # PUBLISHER
        # self.local_pos_pub = rospy.Publisher(f"uav0/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        # self.local_vel_pub = rospy.Publisher(f"uav0/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)
        # self.attitude_pub = rospy.Publisher(f"uav0/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)

            # Service proxies # UAV
        self.arming_client = rospy.ServiceProxy(f'/uav0/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy(f'/uav0/mavros/set_mode', SetMode)
        #self.attitude_pub = rospy.Publisher(f"/uav0/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)

        #TODO(3) GYM SPACE and Parameter
        self.m = m
        self.n = n
        self.r_c = r_c
        self.step_count = None
        self.episode_number = 0
        self.max_step = max_step
        self.seed = seed
        self.n_alpha = 360
        self.n_r = 800
        self.seed = seed
        self.n_u = 2 # 액션 스페이스 개수
        self.d = d
        self.l = l
        self.v = 17
        obs_space = {}
        for uav_id in range(1, m+1):
            for target_id in range(1, n+1):
                key = f"uav_id{uav_id}_target{target_id}"
                obs_space[key] = Box(low=np.float32([r_min, -np.pi]),
                                     high= np.float32([r_max, np.pi]),
                                     dtype = np.float32)
        for uav_id in range(1, m+1):
            key = f"uav{uav_id}_charge_station"
            obs_space[key] = Box(low=np.float32([r_min, -np.pi]),
                                 high = np.float32([r_max, np.pi]),
                                 dtype=np.float32)
        obs_space["battery"] = Box(low=np.float32([0]*m),
                                   high = np.float32([22000]*m),
                                   dtype=np.float32)
        obs_space["age"] = Box(low=np.float32([0]*n),
                               high=np.float32([1000]*n),
                               dtype=np.float32)
        self.observation_space = Dict(obs_space)
        self.action_space = MultiDiscrete([n+1]*m, seed=self.seed)
        #self.uavs=[] 위에서 uavs_namespace로 이미 정함. 위에서 state=뭐시기라 정한 것 좀 고쳐야 할 듯.
        self.targets = []
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        self.distance_keeping_results = np.load(os.path.join(current_file_path, "dkc_r5.0_rt0.04_2a_sig0_val_iter.npz"))
        self.distance_keeping_straightened_policy00 = self.distance_keeping_results["policy"]
        self.time_optimal_straightened_policy00 = np.load(os.path.join(current_file_path, "lengthened_toc_r5.0_2a_sig0_val_iter.npy"))

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
        # Define Action Spaces (Range, Here we set 2 actions)
        self.actions = Actions(
            np.linspace(-1.0 / 4.5, 1.0 / 4.5, self.n_u, dtype=np.float32).reshape(
                (-1, 1)
            ) # Tangentional / max_steering angle (4.5) # Yaw니까
        ) # Shape = (2, 1)
        self.initial_altitude = 10 # Set Altitude
        ## RVIZ VISUALIZATION
        # self.uav_pose_pub = rospy.Publisher("/uav_pose", PoseStamped, queue_size=10)
        # self.target_pose_pub = [rospy.Publisher(f"/target_pose_{i}", PoseStamped,queue_size=10 ) for i in range(n)]

    def update_orientation(self, msg):
            self.attitude_target = msg

    def update_pose(self, msg):
            self.local_position = msg # Current Position
            self.local_position_boolean = True
    def state_cb(self, data):
            self.current_state = data
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

        for uav in self.uavs:
            while uav.local_position_boolean is None:
                print(f"Waiting for {uav.ns} current position to be updated")
                rospy.sleep(0.1)
            print(f"{uav.ns} current position updated successfully")
        if uav_pose is None:
            for uav in self.uavs:
                uav_x = uav.pose.pose.position.x
                uav_y = uav.pose.pose.position.y
                uav_theta = 0
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
        for i in range(self.n): # Target 개수 만큼 self.target에서 에이전트 관리
            # 이 데이터는 결국 n개의 행을 가진 targets agent 행렬임
            self.targets.append(Target(state=target_states[i], age=ages[i],
                                            initial_beta=target1_beta[i], initial_r=target1_r[i],
                                            target_type=target_type, sigma_rayleigh=sigma_rayleigh,
                                            m=self.m, seed=self.seed,))
        # self.initial_altitude = 10
        # for uav in self.uavs:
        #     print(f"UAV ID: {uav.uav_id}, Namespace: {uav.ns}, Armed: {uav.current_state.armed}, Mode: {uav.current_state.mode}")
        # # Arm all UAVs first
        # Then takeoff all UAVs
        self.initial_altitude = 10
        rate = rospy.Rate(20)
        threads = []
        for uav in self.uavs:
            t = threading.Thread(target=self.takeoff_uav, args=(uav,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        # for uav in self.uavs:
        #     uav.pose.pose.position.x = uav.local_position.pose.position.x
        #     uav.pose.pose.position.y = uav.local_position.pose.position.y
        #     uav.pose.pose.position.z = self.initial_altitude
        #     for i in range(100):
        #         if(rospy.is_shutdown()):
        #             break
        #         uav.local_pos_pub.publish(uav.pose)
        #         rate.sleep()
        #         #rospy.sleep(0.3)

        #     uav.offboard()
        #     # Try to set the UAV to OFFBOARD mode
        #     while not uav.offboard():
        #         rospy.logwarn(f"{uav.ns}: Failed to set OFFBOARD mode. Try Again.")
        #         uav.offboard()
        #         continue

        #     # Wait for the UAV to be armed
        #     while not uav.current_state.armed:
        #         try:
        #             uav.arming()
        #             rospy.loginfo(f"UAV ID: {uav.uav_id}, Namespace: {uav.ns}, Armed: {uav.current_state.armed}, Mode: {uav.current_state.mode}")
        #             if uav.current_state.armed:
        #                 rospy.loginfo(f"{uav.ns}: UAV successfully armed")
        #                 break
        #             else:
        #                 rospy.logwarn(f"{uav.ns}: Failed to arm UAV. Retrying...")
        #                 uav.arming()
        #                 rospy.sleep(0.1)
        #         except Exception as e:
        #             rospy.logerr(f"Failed to arm {uav.ns}: {e}")
        #         rospy.sleep(0.5)

        #     print("All UAVs are armed, starting takeoff sequence...")
        #     initial_position_x = uav.local_position.pose.position.x
        #     initial_position_y = uav.local_position.pose.position.y
        #     while True:
        #         uav.pose.pose.position.x = initial_position_x
        #         uav.pose.pose.position.y = initial_position_y
        #         uav.pose.pose.position.z = self.initial_altitude
        #         uav.local_pos_pub.publish(uav.pose)
        #         rospy.sleep(0.1)

        #         current_position_x = uav.local_position.pose.position.x
        #         current_position_y = uav.local_position.pose.position.y
        #         current_position_z = uav.local_position.pose.position.z

        #         distance_to_goal = np.sqrt((uav.pose.pose.position.x - current_position_x) ** 2 +
        #                                     (uav.pose.pose.position.y - current_position_y) ** 2 +
        #                                     (self.initial_altitude - current_position_z) ** 2)
        #         if distance_to_goal < 0.1:
        #             print(f"INITIAL TAKE OFF of {uav.ns} is SUCCESSFUL")
        #         rospy.sleep(0.1)

        return self.dict_observation, {}
    #TODO(5) : STEP FUNCTION [GYM ENVRIONMENT]
#######################################################################################
    def step(self, action):
        terminal = False
        truncated = False
        action - np.squeeze(action)
        reward = 0
        if action.ndim == 0:
            action = np.expand_dims(action, axis = 0)
        for uav_idx, uav_action in enumerate(action):
            self.control_uav(uav_idx, uav_action)
            print("GO TO NEXT STEP")
            rospy.sleep(0.1)
        #TODO(6) : 목표 대상 감시 여부(진행) 확인.
        surveillance_matrix = np.zeros((self.m, self.n)) # mxn Correspondence
        for uav_idx in range(self.m):
            for target_idx in range(self.n):
                surveillance_matrix[uav_idx, target_idx] = self.cal_surveillance(uav_idx, target_idx)
        surveillance = np.any(surveillance_matrix, axis=0).astype(int)
        print("Surveillnace matrix ::: ", surveillance)

        for target_idx in range(self.n):

            self.targets[target_idx].surveillance = surveillance[target_idx]
            self.targets[target_idx].cal_age()
            reward += -self.targets[target_idx].age
        reward = reward / self.n # Average Reward of All targets

        self.step_count += 1
        if self.step_count >= self.max_step:
            truncated = True
        return self.dict_observation, reward, terminal, truncated, {}

####################################################################################
    #TODO(7) : Control UAV Function (Orientation)
    def control_uav(self, uav_idx, action):
        #action is 0 or 1
        self.uavs[uav_idx].charging = 0
        uav = self.uavs[uav_idx]
        if self.uavs[uav_idx].battery <= 0:
            pass
        else:
            if action == 0:
                print(f'self.uavs[uav_idx].obs[0]== {self.uavs[uav_idx].obs[0]}, ### self.r_c == {self.r_c}')
                if (self.uavs[uav_idx].obs[0]<self.r_c):
                    if self.local_position.pose.position.z > 0.1:
                        print("Landing for Charging")
                        uav.pose.pose.position.x = uav.local_position.pose.position.x
                        uav.pose.pose.position.y = uav.local_position.pose.position.y
                        uav.pose.pose.position.z = 0
                        uav.local_pos_pub.publish(uav.pose)
                        rospy.sleep(0.1)
                    else:
                        self.uavs[uav_idx].charging = 1
                        self.uavs[uav_idx].battery = min(self.uavs[uav_idx].battery + 733.33, 22000)
                        print("Charging")
                        if self.uavs[uav_idx].battery >= 17000:
                            print("Go to Target Again")
                        else:
                            rospy.sleep(0.1)
                            pass
                else: # action = 1 모든 uav가 1 그러면 감시 상태
                    self.uavs[uav_idx].battery -= 2.503
                    w1_action = self.toc_get_action(self.uavs[uav_idx].obs[:2])
                    print("TOC GET ATTITUDE", w1_action)
                    self.new_uav_state = self.uavs[uav_idx].move(w1_action)
                    self.publish_attitude(self.uavs[uav_idx], w1_action)
            else:
                self.uavs[uav_idx].battery -= 2.503
                print("TEST####",self.rel_observation(uav_idx, action-1)[:2]) #TEST#### [36.05551     0.98279375]
                w1_action = self.dkc_get_action(self.rel_observation(uav_idx, action-1)[:2]) # 거리 & 알파값을 반환할 것
                print("w1_action_uav_idx", uav_idx, "w1_action", w1_action)
                # uav 상태 업데이트. 여기서 uav.state값이 바뀌게 됨. 즉 move function에서 무언갈 바꿔야 함... 받아서 새로운 값을 받도록 해야 함.
                self.uavs[uav_idx].move(w1_action)
                self.new_uav_state = self.uavs[uav_idx].move(w1_action) # yaw rate에 따라 위치 UAV 업데이트...
                print('new _State ::: INFO in CONTROL ', self.uavs[uav_idx].state)
                self.publish_attitude(self.uavs[uav_idx], w1_action)

    #TODO(8) : UTILS Function
    def publish_attitude(self, uav, yaw_rate):
        yaw = uav.state[2] + yaw_rate*self.dt
        quaternion = quaternion_from_euler(0, 0, yaw)
        print(f'quaternion[0] = {quaternion[0]}, orientation_y = {quaternion[1]}, orientation_z = {quaternion[2]}, orientation_w = {quaternion[3]}')

        self.attitude_target.orientation.x = quaternion[0]
        self.attitude_target.orientation.y = quaternion[1]
        self.attitude_target.orientation.z = quaternion[2]
        self.attitude_target.orientation.w = quaternion[3]
        self.attitude_target.thrust =0.5 # 1m/s
        uav.attitude_pub.publish(self.attitude_target)

        # Velocity control
        vel_msg = TwistStamped()
        vel_msg.header.stamp= rospy.Time.now()
        vel_msg.twist.linear.x = self.v * cos(uav.state[2])
        vel_msg.twist.linear.y = self.v * sin(uav.state[2])
        vel_msg.twist.linear.z = 0
        uav.local_vel_pub.publish(vel_msg)

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
        #print('##### STATE in ComputeBaryCentric Function Find! ERROR!!',state)
        S, P = self.states.computeBarycentric(state)
        action = sum(p * self.actions[int(self.time_optimal_straightened_policy00[s])] for s, p in zip(S, P))
        print("GETTING TOC POLICY IS SUCCESSFUL!, Return dTHETA", action)
        return action

    def dkc_get_action(self, state): # state에 거리와 알파값이 들어감??
        print("####STATE####", state) # 1st iter ####STATE#### [36.05551     0.98279375]
                                      # 2nd iter ####STATE3### [[36.02757   ] [ 0.97283304]]

        S, P = self.states.computeBarycentric(state)
        try:
            action = sum(p * self.actions[int(self.distance_keeping_straightened_policy00[s])] for s, p in zip(S, P))
            print("GETTING DKC POLICY IS SUCCESSFUL!, Return dTHETA", action)
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

    def rel_observation(self, uav_idx, target_idx): # of target relative to uav
        uav_x, uav_y, theta = self.uavs[uav_idx].state # 현재 uav state 정보를 받아야 하는 거 아닌가?
        target_x, target_y = self.targets[target_idx].state # state update도 좋지만.
        x = target_x - uav_x # target x, uav_x 상대 x 좌표
        y = target_y - uav_y # target y, uav y 상대 y 좌표
        r = np.sqrt(x**2 + y**2) # 상대 거리
        beta = arctan2(y, x)
        alpha = wrap(beta - wrap(theta))
        return array([r, alpha, beta],dtype=np.float32)
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
                # rel observation 계산해서, dictionary_obs에다가 넣음. -> 여기서 obs 반환하는 거네..
                dictionary_obs[key] = self.rel_observation(uav_id, target_id)[:2]

        # Add observations for each UAV-charging station
        for uav_id in range(self.m):
            # 이건  uav_charging에 따른 obs 결과: 거리, 각도
            # charge station에 대한 거리.. 이걸로 해야 하는 거 아닌가?
            key = f"uav{uav_id+1}_charge_station"
            dictionary_obs[key] = self.uavs[uav_id].obs[:2]

        # Add observation for battery levels and ages of targets
        dictionary_obs["battery"] = np.float32([self.uavs[uav_id].battery for uav_id in range(self.m)])
        dictionary_obs["age"] = np.float32([self.targets[target_id].age for target_id in range(self.n)])

        return dictionary_obs
if __name__ == '__main__':
    main=MUMT_v1()
    main.reset()
