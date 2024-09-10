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
from sensor_msgs.msg import Imu, BatteryState, NavSatFix
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
HIGHER_LEVEL_FREQUENCY = 1
LOWER_LEVEL_FREQUENCY = 10
def wrap(theta):
    if theta > math.pi:
        theta -= 2*math.pi
    elif theta < -math.pi:
        theta += 2*math.pi
    return theta
'''
        UAV STATE는 절대좌표계 x, y값 및 Yaw Angle
        나머지 처리되는 계산 각도들은 전부 X축 기준 각도
'''
class UAV:
    def __init__(self, ns, state, v=17, battery=None):
        #TODO(1) UAV CLASS INITIALIZATION, UTILS FUNCTION
            # node_name = f"{ns.replace('/', '_')}_controller"  # '/'를 '_'로 대체하여 노드 이름 생성
            # rospy.init_node(node_name, anonymous=True)
            self. v = v
            # uav0, uav1, uav2 -> 0, 1, 2 [ns] 형식
            self.uav_id = int(re.findall(r'\d+', ns)[0]) # get uav_id from namespace
            self.dt = 0.05
            self.state = state
            self.battery = battery
            self.charging = 0
            self.pose = PoseStamped() # Goal Position
            self.attitude_target=  AttitudeTarget()
            self.ns = ns #uav0, uav1, uav2 형식
            self.local_position_boolean = None
            self.current_state = State()
            self.local_position = PoseStamped()
            self.current_state = State()
            self.offb_set_mode = SetModeRequest()
            self.landing_mode_request = SetModeRequest()
            self.arm_cmd = CommandBoolRequest()
            self.arm_cmd.value = True
            self.imu_data = Imu()
            self.local_velocity = TwistStamped()
            self.global_velocity = TwistStamped()
            self.gps_data = NavSatFix()
            self.current_velocity = None
            self.heading_data = Float64()
            rate= rospy.Rate(20)
            self.vel_target = TwistStamped()
            self.landing_request = CommandTOL()
            self.is_landed = False
            self.request_land = False
            self.previous_upper_action = 0 # Previous Action if action==-1
            self.previous_lower_action = None
            self.start_land = False
            # SUBSCRIBER
            self.velocity_sub = rospy.Subscriber(f'{self.ns}/mavros/local_position/velocity', TwistStamped, self.velocity_cb)
            self.state_sub = rospy.Subscriber(f"{self.ns}/mavros/state", State, self.state_cb, queue_size=10)
            self.attitude_sub = rospy.Subscriber(f"{self.ns}/mavros/setpoint_raw/attitude", AttitudeTarget, self.update_orientation, queue_size=10)
            self.pose_sub = rospy.Subscriber(f"{self.ns}/mavros/local_position/pose", PoseStamped, self.update_pose, queue_size=10)
            self.gps_sub = rospy.Subscriber(f"{self.ns}/mavros/global_position/global", NavSatFix, self.gps_cb, queue_size=10)
            self.heading_sub = rospy.Subscriber(f"{self.ns}/mavros/global_position/compass_hdg", Float64, self.orientation_cb)
            # PUBLISHER
            self.local_pos_pub = rospy.Publisher(f"{self.ns}/mavros/setpoint_position/local", PoseStamped, queue_size=10)
            self.local_vel_pub = rospy.Publisher(f"{self.ns}/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)
            self.attitude_pub = rospy.Publisher(f"{self.ns}/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
            # Service proxies # UAV
            self.arming_client = rospy.ServiceProxy(f'{self.ns}/mavros/cmd/arming', CommandBool)
            self.set_mode_client = rospy.ServiceProxy(f'{self.ns}/mavros/set_mode', SetMode)
            self.land_service=rospy.ServiceProxy(f"{self.ns}/mavros/cmd/land", CommandTOL)
            #self.attitude_pub = rospy.Publisher(f"{self.ns}/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
    def orientation_cb(self, data):
        self.heading_data = data
        #rospy.loginfo(f"UAV {self.uav_id} heading data: {self.heading_data.data}")
    def velocity_cb(self, data):
        rospy.loginfo("Velocity callback triggered")

        self.vel_target.twist.linear.x = data.twist.linear.x
        self.vel_target.twist.linear.y = data.twist.linear.y
        self.vel_target.twist.linear.z = data.twist.linear.z
        self.vel_target.twist.angular.z = data.twist.angular.z
        self.current_velocity = data.twist
        print("current velocity", np.sqrt(self.current_velocity.linear.x**2+self.current_velocity.linear.y**2))


    def gps_cb(self, data):
        self.gps_data = data

    def state_cb(self, data):
            self.current_state = data
    def update_orientation(self, msg):
            self.attitude_target = msg

    def update_pose(self, msg):
            self.local_position = msg # Current Position
            self.pose = msg
            self.local_position_boolean = True
            #print("GET MESSAGE")
        # Setpoint Publishing MUST be faster than 2Hz
    def offboard(self):
        rospy.wait_for_service(f'{self.ns}/mavros/set_mode')
        set_mode_request = SetModeRequest()
        set_mode_request.base_mode = 0
        set_mode_request.custom_mode = 'OFFBOARD'
        self.set_mode_client(set_mode_request)
        response = self.set_mode_client(set_mode_request)
        if response.mode_sent:
            rospy.loginfo(f"{self.ns}: Mode successfully set to OFFBOARD")
            return True
        else:
            rospy.logwarn(f"{self.ns}: Failed to set mode to OFFBOARD")
            return False


    def arming(self):
        rospy.wait_for_service(f'{self.ns}/mavros/cmd/arming')
        self.arm_cmd.value = True
        while not self.current_state.armed:
            self.arming_client(True)
            print(f"########{self.ns}{self.arming_client(True)}")
            if self.arming_client(True) == True:
                rospy.loginfo(f'--{self.ns} ready to fly')
                return True
            rospy.sleep(0.1)
        return False
    @property
    def obs(self):
            uav_x = self.local_position.pose.position.x
            uav_y = self.local_position.pose.position.y
            euler_theta = self.heading_data.data*pi/180
            target_x = 0
            target_y = 0
            x = uav_x - target_x
            y = uav_y - target_y
            r = np.sqrt(x**2 + y**2) # self.state[2] = theta
            beta = arctan2(y,x)
            theta = self.convert_angle_from_euler(euler_theta)
            alpha = wrap(beta - theta - math.pi)
            return np.array([r, alpha, beta], dtype=np.float32)

    def copy(self):
            return UAV(ns=self.ns, state=self.state.copy(), v=self.v, battery=self.battery)

    def move(self):
        # self.state[0] : x 좌표
        # self.state[1] : y 좌표
        # self.state[2] : orientation 정보
        self.state[0] = self.local_position.pose.position.x
        self.state[1] = self.local_position.pose.position.y
        yaw = self.heading_data.data * pi / 180 # Degree to Radian
        self.state[2] = yaw
    def convert_angle_from_euler(self, theta):
        euler = pi/2 - theta
        if euler > pi:
            euler -= 2*pi
        elif euler < -pi:
            euler += 2*pi
        return euler
class Target:
        _id_counter = 0
        max_age = 8*3600
        def __init__(self, state, age=0, initial_beta = 0, initial_r = 1000, target_type = 'static', sigma_rayleigh = 0.5, m=None, seed = None ):
            self.dt = 0.05
            self.state = state
            # self.max_age = 72*3600
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
            return Target(state = self.state.copy(), age=self.age, initial_beta = self.initial_beta, target_type = self.target_type, sigma_rayleigh = self.sigma_rayleigh)
        def cal_age(self, duration_time):
            if self.surveillance == 0:
                self.age = min(self.max_age, self.age + duration_time) # 1초에 한 번씩이지? 0.05 is Highr level control frequency
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