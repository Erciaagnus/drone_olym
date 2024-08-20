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
gym_setting_path = os.path.join(current_file_path, '../../gym_setting/mdp')
sys.path.append(os.path.abspath(gym_setting_path))
print(sys.path)
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
    metadata = {"render_modes": [ "human", "rgb_array"], "render_fps:":30}
    # Define UAV Agent
    class UAV:
        def __init__(self, state, v=1.0, battery=None):
            self.v = v
            self.dt =0.05
            self.state = state
            self.battery = battery
            self.charging = 0
        def copy(self):
            return SUST_v3.UAV(state=self.state.copy(), v=self.v, battery=self.battery)
        # state = [x, y, angle]
        def move(self, action):
            dtheta = action*self.dt
            _lambda = dtheta / 2
            if _lambda == 0.0:
                self.state[0] += self.v * self.dt *cos(self.state[-1])
                self.state[1] += self.v * self.dt *sin(self.state[-1])
            else:
                ds = self.v*self.dt*sin(_lambda) / _lambda
                self.state[0] += ds*cos(self.state[-1] + _lambda)
                self.state[1] += ds*sin(self.state[-1] + _lambda)
                self.state[2] += dtheta
                self.state[2] = wrap(self.state[2])
        @property
        def obs(self):
            x, y =self.state[:2]
            r = np.sqrt(x**2+y**2)
            alpha = wrap(np.arctan2(y,x) - wrap(self.state[-1])-pi)
            beta =arctan2(y, x)
            return array([r, alpha, beta], dtype=np.float32)
        # obs[uav] = [distance, angle btwn target and agent, angle about agent]
    class Target:
        _id_counter = 0
        def __init__(self, state, age=0, initial_beta=0, initial_r=30, target_type = 'static', sigma_rayleigh=0.5, m=None, seed=None ):
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
            self.positions =[]
            type(self)._id_counter += 1
            self.id = type(self)._id_counter
            self.step_idx = 0
            # 각가속도(v/r)에다가 미소 시간을 곱해서 미소 각도를 만듦
            self.angle_radians = self.target_v*self.dt/self.initial_r
            # 회전 매트릭스 -> 왜 필요하지?
            self.rotation_matrix = np.array([
                [np.cos(self.angle_radians), -np.sin(self.angle_radians)],
                [np.sin(self.angle_radians), np.cos(self.angle_radians)]
            ])
        def copy(self):
            return SUST_v3(state=self.state.copy(), age=self.age, initial_beta=self.initial_beta, target_type=self.target_type, sigma_rayleigh=self.sigma_rayleigh)
        def cal_age(self):
            if self.surveillance == 0:
                self.age = min(1000, self.age+1)
            else:
                self.age = 0
        # 아래는 target type이  load, both, deterministic, rayleigh일 때 위치가 계속 바뀌는 것을 의미한다.
        # 우리는 'static'이므로 해당사항이 없다.
        def update_position(self):
            if self.target_type == 'load':
                try:
                    trajectory_array = np.load()
                except Exception as e: print(e)
                if trajectory_array.ndim > 2:
                    self.state = trajectory_array[self.id][self.step_idx]
                else:
                    self.state = trajectory_array[self.step_idx]
                    # self.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)
                self.step_idx += 1
            if self.target_type in ('deterministic', 'both'):
                if self.id % 2 == 0:
                    if 0 <= self.time_elapsed < 10:
                        theta_T = np.cos(pi*self.time_elapsed / 10)
                    elif 10 <= self.time_elapsed <25:
                        theta_T = -pi / 4
                    elif 25 <= self.time_elapsed <55:
                        theta_T = pi / 4
                    elif 55 <= self.time_elapsed <100:
                        theta_T = np.cos(pi*self.time_elapsed / 5) - pi/8
                    else:
                        theta_T = np.cos((pi/10 - 0.005*(self.time_elapsed - 100))*self.time_elapsed)
                    heading_angle = wrap(self.initial_beta - np.pi)
                    dx = self.target_v*np.cos(theta_T)*self.dt
                    dy = self.target_v*np.sin(theta_T)*self.dt
                    dx1, dy1 = dx*np.cos(heading_angle) - dy*np.sin(heading_angle), dx*np.sin(heading_angle) + dy*np.cos(heading_angle)
                    self.state += np.array([dx1, dy1])
                else:
                    self.state = np.dot(self.rotation_matrix, self.state)
            if self.target_type in ('rayleigh', 'both'):
                speed = np.random.rayleigh(self.sigma_rayleigh)
                angle = np.random.uniform(0, 2*np.pi)
                dx = speed*np.cos(angle)*self.dt
                dy = speed*np.sin(angle)*self.dt
                self.state += np.array([dx, dy])
            self.positions.append(self.state)
        @property
        def obs(self):
            x, y = self.state
            r = np.sqrt(x**2 + y**2)
            beta = np.arctan2(y,x)
            return np.array([r, beta])
    ## CALLBACK FUNCTION ##
    def lp_cb(self, data):
        self.uav_position = data.pose.position
        self.uav_orientation = data.pose.orientation
    # Percentage로 받으면 안되나??
    def bs_cb(self, data):
        self.battery_state = data
    def __init__(self, render_mode: Optional[str]=None,
                 r_max = 80, r_min=0, dt=0.05, d=10.0, l=3, m=1, n=2, r_c=3, max_step=6000, seed=None):
        super().__init__()
        ######################### ROS SERVICE #########################
        rospy.init_node("uav_env", anonymous= True)
        rospy.Subscriber("mavros/state", State, self.state_cb)

        rate = rospy.Rate(20)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        try:
            arm_cmd = CommandBoolRequest(value=True)
            arming_client(arm_cmd.value)
        except rospy.ServiceException as e:
            rospy.logerr("Arming service call failed %s" %e)

        local_pos_pub = rospy.Publisher("mavros/setpoint_position/local",PoseStamped, queue_size=10)
        arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        land_client = rospy.ServiceProxy("/mavros/cmd/land", CommandTOL)

        # Subscriber
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_cb, queue_size=10)
        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.imu_cb, queue_size=10)
        self.local_pos_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.lp_cb, queue_size=10)
        self.local_vel_sub = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self.lv_cb, queue_size=10)
        self.act_control_sub = rospy.Subscriber("/mavros/act_control/act_control_pub", ActuatorControl, self.act_cb, queue_size=10)
        self.global_alt_sub = rospy.Subscriber("/mavros/global_position/rel_alt", Float64, self.ra_cb, queue_size=10)
        self.global_pos_sub = rospy.Subscriber("/mavros/global_position/gp_vel", TwistStamped, self.gv_cb, queue_size=10)
        self.battery_state_sub = rospy.Subscriber("/mavros/battery", BatteryState, self.bs_cb)

        self.uav_position = None
        self.uav_orientation = None
        self.battery_state = None

        ## define ROS message
        self.offb_set_mode =SetModeRequest()
        self.offb_set_mode.custom_mode = "OFFBOARD"

        self.arm_cmd = CommandBoolRequest()
        self.arm_cmd.value = True
        self.disarm_cmd = CommandBoolRequest()
        self.disarm_cmd.value = False

        ## ROS Services
        rospy.wait_for_service('mavros/cmd/arming')
        self.arming_client = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)

        rospy.wait_for_service('mavros/set_mode')
        self.set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode)

        rospy.wait_for_service('mavros/set_str3eam_rate')
        set_stream_rate = rospy.ServiceProxy("mavros/set_stream_rate", StreamRate)

        set_stream_rate(StreamRateRequest.STREAM_POSITION, 50, 1)
        set_stream_rate(StreamRateRequest.STREAM_ALL, 50, 1)
        self.setpoint_msg = mavros.setpoint.PoseStamped(header=mavros.setpoint.Header(frame_id="att_pose",stamp=rospy.Time.now()),)

        #self.offb_arm()
        ########################################################################
        self.render_mode = render_mode
        self.seed= seed
        obs_space = {}
        '''
        observation space
        1. uav_obs = [x, y, heading angle]
        2. target_obs = [ x, y, beta ]
        3. Total obs = [ alpha_t, alpha_c, battery, age]
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
        # 1. uav_id - target_id, 2. uav_id - charge station, 3. battery, 4. age
        # Define Observation Space, and Action Space
        self.observations_space = Dict(obs_space)
        self.action_space = MultiDiscrete([n+1]*m, seed=self.seed) # Action 0: charge, 1: 1target 2: 2target..
        
        self.discount = 0.999
        self.r_max = r_max
        self.d = d
        self.l = l
        self.m = m
        self.uavs=[]
        self.n = n
        self.targets =[]
        self.r_c = r_c
        self.step_count = None
        self.num2str = {0: "charge", 1: "target1"}
        self.max_step = max_step
        self.episode_counter = 0
        self.frame_counter =0
        self.n_r = 800
        self.n_alpha = 360
        self.n_u =2

        # UPLOAD DKC and TOD Policy Learning Results.
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        self.distance_keeping_resutls = np.load(current_file_path + os.path + "80_dkc_result_0.0.npz")
        self.distance_keeping_straightened_policy00 = self.distance_keeping_resutls["policy"]
        self.time_optimal_straightened_policy00 = np.load(current_file_path + os.path.sep + "80_toc_result_0.0.npz")
        # Define States
        self.states = States(
            np.linspace(0.0, 80.0, self.n_r, dtype=np.float32),
            np.linspace(
                -np.pi, np.pi - np.pi /self.n_alpha,
                self.n_alpha,
                dtype=np.float32
            ),
            cycles = [np.inf, np.pi*2],
            n_alpha=self.n_alpha
        )
        self.actions = Actions(
            np.linspace(-1.0 / 4.5, 1.0 / 4.5, self.n_u, dtype=np.float32).reshape(
                (-1, 1)
            )
        )
        self.uav_trajectory_data = [[] for _ in range(self.m)]
        self.target_trajectory_data = [[] for _ in range(self.n)]
        
    def reset(
        self, uav_pose = None, target_pose=None, batteries=None, ages=None, target_type = 'static',
        sigma_rayleigh = 0.5, seed: Optional[int] = None, options : Optional[dict] = None,
    ):
        self.uavs = []
        self.targets=[]
        self.uav_trajectory_data = [[] for _ in range(self.m)]
        self.target_trajectory_data = [[] for _ in range(self.n)]
        np.random.seed(seed)
        self.seed = seed
        self.episode_counter += 1
        self.step_count = 0
        if self.save_frames:
            os.makedirs(
                os.path.join(self.SAVE_FRAMES_PATH, f"{self.episode_counter:03d}"),
                exist_ok=True,
            )
            self.frame_counter = 0
        if uav_pose is None:
            # 초기 UAV 위치를 정하는 거 같은데 우린 이렇게 할 필요가 없지.
            # uav_r = np.random.uniform(0,40,self.m)
            # uav_beta = np.random.uniform(-pi, pi, self.m)
            # uav_theta = np.random.uniform(-pi, pi, self.m)
            # uav_x = uav_r * np.cos(uav_beta)
            # uav_y = uav_r * np.cos(uav_beta)
            uav_theta = self.uav_orientation
            uav_x = self.local_pos_sub.pose.position.x
            uav_y = self.local_pos_sub.pose.position.y
            uav_states = np.vstack([uav_x, uav_y, uav_theta]).T
        else:
            # uav_pose 가 None이 아니라면, 즉 초기값이 아니라면
            uav_states = uav_pose
        if batteries is None:
            # You need to check
            batteries = self.battery_state
        else:
            # 지금 1개의 UAV라 그런데, 만약 여러대의 UAV의 경우 가져오는 정보를 어떻게 처리할 건지가 필요..
            batteries = batteries
        for i in range(self.m):
            self.uavs.append(self.UAV(state = uav_states[i], battery = batteries[i]))
        for uav_idx, uav in enumerate(self.uavs):
            uav_x, uav_y, uav_theta = self.UAV.state
            uav_battery_level = uav.battery
            self.uav_trajectory_data[uav_idx].append((uav_x, uav_y, uav_battery_level, uav_theta))
        if target_pose is None:
            # 30, 35 mean?
            target1_r = np.random.uniform(30, 35, self.n)
            target1_beta = np.random.uniform(-np.pi, np.pi, self.n)
            target_states = np.array([target1_r*np.cos(target1_beta), target1_r*np.sin(target1_beta)]).T
            ages = [0]*self.n
        else:
            target_states, ages = target_pose

        for i in range(self.n):
            self.targets.append(self.Target(state=target_states[i], age=ages[i],
                                            initial_beta=target1_beta[i], initial_r=target1_r[i],
                                            target_type=target_type, sigma_rayleigh=sigma_rayleigh,
                                            m=self.m, seed=self.seed,))
        for target_idx, target in enumerate(self.targets):
            target_x, target_y = target.state
            self.target_trajectory_data[target_idx].append((target_x, target_y))
        return self.dict_observation, {}

    def toc_get_action(self, state):
        S, P = self.states.computeBarycentric(state)
        action = sum(p * self.actions[int(self.time_optimal_straightened_policy00[s])] for s, p in zip(S, P))
        return action

    def dkc_get_action(self, state):
        S, P = self.states.computeBarycentric(state)
        action = sum(p * self.actions[int(self.distance_keeping_straightened_policy00[s])] for s, p in zip(S, P))
        return action
    def control_uav(self, uav_idx, action):
        self.uavs[uav_idx].charging = 0
        if self.uavs[uav_idx].battery <= 0:
            pass
        else:
            if action == 0:
                if (self.uavs[uav_idx].obs[0] < self.r_c):
                    self.uavs[uav_idx].charging =1
                    self.uavs[uav_idx].battery = min(self.uavs[uav_idx].battery + 10, 3000)
                else:  # not able to land on charge station(too far)
                    self.uavs[uav_idx].battery -= 1
                    w1_action = self.toc_get_action(self.uavs[uav_idx].obs[:2])
                    self.uavs[uav_idx].move(w1_action)
            else:  # surveil target1
                self.uavs[uav_idx].battery -= 1
                w1_action = self.dkc_get_action(self.rel_observation(uav_idx, action-1)[:2])
                self.uavs[uav_idx].move(w1_action)
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
    def step(self, action):
        terminal = False
        truncated = False
        action = np.squeeze(action)
        reward = 0
        if action.ndim == 0:
            action = np.expand_dims(action, axis = 0)
        for uav_idx, uav_action in enumerate(action):
            self.control_uav(uav_idx, uav_action)
        surveillance_matrix = np.zeros((self.m, self.n))
        for uav_idx in range(self.m):
            for target_idx in range(self.n):
                surveillance_matrix[uav_idx, target_idx] = self.cal_surveillance(uav_idx, target_idx)
        surveillance = np.any(surveillance_matrix, axis=0).astype(int)
        for target_idx in range(self.n):
            self.targets[target_idx].surveillance = surveillance[target_idx]
            self.targets[target_idx].cal_age()
            reward += -self.targets[target_idx].age
        reward = reward / self.n # average reward of all targets

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

    def rel_observation(self, uav_idx, target_idx): # of target relative to uav
        uav_x, uav_y, theta = self.uavs[uav_idx].state
        target_x, target_y = self.targets[target_idx].state
        x = target_x - uav_x
        y = target_y - uav_y
        r = np.sqrt(x**2 + y**2)
        beta = arctan2(y, x)
        alpha = wrap(beta - wrap(theta))
        return array([r, alpha, beta],dtype=np.float32)

    @property
    def dict_observation(self):
        dictionary_obs = {}
        # Add observations for UAV-target pairs according to the rule
        for uav_id in range(self.m):
            for target_id in range(self.n):
                key = f"uav{uav_id+1}_target{target_id+1}"
                dictionary_obs[key] = self.rel_observation(uav_id, target_id)[:2]

        # Add observations for each UAV-charging station
        for uav_id in range(self.m):
            key = f"uav{uav_id+1}_charge_station"
            dictionary_obs[key] = self.uavs[uav_id].obs[:2]

        # Add observation for battery levels and ages of targets
        dictionary_obs["battery"] = np.float32([self.uavs[uav_id].battery for uav_id in range(self.m)])
        dictionary_obs["age"] = np.float32([self.targets[target_id].age for target_id in range(self.n)])

        return dictionary_obs
if __name__ == '__main__':
    env=SUST_v3()
    rospy.sleep(1)
    obs = env.reset()
    done = False
    total_reward = 0
    while not done and not rospy.is_shutdown():
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            print(f"Episode finished with total reward: {total_reward}")
            break
        rospy.sleep(0.1)