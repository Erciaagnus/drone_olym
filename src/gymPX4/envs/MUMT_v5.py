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
from gym.envs.registration import register
#import rendering

def wrap(theta):
    if theta > pi:
        theta -= 2 * pi
    elif theta < -pi:
        theta += 2 * pi
    return theta
class MUMT_v5(Env):
    metadata = {"render_modes": [ "human", "rgb_array"], "render_fps":30}
    # Define UAV Agent
    class UAV:
        def __init__(
            self, state, v= 1.0, battery =3000,):
            self.v = v
            self.dt=0.05
            self.state= state
            self.battery = battery
            self.charging = 0
        def copy(self):
            # Define Env ( state, v, battery ) -> What means v?
            return MUMT_v5.UAV(state=self.state.copy(), v=self.v, battery=self.battery)
        def move(self, action):
            dtheta = action*self.dt # differentiate of angular
            _lambda = dtheta/2
            if _lambda == 0.0:
                self.state[0] += self.v*self.dt * cos(self.state[-1])
                self.state[1] += self.v*self.dt * sin(self.state[-1])
            else:
                ds = self.v*self.dt * sin(_lambda) / _lambda
                self.state[0] += ds * cos(self.state[-1] + _lambda)
                self.state[1] += ds * sin(self.state[-1] + _lambda)
                self.state[2] += dtheta
                self.state[2] = wrap(self.state[2])
        @property
        def obs(self): # observation of uav relative to charging station
            x, y = self.state[:2]
            r = np.sqrt(x**2 + y**2)
                        # beta                  # theta
            alpha = wrap(arctan2(y, x) - wrap(self.state[-1]) - pi)
            beta = arctan2(y, x)
            return array([r, alpha, beta], dtype=np.float32)  # beta

    # In Envrionent Define Target Property * Update
    class Target:
        _id_counter = 0
        def __init__(self, state, age=0, initial_beta=0, initial_r=30, target_type='static', sigma_rayleigh=0.5, m=None, seed=None,):
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
            # self.id = random.randint(0,1)
            self.step_idx = 0
            self.angle_radians = self.target_v*self.dt/self.initial_r
            self.rotation_matrix = np.array([
                                [np.cos(self.angle_radians), -np.sin(self.angle_radians)],
                                [np.sin(self.angle_radians), np.cos(self.angle_radians)]
                                ])

        def copy(self):
            # Assuming the copy method is intended to create a copy within the same parent environment
            return MUMT_v5.Target(state=self.state.copy(), age=self.age, initial_beta=self.initial_beta, target_type=self.target_type, sigma_rayleigh=self.sigma_rayleigh)

        def cal_age(self):
            if self.surveillance == 0:  # UAV is not surveilling
                self.age = min(1000, self.age + 1)  # Change age
            else:
                self.age = 0

        def update_position(self):
            if self.target_type == 'load':
                filename = f'traj/target_trajectory_m:{self.m}_seed:{self.seed}.npy'
                try:
                    trajectory_array = np.load(filename)
                except Exception as e: print(e)
                if trajectory_array.ndim > 2:
                    self.state = trajectory_array[self.id][self.step_idx]
                else:
                    self.state = trajectory_array[self.step_idx]
                # print('state: ',self.state)
                self.step_idx += 1
            if self.target_type in ('deterministic', 'both'):
                if self.id % 2 == 0:
                    # Define theta_T based on the current time_elapsed
                    if 0 <= self.time_elapsed < 10:
                        theta_T = np.cos(pi * self.time_elapsed / 10)
                    elif 10 <= self.time_elapsed < 25:
                        theta_T = -pi / 4
                    elif 25 <= self.time_elapsed < 55:
                        theta_T = pi / 4
                    elif 55 <= self.time_elapsed < 100:
                        theta_T = np.cos(pi * self.time_elapsed / 5) - pi / 8
                    else:
                        theta_T = np.cos((pi / 10 - 0.005 * (self.time_elapsed - 100)) * self.time_elapsed)

                    # Update position based on theta_T
                    heading_angle = wrap(self.initial_beta-np.pi)
                    dx = self.target_v*np.cos(theta_T) * self.dt
                    dy = self.target_v*np.sin(theta_T) * self.dt
                    dx1, dy1 = dx*np.cos(heading_angle) - dy*np.sin(heading_angle), dx*np.sin(heading_angle) + dy*np.cos(heading_angle)
                    self.state += np.array([dx1, dy1])
                    # Update the time elapsed
                    self.time_elapsed += self.dt
                else:
                    # rotate the target
                    self.state = np.dot(self.rotation_matrix, self.state)

            if self.target_type in ('rayleigh', 'both'):
                speed = np.random.rayleigh(self.sigma_rayleigh)
                angle = np.random.uniform(0, 2*np.pi)
                dx = speed * np.cos(angle) * self.dt
                dy = speed * np.sin(angle) * self.dt
                self.state += np.array([dx, dy])
            self.positions.append(self.state)
            # elif self.target_type == 'cross':
            #     speed = np.random.rayleigh(self.sigma_rayleigh)
            #     angle = np.random.uniform(0, 2*np.pi)
            #     dx = speed * np.cos(angle) * self.dt
            #     dy = speed * np.sin(angle) * self.dt
            #     self.state += np.array([dx, dy])


        @property
        def obs(self):
            x, y = self.state
            r = np.sqrt(x**2 + y**2)
            beta = np.arctan2(y, x)
            return np.array([r, beta])  # r, beta
    ## Define the Environments
    # Initialize
    def __init__(
        self,
        render_mode: Optional[str] = None,
        r_max=80,
        r_min=0,
        dt=0.05,
        d=10.0,
        l=3, # noqa
        m=2, # of uavs
        n=2, # of targets
        r_c=3,
        max_step=6000,
        seed = None # one circle 1200 time steps
    ):
        super().__init__()
        self.render_mode = render_mode
        self.seed = seed
        # Create the observation space
        obs_space = {}

        # Add observation spaces for each UAV-target pair according to the rule
        for uav_id in range(1, m + 1):
            for target_id in range(1, n + 1):
                key = f"uav{uav_id}_target{target_id}"
                obs_space[key] = Box(low=np.float32([r_min, -np.pi]),
                                        high=np.float32([r_max, np.pi]),
                                        dtype=np.float32)

        # Add observation spaces for each UAV-charging station
        for uav_id in range(1, m + 1):
            key = f"uav{uav_id}_charge_station"
            obs_space[key] = Box(low=np.float32([r_min, -np.pi]),
                                 high=np.float32([r_max, np.pi]),
                                 dtype=np.float32)

        # Add observation space for battery and age
        # Assuming one battery value per UAV and one age value per target
        obs_space["battery"] = Box(low=np.float32([0]*m),
                                   high=np.float32([3000]*m),
                                   dtype=np.float32)
        obs_space["age"] = Box(low=np.float32([0]*n),
                               high=np.float32([1000]*n),
                               dtype=np.float32)

        self.observation_space = Dict(obs_space)
        self.action_space = MultiDiscrete([n + 1] * m, seed=self.seed)
        self.dt = dt
        self.discount = 0.999
        self.r_max = r_max
        self.d = d  # target distance
        self.l = l  # coverage gap: coverage: d-l ~ d+l # noqa
        self.m = m  # of uavs
        self.uavs = []
        self.uav_color = [(random.randrange(0, 11) / 10, random.randrange(0, 11) / 10, random.randrange(0, 11) / 10) for _ in range(m)]
        self.n = n  # of targets
        self.targets = []
        self.trajectory_data = []
        self.r_c = r_c  # charge station radius
        self.step_count = None
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)
        self.num2str = {0: "charge", 1: "target_1"}
        self.max_step = max_step
        self.viewer = None
        self.SAVE_FRAMES_PATH = "../../../../visualized/MUMT" # example. save frames path is set at surveillance_PPO.py
        self.episode_counter = 0
        self.frame_counter = 0
        self.save_frames = False
        # self.q_init()

        # initialization for Dynamic Programming
        self.n_r = 800
        self.n_alpha = 360
        self.n_u = 2 #21

        current_file_path = os.path.dirname(os.path.abspath(__file__))
        self.distance_keeping_result00 = np.load(current_file_path+ os.path.sep + "v1_80_2a_dkc_val_iter.npz")
        self.distance_keeping_straightened_policy00 = self.distance_keeping_result00["policy"] # .data
        self.time_optimal_straightened_policy00 = np.load(current_file_path+ os.path.sep + "v1_terminal_40+40_2a_toc_policy_fp64.npy")
        # print('shape of time_optimal_straightened_policy00',np.shape(self.time_optimal_straightened_policy00)) # 288000

        self.states = States(
            np.linspace(0.0, 80.0, self.n_r, dtype=np.float32),
            np.linspace(
                -np.pi,
                np.pi - np.pi / self.n_alpha,
                self.n_alpha,
                dtype=np.float32,
            ),
            cycles=[np.inf, np.pi * 2],
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
        self,
        uav_pose=None,
        target_pose=None,
        batteries=None,
        ages=None,
        target_type = 'static',
        sigma_rayleigh=0.5,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.uavs = []
        self.targets = []
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
            uav_r = np.random.uniform(0, 40, self.m)  # D=40
            uav_beta = np.random.uniform(-pi, pi, self.m)
            uav_theta = np.random.uniform(-pi, pi, self.m)
            # Create the state arrays
            uav_x = uav_r * np.cos(uav_beta)
            uav_y = uav_r * np.sin(uav_beta)

            # Stack them into a single array
            uav_states = np.vstack([uav_x, uav_y, uav_theta]).T  # Transpose to get the correct shape
        else:
            uav_states = uav_pose
        if batteries is None:
            batteries = np.random.randint(1500, 3000, self.m)
        else:
            batteries = batteries
        # Create UAV instances
        for i in range(self.m):
            self.uavs.append(self.UAV(state=uav_states[i], battery=batteries[i]))
        for uav_idx, uav in enumerate(self.uavs):  # Replace 'self.uavs' with how you access your UAVs
            uav_x, uav_y, uav_theta = self.UAV.state  # Replace with actual position attributes
            uav_battery_level = uav.battery  # Replace with actual battery attribute
            self.uav_trajectory_data[uav_idx].append((uav_x, uav_y, uav_battery_level, uav_theta))

        if target_pose is None:
            target1_r = np.random.uniform(30, 35, self.n)  # 0~ D-d
            target1_beta = np.random.uniform(-np.pi, np.pi, self.n)
            target_states = np.array([target1_r * np.cos(target1_beta), target1_r * np.sin(target1_beta)]).T
            ages = [0] * self.n
        else:
            target_states, ages = target_pose  # Assuming target_pose is an iterable of target states

        # Create Target instances
        for i in range(self.n):
            self.targets.append(self.Target(state=target_states[i], age=ages[i],
                                            initial_beta=target1_beta[i], initial_r=target1_r[i],
                                            target_type=target_type, sigma_rayleigh=sigma_rayleigh,
                                            m=self.m, seed=self.seed,))
        for target_idx, target in enumerate(self.targets):
            target_x, target_y = target.state
            self.target_trajectory_data[target_idx].append((target_x, target_y))
        return self.dict_observation, {}

    def q_init(self):
        self.n_alpha = 10
        # simple version
        # self.target_discretized_r_space = np.arange(0,81,20)
        # self.charge_discretized_r_space = np.arange(0,51,10)
        # self.target_alpha_space = np.linspace(-np.pi, np.pi - np.pi / self.n_alpha, self.n_alpha, dtype=np.float32)
        # self.charge_alpha_space = np.linspace(-np.pi, np.pi - np.pi / self.n_alpha, self.n_alpha, dtype=np.float32)
        # self.battery_space = np.arange(0, 3100,1000)

        # complex version
        self.target_discretized_r_space = np.array([0,4,6,8,10,12,14,16,20,30,40,60,80])
        self.charge_discretized_r_space = np.array([0,2,3,5,6,10,20,30,40])
        self.target_alpha_space = np.linspace(-np.pi, np.pi - np.pi / self.n_alpha, self.n_alpha, dtype=np.float32)
        self.charge_alpha_space = np.linspace(-np.pi, np.pi - np.pi / self.n_alpha, self.n_alpha, dtype=np.float32)
        self.battery_space = np.concatenate([np.arange(0, 500, 100), np.arange(500, 3100, 500)])

        self.age_space = np.arange(0, 1001, 100) #changeage
        self.UAV1Target1_result00 = np.load(current_file_path + f"/1U1T_s6_age1000:100_gamma_{self.discount}_dt_{self.dt}_{'val'}_iter.npz")
        self.UAV1Target1_straightened_policy00 = self.UAV1Target1_result00["policy"]
        self.UAV1Target1_values00 = self.UAV1Target1_result00["values"]
        # print('shape of UAV1Target1_straightened_policy00: ', np.shape(self.UAV1Target1_straightened_policy00))
        # print('shape of UAV1Target1_values00: ', np.shape(self.UAV1Target1_values00))

        self.uav1target1_states = States(
            # uav1_target1
            self.target_discretized_r_space, # [0]
            self.target_alpha_space,         # [1]
            # uav1_charge
            self.charge_discretized_r_space, # [2]
            self.charge_alpha_space,         # [3]
            # battery_state
            self.battery_space,              # [4]
            self.age_space,                  # [5]
            n_alpha=self.n_alpha,
            cycles=[np.inf, np.pi*2, np.inf, np.pi*2, np.inf, np.inf]
        )

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
        if self.uavs[uav_idx].battery <= 0: # UAV dead
            pass
        else: # UAV alive: can take action
            if action == 0:  # go to charging station
                if (self.uavs[uav_idx].obs[0] < self.r_c):
                    # uav1 no move
                    self.uavs[uav_idx].charging = 1
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

    def load_trajectories(self):
        filename = f'target_trajectory_m:{self.m}_seed:{self.seed}.npy'
        if os.path.exists(filename):
            trajectory_array = np.load(filename)
            return trajectory_array
        else:
            return None

    def apply_loaded_trajectories(self):
        loaded_trajectory = self.load_trajectories()
        if loaded_trajectory is not None:
            for step, positions in enumerate(loaded_trajectory):
                for idx, position in enumerate(positions):
                    self.targets[idx].position = position
                    self.targets[idx].state = position  # Assuming 'state' should reflect the current position


    def step(self, action):
        terminal = False
        truncated = False
        action = np.squeeze(action)
        reward = 0
        if action.ndim == 0:
            action = np.expand_dims(action, axis=0)
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

        if self.save_frames and int(self.step_count) % 6 == 0:
            image = self.render(action, mode="rgb_array")
            path = os.path.join(
                self.SAVE_FRAMES_PATH,
                f"{self.episode_counter:03d}",
                f"{self.frame_counter+1:04d}.bmp",
            )
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            # left upper corner
            text = [
                f"r_c: {self.uav1.obs[0]}, a_c: {self.uav1.obs[1]}",
                f"r_t: {self.target1.obs[0]}, a_t: {self.target1.obs[1]}",
                f"max_Q: {self.max_Q:.2f}",
                f"dQ: {self.max_Q - self.min_Q:.2f}",
                f"argmax(Q0,Q1): {self.argmax_Q}",
                "battery: {}".format(self.uav1.battery),
                "age: {}".format(self.target1.age),
                "Reward: {}".format(reward)
            ]
            for i, t in enumerate(text):
                draw.text((0, 20*i), t, color=(255, 255, 255), font=self.font)
                image.save(path)
            self.frame_counter += 1
        self.step_count += 1
        if self.step_count >= self.max_step:
            truncated = True
            self.trajectory_data.append(list(target.positions for target in self.targets))
            self.save_trajectories()
        return self.dict_observation, reward, terminal, truncated, {}

    def save_trajectories(self):
        trajectory_array = np.array(self.trajectory_data)
        # print(np.shape(trajectory_array))
        trajectory_array = trajectory_array.squeeze()
        # trajectory_array = trajectory_array.reshape((6000,2))
        # trajectory_array = trajectory_array[0]
        # print(np.shape(trajectory_array))
        # print(trajectory_array)
        filename = f'traj/target_trajectory_m:{self.m}_seed:{self.seed}.npy'
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory) 
        np.save(filename, trajectory_array)

    def dry_cal_surveillance(self, uav1_copy, target1_copy, r_t):
        if uav1_copy.battery <= 0: # UAV dead
            target1_copy.surveillance = 0
        else: # UAV alive
            if (self.d - self.l < r_t < self.d + self.l
                and uav1_copy.charging != 1):
                target1_copy.surveillance = 1
            else:
                target1_copy.surveillance = 0
        return target1_copy.surveillance

    def dry_step(self, uav_idx, target_idx, action, future, discount):
        # Copying relevant instance variables
        uav1_copy = self.uavs[uav_idx].copy()
        target1_copy = self.targets[target_idx].copy()
        step_count_copy = self.step_count

        terminal = False
        truncated = False
        action = np.squeeze(action)
        reward = 0

        dry_dict_observation = {}
        dry_dict_observation['uav1_target1'] = self.rel_observation(uav_idx, target_idx)[:2]
        dry_dict_observation['uav1_charge_station'] = uav1_copy.obs[:2]
        dry_dict_observation = self.dict_observation.copy()
        # print('Before:')
        # print('uav1_target1: ', dry_dict_observation['uav1_target1'])
        # print('uav1_charge_station: ', dry_dict_observation['uav1_charge_station'])
        for i in range(future):
            if truncated:
                break
            # Logic for UAV1's battery and actions
            uav1_copy.charging = 0
            if uav1_copy.battery <= 0:  # UAV dead
                pass
            else:
                if action == 0:
                    if (dry_dict_observation['uav1_charge_station'][0] < self.r_c):
                        uav1_copy.charging = 1
                        uav1_copy.battery = min(uav1_copy.battery + 10, 3000)
                    else:
                        uav1_copy.battery -= 1
                        w1_action = self.toc_get_action(dry_dict_observation['uav1_charge_station'][:2])
                        uav1_copy.move(w1_action)
                    
                else:
                    uav1_copy.battery -= 1
                    w1_action = self.dkc_get_action(dry_dict_observation['uav1_target1'][:2])
                    uav1_copy.move(w1_action)
            uav_x, uav_y, theta = uav1_copy.state
            target_x, target_y = target1_copy.state
            x = target_x - uav_x
            y = target_y - uav_y
            r_t = np.sqrt(x**2 + y**2)
            beta_t = arctan2(y, x)
            alpha_t = wrap(beta_t - wrap(theta))
            self.dry_cal_surveillance(uav1_copy, target1_copy, r_t)
            target1_copy.cal_age()

            step_count_copy += 1
            if step_count_copy >= self.max_step:
                truncated = True

            dry_dict_observation = { # is this state s_{t+10}?: Yes it is
                # r, alpha
                "uav1_target1": np.array([r_t, alpha_t], dtype=np.float32,),
                "uav1_charge_station": np.array([uav1_copy.obs[0], uav1_copy.obs[1]], dtype=np.float32,),
                "battery":  np.float32(uav1_copy.battery),
                "age": target1_copy.age,
                # "previous_action": action
            }
            reward += -target1_copy.age*discount**i
        return dry_dict_observation, reward, terminal, truncated, {}

    def battery_to_color(self, battery_level, max_battery=3000):
        red = max(1 - battery_level / max_battery, 0)
        green = min(battery_level / max_battery, 1)
        return (red, green, 0)

    def render(self, action, mode="human"):
        if self.viewer is None:
            self.viewer = rendering.Viewer(1000, 1000)
            bound = int(40 * 1.05)
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        # Render all self.targets
        for target_idx, target in enumerate(self.targets):
            target_x, target_y = target.state
            outer_donut = self.viewer.draw_circle(
                radius=self.d + self.l, x=target_x, y=target_y, filled=True
            )
            if target.surveillance == 1:
                outer_donut.set_color(0.6, 0.6, 1.0, 0.3)  # lighter
            else:
                outer_donut.set_color(0.3, 0.3, 0.9, 0.3)  # transparent blue
            inner_donut = self.viewer.draw_circle(
                radius=self.d - self.l, x=target_x, y=target_y, filled=True
            )
            inner_donut.set_color(0, 0, 0)  # erase inner part
            circle = self.viewer.draw_circle(
                radius=self.d, x=target_x, y=target_y, filled=False
            )
            circle.set_color(1, 1, 1)
            target_circle = self.viewer.draw_circle(
                radius=1, x=target_x, y=target_y, filled=True
            )
            if target_idx + 1 in action:
                try:
                    target_circle.set_color(*self.uav_color[int(np.where(action == target_idx + 1)[0])])  # yellow
                except:
                    target_circle.set_color(1, 1, 0)  # multiple uavs are after one target
            else:
                target_circle.set_color(1, 0.6, 0)  # orange

        # charge station @ origin
        charge_station = self.viewer.draw_circle(radius=self.r_c, filled=True)
        
        if 0 in action:
            try:
                charge_station.set_color(*self.uav_color[int(np.where(action == 0)[0])])  # yellow
            except:
                charge_station.set_color(1, 1, 0)  # multiple uavs are after the charge station
            for uav_idx, uav in enumerate(self.uavs):
                if uav.charging == 1:
                    charge_station.set_color(1,0,0)
                    break
        else:
            charge_station.set_color(0.1, 0.9, 0.1)  # green            

        # Render all self.uavs
        for uav_idx, uav in enumerate(self.uavs):
            if uav.battery <= 0:  # UAV dead
                continue
            uav_x, uav_y, uav_theta = uav.state
            uav_transform = rendering.Transform(translation=(uav_x, uav_y), rotation=uav_theta)
            uav_tri = self.viewer.draw_polygon([(-0.8, 0.8), (-0.8, -0.8), (1.6, 0)])
            # try:
            # uav_tri.set_color(*self.uav_color[uav_idx])
            uav_tri.set_color(*self.battery_to_color(uav.battery))
            uav_tri.add_attr(uav_transform)
            # except:
            #     print('len(self.uav_color): ',len(self.uav_color))
            #     print('uav_idx: ', uav_idx)
            #     input()
        # image = self.viewer.render(return_rgb_array=True)
        # image = Image.fromarray(image)
        # draw = ImageDraw.Draw(image)
        # # left upper corner
        # text0 = f"action: {action}"
        # # convert to %
        # text1 = f"battery: {self.dict_observation['battery']/3000}%"
        # text2 = "age: {}".format(self.dict_observation['age'])
        # draw.text((0, 0), text0, color=(200, 200, 200), font=self.font)
        # draw.text((0, 35), text1, color=(200, 200, 200), font=self.font)
        # draw.text((0, 70), text2, color=(255, 255, 0), font=self.font)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def plot_trajectory(self, policy):
        plt.figure(figsize=(10, 6))

        # Set the same scale for both axes
        plt.axis('equal')
        # print(self.target_trajectory_data)
        # input()

        # Draw targets and their coverage bands
        for target_idx, target_data in enumerate(self.target_trajectory_data):
            # Generate a color from red to yellow for each target
            target_color = plt.cm.spring(target_idx / len(self.targets))
            last_target_x, last_target_y = target_data[-1]
            # First draw target from last position
            plt.plot(last_target_x, last_target_y, marker='$\u2691$', color=target_color, markersize=15, 
                    label=f'Target {target_idx+1}', linestyle='None', zorder=5)
            
            # Draw the coverage bands
            coverage_band_outer = plt.Circle((last_target_x, last_target_y), self.d + self.l, 
                                            color=target_color, alpha=0.3)
            coverage_band_inner = plt.Circle((last_target_x, last_target_y), self.d - self.l, 
                                            color='white', alpha=1)
            plt.gca().add_patch(coverage_band_outer)
            plt.gca().add_patch(coverage_band_inner)

            # Previous position (initialize with the first data point)
            pre_target_x, pre_target_y = target_data[0]
            for target_x, target_y in target_data:
                plt.plot([pre_target_x, target_x], [pre_target_y, target_y], color=target_color)  # Draw line segment
                pre_target_x, pre_target_y = target_x, target_y  # Update previous position


        # Draw the charging station
        charging_station = plt.Circle((0, 0), self.r_c, color='blue', fill=True, alpha=0.3, label='Charging Station')
        plt.gca().add_patch(charging_station)
        # Draw the bolt sign inside the charging station
        plt.plot(0, 0, marker='$\u26A1$', color='yellow', markersize=20, linestyle='None', zorder=5)

        # Draw UAV trajectories
        for uav_idx, uav_data in enumerate(self.uav_trajectory_data):  # Assuming self.uav_trajectory_data is a list of lists, one per UAV
            # airplane_marker = [[] for _ in range(len(self.uav_trajectory_data))]
            # Previous position (initialize with the first data point)
            prev_x, prev_y, _, _ = uav_data[0]
            for x, y, battery, _ in uav_data:
                color = self.battery_to_color(battery)  # Define this function based on your needs
                plt.plot([prev_x, x], [prev_y, y], color=color)  # Draw line segment
                prev_x, prev_y = x, y  # Update previous position
                # Get the last position to calculate the direction
            last_x, last_y, _, last_theta = self.uav_trajectory_data[uav_idx][-1]
            second_last_x, second_last_y, _, _ = self.uav_trajectory_data[uav_idx][-2]

            # Calculate the direction vector from the second last to the last position
            direction_x = last_x - second_last_x
            direction_y = last_y - second_last_y

            # Normalize the direction vector
            length = np.hypot(direction_x, direction_y)
            direction_x /= length
            direction_y /= length

            # Offset the position slightly in the opposite direction to align the tail with the trajectory
            offset = 1.5  # Adjust this value as needed
            aligned_x = last_x + direction_x * offset
            aligned_y = last_y + direction_y * offset
            airplane_marker = plt.text(aligned_x, aligned_y, s='$\u2708$', color='black', 
                                                ha='center', va='center', fontsize=25,
                                                rotation=np.degrees(last_theta), zorder=20)

        limit = 60
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)

        plt.legend(loc='upper right')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'UAV Trajectory for Policy {policy}')
        plt.savefig(f'PLOT/UAV{self.m}_{self.targets[0].target_type}_Target{self.n}_{policy}_trajectory_{self.seed}.png')
        plt.show()

    # @property
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

    def compare_q_value(self):
        # dry_step with 1 env
        # 1) get reward for each action: 0 & 1
        state0, reward0, _, _, _ = self.dry_step(action=0)
        state1, reward1, _, _, _ = self.dry_step(action=1)

        # 2) get Value for each action
        S0, P0 = self.uav1target1_states.computeBarycentric(state0)
        value0 = 0
        for s0, p0 in zip(S0, P0):
            value0 += p0 * self.UAV1Target1_values00[s0]

        S1, P1 = self.uav1target1_states.computeBarycentric(state1)
        value1 = 0
        for s1, p1 in zip(S1, P1):
            value1 += p1 * self.UAV1Target1_values00[s1]

        self.Q0 = reward0+self.discount*value0
        self.Q1 = reward1+self.discount*value1
        self.max_Q = max(self.Q0, self.Q1)
        self.min_Q = min(self.Q0, self.Q1)
        self.argmax_Q = np.argmax([self.Q0, self.Q1])
