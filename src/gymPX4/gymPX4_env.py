#!/usr/bin/env python3
import math
import numpy as np
import os
import sys
import time
import gym
from gym import spaces
import rospy
import subprocess
import asyncio

current_file_path = os.path.dirname(__file__)
gym_setting_path = os.path.join(current_file_path, '../gym_setting/mdp')
sys.path.append(os.path.abspath(gym_setting_path))

current_file_path = os.path.dirname(os.path.abspath(__file__))
gym_env_path = os.path.join(current_file_path, '../gym')
sys.path.append(gym_env_path)
# sys.path.append(current_file_path + os.path.sep + "gym")
from mdp import States, Actions, Surveillance_Actions, Rewards, StateTransitionProbability, Policy, MarkovDecisionProcess
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from mavros_msgs.msg import ActuatorControl, AttitudeTarget, Thrust, State
from mavros_msgs.srv import SetMode, SetModeRequest, CommandBool, CommandBoolRequest, CommandBoolResponse, StreamRate
from mavros_msgs.srv import StreamRateRequest
from sensor_msgs.msg import Imu
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Header
from std_msgs.msg import Float64
from std_srvs.srv import Empty, EmptyRequest
import mavros.setpoint
from scipy import sparse as sp
from scipy.stats.qmc import MultivariateNormalQMC
class gymPX4(gym.Env):
    def __init__(self, args):
        # Envrionment에 전부 저장되어 있는 건가요? / States Action etc
        self.env = gym.make('DKC_real_Unicycle') if args.real else gym.make('DKC_Unicycle')
        self.n_samples = 2**8
        self.sample_reward = False
        self.n_r = round(self.env.r_max/self.env.v*10)
        print('n_r: ', self.n_r)
        self.n_alpha = 360
        self.n_u = 2

        self.r_space = np.linspace(0, self.env.r_max, self.n_r, dtype = np.float64)
        self.alpha_space = np.linspace(-np.pi, np.pi-np.pi / self.n_alpha, self.n_alpha, dtype=np.float64)
        self.action_space = np.linspace(-1.0 / 4.5, 1.0 / 4.5, self.n_u, dtype=np.float64).reshape((-1, 1))

        # ## define gym spaces ##
        # self.min_action = 0.0
        # self.max_action = 1.0

        # self.min_position = 0.1
        # self.max_position = 25
        # self.max_speed = 3

        # self.low_state = np.array([self.min_position, -self.max_speed])
        # self.high_state = np.array([self.max_position, self.max_speed])

        # # Define Action Spaces
        # self.action_space = spaces.Box(low = self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)

        # # Define Observation Spaces
        # self.observation_space = spaces.Box(low = self.low_state, high = self.high_state, dtype = np.float32)

        # Define States
        self.current_state= State() # mavros_msgs.msg
        self.imu_data = Imu()
        self.act_controls = ActuatorControl()
        self.pose = PoseStamped()
        self.thrust_ctrl = Thrust()
        self.attitude_target = AttitudeTarget()
        self.local_velocity = TwistStamped()
        self.global_velocity = TwistStamped()

        ### Define ROS message ###
        self.offb_set_mode = SetModeRequest()
        self.offb_set_mode.custom_mode = "OFFBOARD"

        self.arm_cmd = CommandBoolRequest()
        self.arm_cmd.value = True

        self.disarm_cmd = CommandBoolRequest()
        self.disarm_cmd.value = False

        ## ROS Subscriber
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

        ## Initiate gym Environment
        self.init_env()

        ## ROS Services
        ## Wairing For Arming
        rospy.wait_for_service("mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

        rospy.wait_for_service('mavros/set_mode')
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        rospy.wait_for_service('mavros/set_strea_rate')
        set_stream_rate = rospy.ServiceProxy("mavros/set_stream_rate", StreamRate)

        set_stream_rate(StreamRateRequest.STREAM_POSITION, 50, 1)
        set_stream_rate(StreamRateRequest.STREAM_ALL, 50, 1)

        self.setpoint_msg = mavros.setpoint.PoseStamped(header=mavros.setpoint.Header(frame_id="att_pose", stamp=rospy.Time.now()),)

        self.offb_arm()
        ## Callback Function
        # local Velocity
        def lv_cb(self, data):
            self.local_velocity = data

        # local position
        def lp_cb(self,data):
            self.local_position = data

        # state
        def state_cb(self, data):
            self.current_state = data

        # imu
        def imu_cb(self, data):
            self.imu_data = data

        # act
        def act_cb(self, data):
            self.act_controls = data
        # global velocity
        def gv_cb(self, data):
            self.global_velocity = data

        # relative altitude
        def ra_cb(self, data):
            self.relative_atlitude = data
    ## Initialize Function
    def init_env(self):
        ## Initialize ROS node
        print('--connecting to mavros')
        rospy.init_node('gym_px4_mavros', anonymous=True)
        print('connected')

    def reset(self):
        self.success_steps = 0
        self.steps = 0
        reset_steps = 0

        self.desired = 2
        self.initial = 2

        print("Initial: ", self.initial, "Desired: ", self.desired)
        reset_pos = [0, 0, self.initial, 0]
        print('-- Resetting position')

        ### wait for quad to arrive to desired position
        while True:
            if self.current_state.armed == False:
                self.offb_arm()
            self.setpoint_msg.pose.position.x = reset_pos[0]
            self.setpoint_msg.pose.position.y = reset_pos[1]
            self.setpoint_msg.pose.position.z = reset_pos[2]

            self.local_pos_pub.publish(self.setpoint_msg)

            x= self.local_position.pose.position.x
            y= self.local_position.pose.position.y
            z= self.local_position.pose.position.z
            lin_pos = [x, y, z]

            vx= self.local_velocity.twist.linear.x
            vy= self.local_velocity.twist.linear.y
            vz= self.local_velocity.twist.linear.z
            lin_vel = [vx, vy, vz]

            if np.abs(np.linalg.norm(lin_pos[2]-reset_pos[2])) < 0.2 and np.abs(np.linag.norm(lin_vel))<0.2: # wait for UAV to reach desired position
                time.sleep(0.2)
                break
            print('Resetting position: ', reset_steps, '/100')
            sys.stdout.write("\033[F]")
            reset_steps +=1
            time.sleep(0.5)
        print('-------- Position Reset ---------')
        ob = np.array([ self.desired - lin_pos[2], lin_vel[2] ])
        self.last_pos = lin_pos

        return ob
    def step(self, action):
        start_time = time.time()
        rate = rospy.Rate(20)

        ## recieve updated position and velocity
        qx=self.local_position.pose.orientation.x
        qy=self.local_position.pose.orientation.y
        qz=self.local_position.pose.orientation.z
        qw=self.local_position.pose.orientation.w

        roll, pitch, yaw = quaternion_to_euler(qx,qy,qz,qw)
        ang_pos = [roll, pitch, yaw]

        while True:
            x=self.local_position.pose.position.x
            y=self.local_position.pose.position.y
            z=self.local_position.pose.position.z
            lin_pos = [x,y,z]
            if lin_pos[2] != self.last_pos[2]:
                self.last_pos = lin_pos
                break

        vx=self.local_velocity.twist.linear.x
        vy=self.local_velocity.twist.linear.y
        vz=self.local_velocity.twist.linear.z
        lin_vel = [vx,vy,vz]

        # ### send actuator control commands
        # self.act_controls.group_mix=0
        # self.act_controls.controls[0]=0
        # self.act_controls.controls[1]=0
        # self.act_controls.controls[2]=0
        # self.act_controls.controls[3]=action
        # self.acutator_control_pub.publish(self.act_controls)

        self.attitude_target.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | AttitudeTarget.IGNORE_PITCH_RATE | AttitudeTarget.IGNORE_YAW_RATE | AttitudeTarget.IGNORE_ATTITUDE
        self.attitude_target.thrust = action

        self.setpoint_raw_pub.publish(self.attitude_target)
        reward = -np.power( self.desired-lin_pos[2], 2)
        ob=np.array([ self.desired-lin_pos[2], lin_vel[2]] )

        done= False
        reset = 'No'
        # Rewards..
        if np.abs(lin_pos[0]) > 2 or np.abs(lin_pos[1]) > 2 or lin_pos[2]>3 or lin_pos[2]<1:
            done=True
            reset = 'out of region'
        self.steps = self.steps + 1
        self.last_pos = lin_pos
        step_prelen = time.time()-start_time
        if step_prelen < 0.03:
            time.sleep(0.03-step_prelen)

        step_len=time.time()-start_time

        rate.sleep()

        # print('state: ', ob , 'action: ', action , 'reward: ', reward, 'time: ', step_len)

        info = {"state" : ob, "action": action, "reward": reward, "step": self.steps, "step length": step_len, "reset reason": reset}
        return ob, reward, done, info

    def offb_arm(self):
        last_request = rospy.Time.now()
        flag1 = False
        flag2 = False

        prev_imu_data = Imu()
        prev_time = rospy.Time.now()
        count=0
        print('-- Enabling offboard mode and arming')
        rospy.wait_for_service('mavros/set_mode')
        self.set_mode_client(0, 'OFFBOARD')
        self.arming_client(True)

        rospy.loginfo('-- Ready to fly')

