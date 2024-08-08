#!/usr/bin/env python3

import rospy
import smach
from smach import State, StateMachine
from mavros_msgs.msg import State as MavrosState
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL