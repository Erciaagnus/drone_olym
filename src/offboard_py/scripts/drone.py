#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#https://github.com/CankayaUniversity/ceng-407-408-2022-2023-Autonomous-VTOL-Design/blob/main/simulation_Uav_Control/track_and_follow.py
import rospy
from mavros_msgs.msg import State, GlobalPositionTarget
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from geographic_msgs.msg import GeoPoseStamped
from std_msgs.msg import String
import time

current_state=State()
offb_set_mode=SetMode
arm_cmd=CommandBool
takeoff_cmd=CommandBool

# Define the GPS waypoints [latitude, longitude, altitude]
waypoints = [
    [47.397748, 8.545596, 10],
    [47.398039, 8.545461, 10],
    [47.398036, 8.545228, 10],
    [47.397802, 8.545259, 10]
]


def state_cb(msg):
    global current_state
    current_state=msg

def set_waypoint(latitude, longitude, altitude):
    waypoint=GlobalPositionTarget()
    waypoint.latitude=latitude
    waypoint.longitude=longitude
    waypoint.altitude=altitude
    return waypoint

def move_drone(waypoints):
    rospy.loginfo("Moving to waypoints...")
    for waypoint in waypoints:
        target=set_waypoint(waypoint[0], waypoint[1], waypoint[2])
        for i in range(100):
            local_pos_pub.publish(target)
            rate.sleep()
        rospy.loginfo("Reached waypoint: %s" %waypoint)

if __name__=="__main__":
    rospy.init_node('drone_auto_mission', anonymous=True)
    rospy.Subscriber("/mavros/state", State, state_cb)
    
    local_pos_pub = rospy.Publisher('/mavros/setpoint_raw/global', GlobalPositionTarget, queue_size=10)
    
    # Services
    rospy.wait_for_service('/mavros/cmd/arming')
    rospy.wait_for_service('/mavros/set_mode')
    arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
    set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

    rate = rospy.Rate(20.0) # MUST be more then 2Hz

    # Wait for FCU connection
    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()

    # Arming
    arming_client(True)

    # Switch to AUTO mode
    set_mode_client(base_mode=0, custom_mode="AUTO.MISSION")

    # Moving to waypoints
    move_drone(waypoints)

    # Return to Launch
    rospy.loginfo("Returning to Launch")
    set_mode_client(base_mode=0, custom_mode="AUTO.RTL")

    rate.sleep()