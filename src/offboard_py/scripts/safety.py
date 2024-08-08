#!/usr/bin/env python3
#
# AUTHOR: JeongHyeok Lim
# Maintainer: JeongHyeok Lim
# E-mail: henricus0973@korea.ac.kr
# Date: 2024-03-11
# COPYRIGHT@2024 It is not permitted to use without AUTHOR's Permission
#

import rospy
import smach
from smach import State, StateMachine
from mavros_msgs.msg import State as MavrosState
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from geometry_msgs.msg import PoseStamped
import time
from std_msgs.msg import String
from pynput.keyboard import Key, Listener

# Current State
current_state = None
# Offboard Mode Flag
offboard_mode_enabled = True

# Define the Waypoints
waypoints = [
    (5, 0, 2),
    (5, 5, 2),
    (0, 5, 2),
    (0, 0, 2)
]

# Define Callback Function
def state_cb(msg):
    global current_state
    current_state = msg

# Keyboard Event Callback
def on_press(key):
    global offboard_mode_enabled
    if key == Key.space:
        print("Space key pressed, exiting offboard mode...")
        offboard_mode_enabled = False

# Define The Takeoff Class. First state of the flight
class TakeOff(State):
    def __init__(self):
        State.__init__(self, outcomes=['success', 'failed'])
        self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        self.pose = PoseStamped()
        self.rate = rospy.Rate(20)

    def execute(self, userdata):
        global offboard_mode_enabled
        rospy.wait_for_service("mavros/cmd/arming")
        rospy.wait_for_service("mavros/set_mode")

        for _ in range(100):
            if rospy.is_shutdown() or not offboard_mode_enabled:
                return 'failed'
            self.local_pos_pub.publish(self.pose)
            self.rate.sleep()

        # OFFBOARD Mode (Custom Mode)
        try:
            set_mode = self.set_mode_client(custom_mode="OFFBOARD")
            if not set_mode.mode_sent or not offboard_mode_enabled:
                rospy.logerr("Failed to set OFFBOARD mode")
                return 'failed'
        except rospy.ServiceException as e:
            rospy.logerr("Set mode service call failed: %s" % e)
            return 'failed'

        # Landing Position
        self.pose.header.stamp = rospy.Time.now()

        # Arming the Drone
        try:
            arming = self.arming_client(True)
            if not arming.success or not offboard_mode_enabled:
                rospy.logerr("Failed to arm drone")
                return 'failed'
        except rospy.ServiceException as e:
            rospy.logerr("Arming service call failed: %s" % e)
            return 'failed'

        self.pose.pose.position.x = 0
        self.pose.pose.position.y = 0
        self.pose.pose.position.z = 2

        # Publish Take Off Coordinate
        for _ in range(100):
            if not offboard_mode_enabled:
                return 'failed'
            self.local_pos_pub.publish(self.pose)
            rospy.sleep(0.1)

        return 'success'

class GetPath(State):
    def __init__(self):
        State.__init__(self, outcomes=['success', 'failed'])
        self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.rate = rospy.Rate(20)

    def execute(self, userdata):
        global offboard_mode_enabled
        rospy.loginfo("Getting path and moving to waypoints")
        for waypoint in waypoints:
            if not offboard_mode_enabled:
                return 'failed'
            target_pose = PoseStamped()
            target_pose.header.stamp = rospy.Time.now()
            target_pose.header.frame_id = "map"
            target_pose.pose.position.x = waypoint[0]
            target_pose.pose.position.y = waypoint[1]
            target_pose.pose.position.z = waypoint[2]

            for _ in range(100):
                if not offboard_mode_enabled:
                    return 'failed'
                self.local_pos_pub.publish(target_pose)
                try:
                    self.rate.sleep()
                except rospy.ROSInterruptException:
                    return 'failed'
            rospy.sleep(0.01)

        return 'success'

class Landing(State):
    def __init__(self):
        State.__init__(self, outcomes=['success', 'failed'])
        self.land_client = rospy.ServiceProxy("/mavros/cmd/land", CommandTOL)

    def execute(self, userdata):
        global offboard_mode_enabled
        if not offboard_mode_enabled:
            return 'failed'

        rospy.wait_for_service("/mavros/cmd/land")
        try:
            self.land_client(altitude=0, latitude=0, longitude=0, min_pitch=0, yaw=0)
            rospy.sleep(4)
            return 'success'
        except rospy.ServiceException as e:
            rospy.logerr("Landing service call failed: %s" % e)
            return 'failed'

def main():
    rospy.init_node('offb_node_sm', anonymous=True)
    rospy.Subscriber("mavros/state", MavrosState, state_cb)

    # Start listening to keyboard events
    listener = Listener(on_press=on_press)
    listener.start()

    # State Machine
    sm = smach.StateMachine(outcomes=['mission_completed', 'mission_failed'])
    with sm:
        smach.StateMachine.add('TAKEOFF', TakeOff(), transitions={'success':'GETPATH', 'failed':'mission_failed'})
        smach.StateMachine.add('GETPATH', GetPath(), transitions={'success':'LANDING', 'failed':'mission_failed'})
        smach.StateMachine.add('LANDING', Landing(), transitions={'success':'mission_completed', 'failed':'mission_failed'})

    outcome = sm.execute()

if __name__ == '__main__':
    main()
