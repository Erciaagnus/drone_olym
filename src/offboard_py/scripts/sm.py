#!/usr/bin/env python3


import rospy
import smach
from smach import State, StateMachine
from mavros_msgs.msg import State as MavrosState
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL, CommandTOLRequest, SetModeRequest
from geometry_msgs.msg import PoseStamped, Pose, Point, PoseWithCovarianceStamped
import time

# Current State
current_state=None

waypoints=[]
waypoints=[
    (5,0,2),
    (5,5,2),
    (0,5,2),
    (0,0,2)
]
def state_cb(msg):
    global current_state
    current_state=msg

class TakeOff(State):
    def __init__(self):
        State.__init__(self, outcomes=['success', 'failed'])
        self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        self.pose = PoseStamped()
        self.pose.pose.position.x = 0
        self.pose.pose.position.y = 0
        self.pose.pose.position.z = 2
        self.rate = rospy.Rate(20)

    def execute(self, userdata):
        rospy.wait_for_service("mavros/cmd/arming")
        rospy.wait_for_service("mavros/set_mode")

        for _ in range(100):
            if rospy.is_shutdown():
                break
            self.local_pos_pub.publish(self.pose)
            self.rate.sleep()

        # OFFBOARD Mode(Custom Mode)
        try:
            set_mode = self.set_mode_client(custom_mode="OFFBOARD")
            if not set_mode.mode_sent:
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
            if not arming.success:
                rospy.logerr("Failed to arm drone")
                return 'failed'
        except rospy.ServiceException as e:
            rospy.logerr("Arming service call failed: %s" % e)
            return 'failed'

        # Publish Take Off Coordinate
        for _ in range(100):
            self.local_pos_pub.publish(self.pose)
            rospy.sleep(0.05)

        return 'success'

class GetPath(State):
    def __init__(self):
        State.__init__(self, outcomes=['success', 'failed'])
        self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.rate = rospy.Rate(20)  # 20Hz, It have to be faster than 20Hz

    def execute(self, userdata):
        rospy.loginfo("Getting path and moving to waypoints")
        for waypoint in waypoints:
            target_pose = PoseStamped()
            target_pose.header.stamp = rospy.Time.now()
            target_pose.header.frame_id = "map" # Certain Frame
            target_pose.pose.position.x = waypoint[0]
            target_pose.pose.position.y = waypoint[1]
            target_pose.pose.position.z = waypoint[2]

            for _ in range(100):  # The number of Publishing!!
# It have to be enough to move goal position
                self.local_pos_pub.publish(target_pose)
                try:
                    self.rate.sleep()
                except rospy.ROSInterruptException:
                    return 'failed'
            rospy.sleep(0.01)  # Waiting Time

        return 'success'

# Landing State
class Landing(State):
    def __init__(self):
        State.__init__(self, outcomes=['success', 'failed'])
        self.land_client = rospy.ServiceProxy("/mavros/cmd/land", CommandTOL)

    def execute(self, userdata):
        rospy.wait_for_service("/mavros/cmd/land")
        try:
            landing = self.land_client(altitude=0, latitude=0, longitude=0, min_pitch=0, yaw=0)
            if not landing.success:
                rospy.logerr("Failed to send landing command")
                return 'failed'
        except rospy.ServiceException as e:
            rospy.logerr("Landing service call failed: %s" % e)
            return 'failed'

        return 'success'

def main():
    rospy.init_node('offb_node_sm', anonymous=True)

    rospy.Subscriber("mavros/state", MavrosState, state_cb)

    # State Machine
    sm = smach.StateMachine(outcomes=['mission_completed', 'mission_failed'])
    with sm:
        smach.StateMachine.add('TAKEOFF', TakeOff(), transitions={'success':'GETPATH', 'failed':'mission_failed'})
        smach.StateMachine.add('GETPATH', GetPath(), transitions={'success':'LANDING', 'failed':'mission_failed'})
        smach.StateMachine.add('LANDING', Landing(), transitions={'success':'mission_completed', 'failed':'mission_failed'})

    # State Machine
    outcome = sm.execute()

if __name__ == '__main__':
    main()