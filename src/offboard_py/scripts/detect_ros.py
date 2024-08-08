#!/usr/bin/env python3
#
# AUTHOR: JeongHeyok Lim
# Maintainer: JeongHyeok Lim
# E-mail: henricus0973@korea.ac.kr
# Date: 2024-03-19
# COPYRIGHT@2024 It is not permitted to use without AUTHOR's Permission
#
import rospy
import smach
from smach import State, StateMachine
from mavros_msgs.msg import State as MavrosState
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL, CommandTOLRequest, SetModeRequest
from geometry_msgs.msg import PoseStamped, Pose, Point, PoseWithCovarianceStamped
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
# Current State
current_state=None

# Define the Waypoints
waypoints=[]
waypoints=[
    (5,0,2),
    (5,5,2),
    (0,5,2),
    (0,0,2)
]

bridge=CvBridge()

# Define Callback Function
def state_cb(msg):
    global current_state
    current_state=msg

# Define Callback Function for Camera Image
def image_callback(msg):
    try:
        cv2_img=bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("Camera Image", cv2_img)
        print("Image Output is Success!")
        cv2.waitKey(3) # Wait for a key press for 3 millisceconds
    except CvBridgeError as e:
        print(e)


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
        self.pose.pose.position.x = 0
        self.pose.pose.position.y = 0
        self.pose.pose.position.z = 2

        # Publish Take Off Coordinate
        for _ in range(100):
            self.local_pos_pub.publish(self.pose)
            rospy.sleep(0.1) # If you set 0.01 or 0.05 It is too short

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
            # Define the next position.
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

#TODO (1) RECEIVE the CAMERA Message and Collison Message

# class AvoidObstacle(State):
#     def __init__(self):
#         State.__init__(self, outcomes=['success', 'failed', 'avoided'])
#         self.direction_sub = rospy.Subscriber('avoidance_direction', String, self.direction_cb)
#         self.avoid_direction = None
#         self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)

#     def direction_cb(self, msg):
#         self.avoid_direction = msg.data

#     def execute(self, userdata):
#         if self.avoid_direction is None:
#             rospy.loginfo("No direction received, continuing on path")
#             return 'failed'

#         rospy.loginfo(f"Received direction to avoid: {self.avoid_direction}")
#         # 여기에서 회피 로직을 구현합니다.
#         # 예시: self.avoid_direction 값에 따라 새로운 위치 목표를 설정하고 이동합니다.
        
#         # 장애물 회피 후의 새로운 위치 목표 설정
#         # 이 부분은 프로젝트의 요구사항에 따라 조정될 수 있습니다.
#         new_target_pose = PoseStamped()
#         new_target_pose.header.stamp = rospy.Time.now()
#         new_target_pose.header.frame_id = "map"
#         new_target_pose.pose.position.x = # 새로운 X 좌표
#         new_target_pose.pose.position.y = # 새로운 Y 좌표
#         new_target_pose.pose.position.z = 2  # 고정 고도

#         # 새로운 위치 목표로 이동
#         self.local_pos_pub.publish(new_target_pose)
#         rospy.sleep(2)  # 잠시 대기, 실제 환경에서는 이동 완료를 확인하는 로직이 필요할 수 있습니다.

#         return 'avoided'

# Landing State
class Landing(State):
    def __init__(self):
        State.__init__(self, outcomes=['success', 'failed'])
        self.land_client = rospy.ServiceProxy("/mavros/cmd/land", CommandTOL)

    def execute(self, userdata):
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
    rospy.Subscriber("/iris/camera/link/camera/image", Image, image_callback)
    # State Machine
    sm = smach.StateMachine(outcomes=['mission_completed', 'mission_failed'])
    with sm:
        # Transition!!
        smach.StateMachine.add('TAKEOFF', TakeOff(), transitions={'success':'GETPATH', 'failed':'mission_failed'})
        smach.StateMachine.add('GETPATH', GetPath(), transitions={'success':'LANDING', 'failed':'mission_failed'})
        smach.StateMachine.add('LANDING', Landing(), transitions={'success':'mission_completed', 'failed':'mission_failed'})

    cv2.destroyAllWindows()
    # State Machine
    outcome = sm.execute()

if __name__ == '__main__':
    main()