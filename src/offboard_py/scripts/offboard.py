#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from mavros_msgs.srv import CommandTOL, CommandTOLRequest
import time

def state_cb(msg):
    global current_state
    current_state = msg

if __name__ == "__main__":
    rospy.init_node("offb_node_py", anonymous=True)
    current_state = State()

    rospy.Subscriber("mavros/state", State, state_cb)
    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)

    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
    land_client = rospy.ServiceProxy("/mavros/cmd/land", CommandTOL)  # 착륙 서비스 클라이언트

    rate = rospy.Rate(20)  # Hz
    pose=PoseStamped()
    pose.header.stamp=rospy.Time.now()
    
    for _ in range(100):
        local_pos_pub.publish(pose)
        rate.sleep()

    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()

    waypoints = [
        (0, 0, 2),
        (10, 0, 2),
        (10, 10, 2),
        (0, 10, 2),
        (0, 0, 2)
    ]

    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()

    for _ in range(100):
        if rospy.is_shutdown():
            break
        local_pos_pub.publish(pose)
        rate.sleep()

    try:
        offb_set_mode = SetModeRequest(custom_mode="OFFBOARD")
        print("Successfully connected to OFFBOARD Mode")
        set_mode_client(base_mode=0, custom_mode=offb_set_mode.custom_mode)
    except rospy.ServiceException as e:
        rospy.logerr("Set mode service call failed: %s" % e)

    try:
        arm_cmd = CommandBoolRequest(value=True)
        arming_client(arm_cmd.value)
        print("The drone is armed")
    except rospy.ServiceException as e:
        rospy.logerr("Arming service call failed: %s" % e)

    for wp in waypoints:
        rospy.loginfo("Moving to waypoint: {}".format(wp))
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = wp
        for _ in range(100):
            if rospy.is_shutdown():
                 break
            local_pos_pub.publish(pose)
            rate.sleep()

    rospy.loginfo("Mission completed. Initiating landing sequence...")

    # Landing Command
    try:
        land_cmd = CommandTOLRequest(altitude=0, latitude=0, longitude=0, min_pitch=0, yaw=0)
        response = land_client(land_cmd)
        if response.success:
            rospy.loginfo("Landing command sent successfully")
    except rospy.ServiceException as e:
        rospy.logerr("Landing service call failed: %s" % e)

    time.sleep(5)  # Waiting Time
    rospy.loginfo("Mission and landing completed. Exiting script...")
