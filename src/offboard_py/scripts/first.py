#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from std_msgs.msg import String
import time

# 현재 상태를 업데이트하는 콜백 함수
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

    rate = rospy.Rate(20)  # Hz

    # FCU 연결을 기다립니다.
    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()

    # Waypoints 정의
    waypoints = [
        (0, 0, 2),
        (10, 0, 2),
        (10, 10, 2),
        (0, 10, 2),
        (0, 0, 2)  # 초기 위치로 복귀
    ]

    # 이륙을 위한 초기 위치 설정
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = waypoints[0][0]
    pose.pose.position.y = waypoints[0][1]
    pose.pose.position.z = waypoints[0][2]

    # OFFBOARD 모드로 전환하고 기체를 이륙시키기 전 몇 번의 위치 정보를 미리 보냅니다.
    for _ in range(100):
        if rospy.is_shutdown():
            break
        local_pos_pub.publish(pose)
        rate.sleep()

    # OFFBOARD 모드로 전환 시도
    try:
        offb_set_mode = SetModeRequest(custom_mode="OFFBOARD")
        set_mode_client(base_mode=0, custom_mode=offb_set_mode.custom_mode)
    except rospy.ServiceException as e:
        rospy.logerr("Set mode service call failed: %s" % e)

    # 기체 활성화 시도
    try:
        arm_cmd = CommandBoolRequest(value=True)
        arming_client(arm_cmd.value)
    except rospy.ServiceException as e:
        rospy.logerr("Arming service call failed: %s" % e)

    # 각 Waypoint로 이동
    for wp in waypoints:
        rospy.loginfo("Moving to waypoint: {}".format(wp))
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = wp
        for _ in range(100):  # 각 waypoint에 대해 일정 시간 동안 명령을 보냅니다.
            if rospy.is_shutdown():
                break
            local_pos_pub.publish(pose)
            rate.sleep()

        time.sleep(5)  # 간단한 예제로, 실제 사용 시에는 현재 위치가 목표에 도달했는지 확인하는 로직이 필요합니다.

    rospy.loginfo("Mission completed. Returning...")

    # Landing Logic...

    rospy.spin()
