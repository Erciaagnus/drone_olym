#! /usr/bin/env python
# 가상의 센서값을 출력 : 목표 위치(pWaypoints를 가상의 센서값으로 대체)
# 컨트롤러가 이 값을 받아서 PID 계산. 
# pesudo_sensor_node_py 노드

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State #상태를 전달받아 잘 작동할 때 속도를 전달하기 위함. 

# 주요변수
current_state = State()
pWaypoints = [
    (0, 0, 2),
    (10, 0, 2),
    (10, 10, 2),
    (0, 10, 2),
    (0, 0, 2)
]

# 함수정의
def state_cb(msg):
    global current_state
    current_state = msg



if __name__ == "__main__":
    rospy.init_node("pesudo_sensor_node_py", anonymous=True)

    # 서브스크라이버
    state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)

    # 퍼블리셔
    pSensor_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)

    # 서비스

    # 타이밍 설정
    rate = rospy.Rate(20)

    # OFFBOARD 모드를 기다림
    while(not rospy.is_shutdown() and current_state.connected):
        rate.sleep()


    rospy.spin()