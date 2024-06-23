#! /usr/bin/env python
# 센서의 목표위치와 현재위치를 받아서 PID 산출 -> 속도 제어 명령
# 노드이름 : xyz_controller_node_py

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State

# 주요변수
current_state = State()

# 함수정의
def state_cb(msg):
    global current_state
    current_state = msg

if __name__ =="__main__":
    rospy.init_node("xyz_controller_node_py", anonymous=True)

    # 서브스크라이버
    state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)

    # 퍼블리셔

    # 서비스

    # 타이밍 설정
    rate = rospy.Rate(20)

    # 이륙준비 기다림

    # 

    rospy.spin()