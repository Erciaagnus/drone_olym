#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pickle
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_dir, '../', 'traj', 'uav_trajectory_m_3_seed_None.pkl')

# 파일을 엽니다.
with open(relative_path, 'rb') as file:
    uav_trajectory_data = pickle.load(file)

# 각 UAV의 좌표 추출
uav0_coords = [(data[0], data[1]) for data in uav_trajectory_data[0]]
uav1_coords = [(data[0], data[1]) for data in uav_trajectory_data[1]]
uav2_coords = [(data[0], data[1]) for data in uav_trajectory_data[2]]

# 좌표 분리 (X, Y)
uav0_x, uav0_y = zip(*uav0_coords)
uav1_x, uav1_y = zip(*uav1_coords)
uav2_x, uav2_y = zip(*uav2_coords)

# 그래프 그리기
plt.figure(figsize=(10, 8))
plt.plot(uav0_x, uav0_y, label='UAV 0', marker='o')
plt.plot(uav1_x, uav1_y, label='UAV 1', marker='x')
plt.plot(uav2_x, uav2_y, label='UAV 2', marker='s')

plt.title('UAV Trajectories')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()