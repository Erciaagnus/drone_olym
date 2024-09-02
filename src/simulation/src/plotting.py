#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pickle
import os
import h5py
current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_dir, '../', 'traj', 'uav_trajectory_m_3_seed_None.h5') # if pickle. pkl


# HDF5 파일을 엽니다.
with h5py.File(relative_path, 'r') as file:
    uav0_coords = file['uav_0'][:]
    uav1_coords = file['uav_1'][:]
    uav2_coords = file['uav_2'][:]
uav0_x, uav0_y = uav0_coords[:, 0], uav0_coords[:, 1]
uav1_x, uav1_y = uav1_coords[:, 0], uav1_coords[:, 1]
uav2_x, uav2_y = uav2_coords[:, 0], uav2_coords[:, 1]

# PICKLE 파일 형식
# # 파일을 엽니다.
# with open(relative_path, 'rb') as file:
#     uav_trajectory_data = pickle.load(file)

# # 각 UAV의 좌표 추출
# uav0_coords = [(data[0], data[1]) for data in uav_trajectory_data[0]]
# uav1_coords = [(data[0], data[1]) for data in uav_trajectory_data[1]]
# uav2_coords = [(data[0], data[1]) for data in uav_trajectory_data[2]]

# 좌표 분리 (X, Y)
# uav0_x, uav0_y = zip(*uav0_coords)
# uav1_x, uav1_y = zip(*uav1_coords)
# uav2_x, uav2_y = zip(*uav2_coords)

target_position = [[800, 1430], [-1180, 700], [1200, 1300], [-1100, -1200], [800, -1120]]
target_x, target_y = zip(*target_position)

# 그래프 그리기
plt.figure(figsize=(10, 8))

# UAV 경로를 먼저 그리기
plt.plot(uav0_x, uav0_y, label='UAV 0', marker='o', linewidth = 0.5, markersize=2, zorder=1)
plt.plot(uav1_x, uav1_y, label='UAV 1', marker='x', linewidth = 0.5, markersize=2, zorder=1)
plt.plot(uav2_x, uav2_y, label='UAV 2', marker='s', linewidth = 0.5, markersize=2, zorder=1)


# 타겟 마커를 나중에 그리기 (가장 위에 그려짐)
plt.scatter(target_x, target_y, color='red', marker='^', s=100, label='Targets', zorder=2)

plt.title('UAV Trajectories')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()