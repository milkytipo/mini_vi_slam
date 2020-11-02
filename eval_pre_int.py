import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gt_data_1 = pd.read_csv("../build/gt_dr_1.csv")
gt_data_2 = pd.read_csv("../build/gt_dr_2.csv")
gt_data_3 = pd.read_csv("../build/gt_dr_3.csv")
gt_data_4 = pd.read_csv("../build/gt_dr_4.csv")
gt_data_6 = pd.read_csv("../build/gt_dr_6.csv")
gt_data_6_new = pd.read_csv("../build/gt_dr_6_new.csv")
gt_data_8 = pd.read_csv("../build/gt_dr_8.csv")
gt_data_10 = pd.read_csv("../build/gt_dr_10.csv")

traj_data_1 = pd.read_csv("../build/traj_dr_1.csv")
traj_data_2 = pd.read_csv("../build/traj_dr_2.csv")
traj_data_3 = pd.read_csv("../build/traj_dr_3.csv")
traj_data_4 = pd.read_csv("../build/traj_dr_4.csv")
traj_data_6 = pd.read_csv("../build/traj_dr_6.csv")
traj_data_6_new = pd.read_csv("../build/traj_dr_6_new.csv")
traj_data_8 = pd.read_csv("../build/traj_dr_8.csv")
traj_data_10 = pd.read_csv("../build/traj_dr_10.csv")




fig = plt.figure(1)
ax = fig.gca(projection='3d')


ax.plot(gt_data_2['p_x'], gt_data_2['p_y'], gt_data_2['p_z'], label='ground truth')

# ax.plot(traj_data_1['p_x'], traj_data_1['p_y'], traj_data_1['p_z'], label='downsample rate = 4')
# ax.plot(traj_data_2['p_x'], traj_data_2['p_y'], traj_data_2['p_z'], label='downsample rate = 2')
ax.plot(traj_data_3['p_x'], traj_data_3['p_y'], traj_data_3['p_z'], label='downsample rate = 3')
ax.plot(traj_data_4['p_x'], traj_data_4['p_y'], traj_data_4['p_z'], label='downsample rate = 4')
ax.plot(traj_data_6['p_x'], traj_data_6['p_y'], traj_data_6['p_z'], label='downsample rate = 6')
ax.plot(traj_data_6_new['p_x'], traj_data_6_new['p_y'], traj_data_6_new['p_z'], label='downsample rate = 6 new')
ax.plot(traj_data_8['p_x'], traj_data_8['p_y'], traj_data_8['p_z'], label='downsample rate = 8')
ax.plot(traj_data_10['p_x'], traj_data_10['p_y'], traj_data_10['p_z'], label='downsample rate = 10')


# trajectory only
# ax.set_xlim(-2.8,1.8)
# ax.set_ylim( 4.2,8.8)
# ax.set_zlim(-1.8,2.8)

boarder = 1

# ax.set_xlim(-30-boarder, 30+boarder)
# ax.set_ylim(-40-boarder, 20+boarder)
# ax.set_zlim(-20-boarder, 20+boarder)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')

ax.legend()

plt.show()


fig = plt.figure(2)

position_error_1 = np.zeros_like(gt_data_1['p_x']);
position_error_2 = np.zeros_like(gt_data_2['p_x']);
position_error_3 = np.zeros_like(gt_data_3['p_x']);
position_error_4 = np.zeros_like(gt_data_4['p_x']);
position_error_6 = np.zeros_like(gt_data_6['p_x']);
position_error_6_new = np.zeros_like(gt_data_6_new['p_x']);
position_error_8 = np.zeros_like(gt_data_8['p_x']);
position_error_10 = np.zeros_like(gt_data_10['p_x']);


for i in range(len(gt_data_1['p_x'])):
	position_error_1[i] = math.sqrt( (gt_data_1['p_x'][i]-traj_data_1['p_x'][i])**2 + (gt_data_1['p_y'][i]-traj_data_1['p_y'][i])**2 + (gt_data_1['p_z'][i]-traj_data_1['p_z'][i])**2)

for i in range(len(gt_data_2['p_x'])):
	position_error_2[i] = math.sqrt( (gt_data_2['p_x'][i]-traj_data_2['p_x'][i])**2 + (gt_data_2['p_y'][i]-traj_data_2['p_y'][i])**2 + (gt_data_2['p_z'][i]-traj_data_2['p_z'][i])**2)

for i in range(len(gt_data_3['p_x'])):
	position_error_3[i] = math.sqrt( (gt_data_3['p_x'][i]-traj_data_3['p_x'][i])**2 + (gt_data_3['p_y'][i]-traj_data_3['p_y'][i])**2 + (gt_data_3['p_z'][i]-traj_data_3['p_z'][i])**2)

for i in range(len(gt_data_4['p_x'])):
	position_error_4[i] = math.sqrt( (gt_data_4['p_x'][i]-traj_data_4['p_x'][i])**2 + (gt_data_4['p_y'][i]-traj_data_4['p_y'][i])**2 + (gt_data_4['p_z'][i]-traj_data_4['p_z'][i])**2)

for i in range(len(gt_data_6['p_x'])):
	position_error_6[i] = math.sqrt( (gt_data_6['p_x'][i]-traj_data_6['p_x'][i])**2 + (gt_data_6['p_y'][i]-traj_data_6['p_y'][i])**2 + (gt_data_6['p_z'][i]-traj_data_6['p_z'][i])**2)

for i in range(len(gt_data_6_new['p_x'])):
	position_error_6_new[i] = math.sqrt( (gt_data_6_new['p_x'][i]-traj_data_6_new['p_x'][i])**2 + (gt_data_6_new['p_y'][i]-traj_data_6_new['p_y'][i])**2 + (gt_data_6_new['p_z'][i]-traj_data_6_new['p_z'][i])**2)



for i in range(len(gt_data_8['p_x'])):
	position_error_8[i] = math.sqrt( (gt_data_8['p_x'][i]-traj_data_8['p_x'][i])**2 + (gt_data_8['p_y'][i]-traj_data_8['p_y'][i])**2 + (gt_data_8['p_z'][i]-traj_data_8['p_z'][i])**2)

for i in range(len(gt_data_10['p_x'])):
	position_error_10[i] = math.sqrt( (gt_data_10['p_x'][i]-traj_data_10['p_x'][i])**2 + (gt_data_10['p_y'][i]-traj_data_10['p_y'][i])**2 + (gt_data_10['p_z'][i]-traj_data_10['p_z'][i])**2)


# plt.plot(traj_data_1['timestamp'], position_error_1, label='downsample rate = 1')
# plt.plot(traj_data_2['timestamp'], position_error_2, label='downsample rate = 2')
plt.plot(traj_data_3['timestamp'], position_error_3, label='downsample rate = 3')
plt.plot(traj_data_4['timestamp'], position_error_4, label='downsample rate = 4')
plt.plot(traj_data_6['timestamp'], position_error_6, label='downsample rate = 6')
plt.plot(traj_data_6_new['timestamp'], position_error_6_new, label='downsample rate = 6 new')
plt.plot(traj_data_8['timestamp'], position_error_8, label='downsample rate = 8')
plt.plot(traj_data_10['timestamp'], position_error_10, label='downsample rate = 10')

plt.legend()

plt.xlabel('time [s]')
plt.ylabel('error [m]')

# plt.xlim(649, 659)
# plt.ylim(0, 5)

plt.show()