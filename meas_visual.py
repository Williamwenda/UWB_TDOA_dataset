'''
visualize the uwb measurements
'''
from mpl_toolkits.mplot3d import Axes3D
import rosbag, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error
from scipy import interpolate
import matplotlib.style as style
import math
# import packages for NN
import torch
import joblib

from tkinter.filedialog import askopenfilename

style.use('ggplot')


fileDir = os.path.dirname(__file__)
dataPath = os.path.join(fileDir, 'training_csv/meas_data')
dataPath = os.path.abspath(os.path.realpath(dataPath))

curr = os.chdir(dataPath)
csvFile = askopenfilename(initialdir = curr, title = "Select csv file")
data = pd.read_csv(csvFile, index_col = None, header=0)


traj_x = data['x'].values.reshape(-1,1)
traj_y = data['y'].values.reshape(-1,1)
traj_z = data['z'].values.reshape(-1,1)
vicon_gt = data['vicon_gt'].values.reshape(-1,1)
uwb_meas = data['uwb_tdoa_meas'].values.reshape(-1,1)
time = data['time'].values.reshape(-1,1)


# --------------------- plot trajectory -------------------------- #
fig_traj = plt.figure(facecolor = "white")
ax_t = fig_traj.add_subplot(111, projection = '3d')
# for idx in range(8):
#     ax_t.scatter(anchor_pos[idx,0], anchor_pos[idx,1], anchor_pos[idx,2], marker='o',color = c)
#     ax_t.text(anchor_pos[idx,0]+0.3, anchor_pos[idx,1]+0.3, anchor_pos[idx,2]+0.3, "An"+str(idx))
    
ax_t.plot(traj_x[:,0], traj_y[:,0], traj_z[:,0], color='steelblue',linewidth=1.9, alpha=0.9)

ax_t.set_xlim3d(np.amin(traj_x)-0.5, np.amax(traj_x)+0.5)  
ax_t.set_ylim3d(np.amin(traj_y)-0.5, np.amax(traj_y)+0.5)  
ax_t.set_zlim3d(np.amin(traj_z)-0.5, np.amax(traj_z)+0.5)  

ax_t.set_xlabel(r'X [m]')
ax_t.set_ylabel(r'Y [m]')
ax_t.set_zlabel(r'Z [m]')
plt.title(r"UWB Anchor Constellation", fontsize=13, fontweight=0, color='black', style='italic', y=1.02 )
ax_t.set_xlim([-5.0, 5.0])
ax_t.set_ylim([-5.0, 5.0])
ax_t.set_zlim([0.0, 5.0])


# --------------------- plot measurement -------------------------- #
fig = plt.figure(facecolor="white")
ax = fig.add_subplot(111)

ax.plot(time, vicon_gt, color='red',linewidth=1.5, label = "Interpolate vicon" )
ax.scatter(time, uwb_meas, color = "steelblue", s = 2.5, alpha = 0.9, label = "tdoa measurements")
ax.legend(loc='best')
ax.set_xlabel(r'Time [s]')
ax.set_ylabel(r'TDoA measurement [m]') 

# plt.ylim((-1.0, 1.0))
plt.title(r"UWB tdoa measurements", fontsize=13, fontweight=0, color='black')



plt.show()