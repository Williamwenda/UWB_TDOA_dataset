'''
visualize the UWB TDOA bias and dnn regression
'''
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

# help function
def denormalize(scl, norm_data):
    # @param: scl: the saved scaler   norm_data: 
    norm_data = norm_data.reshape(-1,1)
    new = scl.inverse_transform(norm_data)
    return new

# PARAM
WithAn = True;     ShowNN = True

fileDir = os.path.dirname(__file__)
# network with anchor information
if WithAn:
    netPath = os.path.join(fileDir, 'network/dnn_01')
## network without anchor information
else:
    netPath = os.path.join(fileDir, 'network/dnn_02')
    
netPath = os.path.abspath(os.path.realpath(netPath))

curr = os.chdir(netPath)
device = torch.device('cpu')
DNN = torch.load('BiasNet_tdoa.pkl', map_location=device)    # load the model in cpu
scaler_tdoa_x = joblib.load("scaler_x")       # scaler for input x feature
scaler_tdoa_y = joblib.load("scaler_y")       # scaler for output (UWB bias)
# ----------------- load data ---------------------- #

dataPath = os.path.join(fileDir, 'testing_csv/bias_data')
# dataPath = os.path.join(fileDir, 'training_csv/bias_data')
dataPath = os.path.abspath(os.path.realpath(dataPath))

curr = os.chdir(dataPath)
csvFile = askopenfilename(initialdir = curr, title = "Select csv file")
data = pd.read_csv(csvFile, index_col = None, header=0)


# change w.r.t different dnn
if WithAn:
    feature_num = 14
    x_feature =  data.drop(['tdoa_bias', 'time'],axis=1).values
else:
    feature_num = 10
    x_feature =  data.drop(['tdoa_bias', 'azm1_an', 'ele1_an', 'azm2_an','ele2_an', 'time'],axis=1).values

y_error = data['tdoa_bias'].values.reshape(-1,1)  
t_uwb = data['time'].values.reshape(-1,1)

x_feature = scaler_tdoa_x.transform(x_feature)
x_input = torch.from_numpy(x_feature)

# prediction
bias = DNN(x_input)

uwbBias = denormalize(scaler_tdoa_y, bias.data.numpy())      
# y_test = denormalize(scaler_tdoa_y, y_error.data.numpy())
y_test = y_error

fig = plt.figure(facecolor="white")
ax = plt.subplot()
ax.scatter(t_uwb, y_test, color = "steelblue", s = 2.5, alpha = 0.9, label = "Original bias")
if ShowNN:
    ax.plot(t_uwb, uwbBias,  linewidth=2.5,color="tomato",label = "Predicted error")
ax.legend(loc='best')
ax.set_xlabel(r'time [s]',fontsize=18)
ax.set_ylabel(r'Error [m]',fontsize=18) 
ax.set_ylim([-1.0, 1.0]) 
plt.title("UWB TDOA measurement bias", fontsize=18, fontweight=2, color='black')

plt.show()



