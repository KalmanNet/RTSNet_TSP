import numpy as np
import torch
from datetime import datetime
from Linear_sysmdl import SystemModel

from KalmanFilter_test import KFTest
from RTS_Smoother_test import S_Test

from RTSNet_nn import RTSNetNN
from Pipeline_ERTS import Pipeline_ERTS as Pipeline

from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from model import gt_data, F_kitti, H_kitti, delta_t, lambda_q, lambda_r, Q, R, m, n, m2_0

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")
   
print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results = 'RTSNet/'

##################
### Load KITTI ###
##################

####################
### Design Model ###
####################
r = lambda_r[0]
q = lambda_q[0]
print("Observation noise 1/r2 [dB]: ", 10 * torch.log10(1/(r**2)))
print("Process noise 1/q2 [dB]: ", 10 * torch.log10(1/(q**2)))

### T and T_test for alternative 1
# T = round(gt_data[0].size()[-1] * 0.8)
# T_test = gt_data[0].size()[-1] - T

### T and T_test for alternative 2
T = gt_data[0].size()[-1]
T_test = T

m1_0 = gt_data[0][:,0]

# True model 
sys_model = SystemModel(F_kitti, q, Q, H_kitti, r, R, T, T_test,m,n)
sys_model.InitSequence(m1_0, m2_0)

########################
### Generate dataset ###
########################
# Training dataset
train_target = []
cv_target = []
test_target = []
train_input = []
cv_input = []
test_input = []

### Alternative 1
# for sequence in gt_data:
#     T1 = round(sequence.size()[-1] * 0.7) # split for training set
#     T2 = round(sequence.size()[-1] * 0.8) # split for cv set
#     train_target.append(sequence[:,0:T1])
#     cv_target.append(sequence[:,T1:T2])
#     test_target.append(sequence[:,T2:])

#     noise_free_obs = torch.matmul(H_kitti, sequence)
#     obs = noise_free_obs + torch.randn_like(noise_free_obs) * r # Observations; additive Gaussian Noise
#     train_input.append(obs[:,0:T1])
#     cv_input.append(obs[:,T1:T2])
#     test_input.append(obs[:,T2:])

### Alternative 2
NumTrain = 56
NumCV = 8
NumTest = 16
train_target = gt_data[0:NumTrain]
cv_target = gt_data[NumTrain:NumTrain+NumCV]
test_target = gt_data[NumTrain+NumCV:NumTrain+NumCV+NumTest]

obs = []
for sequence in gt_data:
    noise_free_obs = torch.matmul(H_kitti, sequence)
    obs.append(noise_free_obs + torch.randn_like(noise_free_obs) * r) # Observations; additive Gaussian Noise
train_input = obs[0:NumTrain]
cv_input = obs[NumTrain:NumTrain+NumCV]
test_input = obs[NumTrain+NumCV:NumTrain+NumCV+NumTest]


##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter True")
[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, test_input, test_target,kitti=True)

##############################
### Evaluate RTS Smoother ###
##############################
print("Evaluate RTS Smoother True")
[MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg] = S_Test(sys_model, test_input, test_target,kitti=True)

#######################
### RTSNet Pipeline ###
#######################
# RTSNet with full info
## Build Neural Network
print("RTSNet with full model info")
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model)
## Train Neural Network
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model)
RTSNet_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=5, learningRate=1E-3, weightDecay=1E-4)
# RTSNet_Pipeline.model = torch.load('RTSNet/new_architecture/linear/best-model_linear2x2rq020T100.pt',map_location=dev)
[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results,kitti=True)
## Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,kitti=True)
RTSNet_Pipeline.save()

# RTSNet with mismatched model
## Build Neural Network
# print("RTSNet with observation model mismatch")
# RTSNet_model = RTSNetNN()
# RTSNet_model.NNBuild(sys_model_partialh)
# ## Train Neural Network
# RTSNet_Pipeline = Pipeline(strTime, "RTSNetPartialH", "RTSNetPartialH")
# RTSNet_Pipeline.setssModel(sys_model_partialh)
# RTSNet_Pipeline.setModel(RTSNet_model)
# RTSNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=30, learningRate=1E-3, weightDecay=1E-5)
# # RTSNet_Pipeline.model = torch.load('ERTSNet/best-model_DTfull_rq3050_T2000.pt',map_location=dev)
# [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_partialh, cv_input, cv_target, train_input, train_target, path_results)
# ## Test Neural Network
# [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model_partialh, test_input, test_target, path_results)
# RTSNet_Pipeline.save()


# DatafolderName = 'Data' + '/'
# DataResultName = '10x10_Ttest1000' 
# torch.save({
#             'MSE_KF_linear_arr': MSE_KF_linear_arr,
#             'MSE_KF_dB_avg': MSE_KF_dB_avg,
#             'MSE_RTS_linear_arr': MSE_RTS_linear_arr,
#             'MSE_RTS_dB_avg': MSE_RTS_dB_avg,
#             }, DatafolderName+DataResultName)

# print("Plot")
# RTSNet_Pipeline.PlotTrain_RTS(MSE_KF_linear_arr, MSE_KF_dB_avg, MSE_RTS_linear_arr, MSE_RTS_dB_avg)




