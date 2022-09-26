import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from Linear_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, DecimateData
from Pipeline_ERTS import Pipeline_ERTS as Pipeline

from datetime import datetime
from RTSNet_nn import RTSNetNN
from RNN_FWandBW import Vanilla_RNN

from KalmanFilter_test import KFTest
from RTS_Smoother_test import S_Test

from Plot import Plot_RTS as Plot

import wandb

from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from parameters import F, F_gen, H_identity, H_onlyPos,Q,Q_gen,R,R_onlyPos,\
m1x_0, m2x_0, m, n,delta_t_gen,delta_t,T,T_test,T_gen,T_test_gen

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
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

#############################
###  Dataset Generation   ###
#############################
offset = 0
InitIsRandom_train = True
InitIsRandom_cv = True
InitIsRandom_test = True
Loss_On_AllState = False # if false: only calculate loss on position
Train_Loss_On_AllState = True # if false: only calculate training loss on position
DatafolderName = 'Simulations/Linear_CA/data/'
DatafileName = 'decimated_dt1e-2_T100_r0_randnInit.pt'
# data_gen = 'dt1e-3_T10000_rq00.pt'
# Generation model
sys_model_gen = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, T_gen, T_test_gen,onlyPos=True)
sys_model_gen.InitSequence(m1x_0, m2x_0)# x0 and P0
# Decimated model
# sys_model = SystemModel(F, Q, H_onlyPos, R_onlyPos, T, T_test,onlyPos=True)
# sys_model.InitSequence(m1x_0, m2x_0)

# print("Start Data Gen")
# DataGen(sys_model_gen, DatafolderName+DatafileName, T_gen, T_test_gen,randomInit_train=InitIsRandom_train,randomInit_cv=InitIsRandom_cv,randomInit_test=InitIsRandom_test)
print("Load Original Data")
if(InitIsRandom_train or InitIsRandom_cv or InitIsRandom_test):
   [train_input, train_target, train_init, cv_input, cv_target, cv_init, test_input, test_target, test_init] = torch.load(DatafolderName+DatafileName,map_location=dev)
else:
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(DatafolderName+DatafileName, map_location=dev)
print("Data Shape")
print("testset size:",test_target.size())
print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())

### Further Decimation
# print("Start Data Decimation")
# test_target = DecimateData(test_target,delta_t_gen,delta_t, offset=offset) 
# train_target = DecimateData(train_target,delta_t_gen,delta_t, offset=offset)
# cv_target = DecimateData(cv_target,delta_t_gen,delta_t, offset=offset)
# test_input = DecimateData(test_input,delta_t_gen,delta_t, offset=offset) 
# train_input = DecimateData(train_input,delta_t_gen,delta_t, offset=offset)
# cv_input = DecimateData(cv_input,delta_t_gen,delta_t, offset=offset)
# print("Decimated Data Shape")
# print("testset size:",test_target.size())
# print("trainset size:",train_target.size())
# print("cvset size:",cv_target.size())
# print("Load Decimated Data")
# [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(DatafolderName+DatafileName, map_location=dev)

print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter")
if InitIsRandom_test:
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model_gen, test_input, test_target, allStates=Loss_On_AllState, randomInit = True, test_init=test_init)
else: 
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model_gen, test_input, test_target, allStates=Loss_On_AllState)

#############################
### Evaluate RTS Smoother ###
#############################
print("Evaluate RTS Smoother")
if InitIsRandom_test:
   [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out] = S_Test(sys_model_gen, test_input, test_target, allStates=Loss_On_AllState, randomInit = True,test_init=test_init)
else:
   [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out] = S_Test(sys_model_gen, test_input, test_target, allStates=Loss_On_AllState)

#######################
### RTSNet Pipeline ###
#######################

### RTSNet with full info ##############################################################################################
# Build Neural Network
wandb.init(project="RTSNet_LinearCA")
print("RTSNet pipeline start!")
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model_gen)
print("Number of trainable parameters for RTSNet:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
## Train Neural Network
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model_gen)
RTSNet_Pipeline.setModel(RTSNet_model)
RTSNet_Pipeline.setTrainingParams(n_Epochs=5000, n_Batch=10, learningRate=1E-4, weightDecay=1E-4)
# RTSNet_Pipeline.model = torch.load('RTSNet/new_architecture/linear_Journal/rq020_T100_randinit.pt',map_location=dev)
### Optinal: record parameters to wandb
wandb.log({
"Train_Loss_On_AllState": Train_Loss_On_AllState,
"Test_Loss_On_AllState": Loss_On_AllState,
"learning_rate": RTSNet_Pipeline.learningRate,
"batch_size": RTSNet_Pipeline.N_B,
"weight_decay": RTSNet_Pipeline.weightDecay})
#######################################
if (InitIsRandom_train or InitIsRandom_cv or InitIsRandom_test):
   print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_gen, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState, randomInit = True, cv_init=cv_init,train_init=train_init)
   ## Test Neural Network
   print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model_gen, test_input, test_target, path_results,MaskOnState=not Loss_On_AllState,randomInit=True,test_init=test_init)
else:
   print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_gen, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState)
   ## Test Neural Network
   print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model_gen, test_input, test_target, path_results,MaskOnState=not Loss_On_AllState)
RTSNet_Pipeline.save()

### Vanilla RNN with full info ###################################################################################
## Build RNN
# print("Vanilla RNN with full model info")
# RNN_model = Vanilla_RNN()
# RNN_model.Build(sys_model,fully_agnostic = False)
# print("Number of trainable parameters for RNN:",sum(p.numel() for p in RNN_model.parameters() if p.requires_grad))
# RNN_Pipeline = Pipeline(strTime, "RTSNet", "VanillaRNN")
# RNN_Pipeline.setssModel(sys_model)
# RNN_Pipeline.setModel(RNN_model)
# RNN_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=50, learningRate=1e-3, weightDecay=1e-5)
# RNN_Pipeline.model = torch.load('RNN/linear/2x2_rq020_T100.pt',map_location=dev)
# RNN_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, rnn=True)
# ## Test Neural Network
# [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RNN_Pipeline.NNTest(sys_model, test_input, test_target, path_results, rnn=True)
# RNN_Pipeline.save()

# Close wandb run 
wandb.finish()  