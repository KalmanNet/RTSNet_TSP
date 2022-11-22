import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from Linear_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, DecimateData,wandb_switch
from Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Pipeline_concat_models import Pipeline_twoRTSNets

from datetime import datetime
from RTSNet_nn import RTSNetNN
from RNN_FWandBW import Vanilla_RNN

from KalmanFilter_test import KFTest
from RTS_Smoother_test import S_Test

from Plot import Plot_RTS as Plot

if wandb_switch:
   import wandb
   wandb.init(project="RTSNet_LinearCA")

from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from parameters import F, F_gen, F_CV, H_identity, H_onlyPos,Q,Q_gen,Q_CV, R_2,R_3,R_onlyPos,\
m1x_0, m2x_0,m1x_0_cv, m2x_0_cv, m, m_cv,n,delta_t_gen,delta_t,T,T_test,T_gen,T_test_gen

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
# Init condition of dataset
offset = 0
InitIsRandom_train = True
KnownRandInit_train = True
InitIsRandom_cv = True
KnownRandInit_cv = True
InitIsRandom_test = True
KnownRandInit_test = True

# PVA or P
Loss_On_AllState = True # if false: only calculate loss on position
Train_Loss_On_AllState = True # if false: only calculate training loss on position
CV_model = False # if true: use CV model, else: use CA model

# 1pass or 2pass
two_pass = True # if true: use two pass method, else: use one pass method
load_trained_pass1 = True # if True: load trained RTSNet pass1, else train pass1
# if true, specify the path to the trained pass1 model
RTSNetPass1_path = "RTSNet/new_architecture/linear_Journal/linearCA/knownInit/CA_trainPVA.pt"
# Save the dataset generated from testing RTSNet1 on train and CV data
load_dataset_for_pass2 = True # if True: load dataset generated from testing RTSNet1 on train and CV data
# if true, specify the path to the dataset
DatasetPass1_path = "Simulations/Linear_CA/data/ResultofPass1_PVA.pt" 

DatafolderName = 'Simulations/Linear_CA/data/'
DatafileName = 'New_decimated_dt1e-2_T100_r0_randnInit.pt'
# data_gen = 'dt1e-3_T10000_rq00.pt'

####################
### System Model ###
####################
# Generation model (CA)
sys_model_gen = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, T_gen, T_test_gen)
sys_model_gen.InitSequence(m1x_0, m2x_0)# x0 and P0

# Feed model (to KF, RTS and RTSNet) 
if(KnownRandInit_train or KnownRandInit_cv or KnownRandInit_test):
   std = 0
   m2x_0 = std * std * torch.eye(m) # Initial Covariance
   m2x_0_cv = std * std * torch.eye(m_cv) # Initial Covariance for CV
sys_model = SystemModel(F_gen, Q_gen, H_onlyPos, R_onlyPos, T_gen, T_test_gen)
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0
if CV_model:
   m1x_0 = m1x_0_cv
   m2x_0 = m2x_0_cv
   H_onlyPos = torch.tensor([[1, 0]]).float()
   sys_model = SystemModel(F_CV, Q_CV, H_onlyPos, R_onlyPos, T_gen, T_test_gen)
   sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

# Feed model (RTSNet pass2) 
sys_model_pass2 = SystemModel(F_gen, Q_gen, H_identity, R_3, T_gen, T_test_gen)
sys_model_pass2.InitSequence(m1x_0, m2x_0)# x0 and P0
if CV_model:
   sys_model_pass2 = SystemModel(F_CV, Q_CV, torch.eye(2), R_2, T_gen, T_test_gen)
   sys_model_pass2.InitSequence(m1x_0, m2x_0)# x0 and P0

# Decimated model
# sys_model = SystemModel(F, Q, H_onlyPos, R_onlyPos, T, T_test)
# sys_model.InitSequence(m1x_0, m2x_0)

# print("Start Data Gen")
# DataGen(sys_model_gen, DatafolderName+DatafileName, T_gen, T_test_gen,randomInit_train=InitIsRandom_train,randomInit_cv=InitIsRandom_cv,randomInit_test=InitIsRandom_test)
print("Load Original Data")
if(InitIsRandom_train or InitIsRandom_cv or InitIsRandom_test):
   [train_input, train_target, train_init, cv_input, cv_target, cv_init, test_input, test_target, test_init] = torch.load(DatafolderName+DatafileName,map_location=dev)
   if CV_model:# set state as (p,v) instead of (p,v,a)
      train_target = train_target[:,0:m_cv,:]
      train_init = train_init[:,0:m_cv]
      cv_target = cv_target[:,0:m_cv,:]
      cv_init = cv_init[:,0:m_cv]
      test_target = test_target[:,0:m_cv,:]
      test_init = test_init[:,0:m_cv]

else:
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(DatafolderName+DatafileName, map_location=dev)
   if CV_model:# set state as (p,v) instead of (p,v,a)
      train_target = train_target[:,0:m_cv,:]
      cv_target = cv_target[:,0:m_cv,:]
      test_target = test_target[:,0:m_cv,:]

print("Data Shape")
print("testset state x size:",test_target.size())
print("testset observation y size:",test_input.size())
print("trainset state x size:",train_target.size())
print("trainset observation y size:",train_input.size())
print("cvset state x size:",cv_target.size())
print("cvset observation y size:",cv_input.size())

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
if InitIsRandom_test and KnownRandInit_test:
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, test_input, test_target, allStates=Loss_On_AllState, randomInit = True, test_init=test_init)
else: 
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, test_input, test_target, allStates=Loss_On_AllState)

#############################
### Evaluate RTS Smoother ###
#############################
print("Evaluate RTS Smoother")
if InitIsRandom_test and KnownRandInit_test:
   [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out] = S_Test(sys_model, test_input, test_target, allStates=Loss_On_AllState, randomInit = True,test_init=test_init)
else:
   [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out] = S_Test(sys_model, test_input, test_target, allStates=Loss_On_AllState)

#######################
### RTSNet Pipeline ###
#######################
### RTSNet with full info ##############################################################################################
if load_trained_pass1:
   print("Load RTSNet pass 1")
else:
   # Build Neural Network
   print("RTSNet pass 1 pipeline start!")
   RTSNet_model = RTSNetNN()
   RTSNet_model.NNBuild(sys_model)
   print("Number of trainable parameters for RTSNet pass 1:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
   ## Train Neural Network
   RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
   RTSNet_Pipeline.setssModel(sys_model)
   RTSNet_Pipeline.setModel(RTSNet_model)
   RTSNet_Pipeline.setTrainingParams(n_Epochs=4000, n_Batch=10, learningRate=1E-4, weightDecay=1E-4)
   # RTSNet_Pipeline.model = torch.load('RTSNet/new_architecture/linear_Journal/linearCA/CA_TrainP.pt',map_location=dev)
   ### Optinal: record parameters to wandb
   if wandb_switch:
      wandb.log({
      "Train_Loss_On_AllState": Train_Loss_On_AllState,
      "Test_Loss_On_AllState": Loss_On_AllState,
      "learning_rate": RTSNet_Pipeline.learningRate,
      "batch_size": RTSNet_Pipeline.N_B,
      "weight_decay": RTSNet_Pipeline.weightDecay})
   #######################################
   if (KnownRandInit_train):
      print("Train RTSNet with Known Random Initial State")
      print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState, randomInit = True, cv_init=cv_init,train_init=train_init)
   else:
      print("Train RTSNet with Unknown Initial State")
      print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, MaskOnState=not Train_Loss_On_AllState)
      
   if (KnownRandInit_test): 
      print("Test RTSNet pass 1 with Known Random Initial State")
      ## Test Neural Network
      print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
      [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,MaskOnState=not Loss_On_AllState,randomInit=True,test_init=test_init)
   else: 
      print("Test RTSNet pass 1 with Unknown Initial State")
      ## Test Neural Network
      print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
      [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,MaskOnState=not Loss_On_AllState)

   RTSNet_Pipeline.save()
   print("RTSNet pass 1 pipeline end!")

if two_pass:
   #########################
   ## Concat two RTSNets ###
   #########################
   if load_dataset_for_pass2:
      print("Load dataset for pass 2")
      if(InitIsRandom_train or InitIsRandom_cv or InitIsRandom_test):
         [train_input_pass2, train_target_pass2, train_init, cv_input_pass2, cv_target_pass2, cv_init, test_input, test_target, test_init] = torch.load(DatasetPass1_path,map_location=dev)
         if CV_model:# set state as (p,v) instead of (p,v,a)
            train_target_pass2 = train_target_pass2[:,0:m_cv,:]
            train_init = train_init[:,0:m_cv]
            cv_target_pass2 = cv_target_pass2[:,0:m_cv,:]
            cv_init = cv_init[:,0:m_cv]
            test_target = test_target[:,0:m_cv,:]
            test_init = test_init[:,0:m_cv]

      else:
         [train_input_pass2, train_target_pass2, cv_input_pass2, cv_target_pass2, test_input, test_target] = torch.load(DatasetPass1_path, map_location=dev)
         if CV_model:# set state as (p,v) instead of (p,v,a)
            train_target_pass2 = train_target_pass2[:,0:m_cv,:]
            cv_target_pass2 = cv_target_pass2[:,0:m_cv,:]
            test_target = test_target[:,0:m_cv,:]

      

      print("Data Shape for RTSNet pass 2:")
      print("testset state x size:",test_target.size())
      print("testset observation y size:",test_input.size())
      print("trainset state x size:",train_target_pass2.size())
      print("trainset observation y size:",len(train_input_pass2),train_input_pass2[0].size())
      print("cvset state x size:",cv_target_pass2.size())
      print("cvset observation y size:",len(cv_input_pass2),cv_input_pass2[0].size())
   else:
      ### load result of RTSNet1 as dataset for RTSNet2 ###############################################################
      RTSNet_model_pass1 = RTSNetNN()
      RTSNet_model_pass1.NNBuild(sys_model)
      RTSNet_Pipeline_pass1 = Pipeline(strTime, "RTSNet", "RTSNet")
      RTSNet_Pipeline_pass1.setssModel(sys_model)
      RTSNet_Pipeline_pass1.setModel(RTSNet_model_pass1)
      if (KnownRandInit_train):
         print("Test RTSNet pass 1 on training set")
         [_, _, _,rtsnet_out_train,_] = RTSNet_Pipeline_pass1.NNTest(sys_model, train_input, train_target, path_results,MaskOnState=False,randomInit=True,test_init=train_init,load_model=True,load_model_path=RTSNetPass1_path)
         print("Test RTSNet pass 1 on cv set")
         [_, _, _,rtsnet_out_cv,_] = RTSNet_Pipeline_pass1.NNTest(sys_model, cv_input, cv_target, path_results,MaskOnState=False,randomInit=True,test_init=cv_init,load_model=True,load_model_path=RTSNetPass1_path)
      else:
         print("Test RTSNet pass 1 on training set")
         [_, _, _,rtsnet_out_train,_] = RTSNet_Pipeline_pass1.NNTest(sys_model, train_input, train_target, path_results,MaskOnState=False,load_model=True,load_model_path=RTSNetPass1_path)
         print("Test RTSNet pass 1 on cv set")
         [_, _, _,rtsnet_out_cv,_] = RTSNet_Pipeline_pass1.NNTest(sys_model, cv_input, cv_target, path_results,MaskOnState=False,load_model=True,load_model_path=RTSNetPass1_path)


      train_input_pass2 = rtsnet_out_train
      train_target_pass2 = train_target
      cv_input_pass2 = rtsnet_out_cv
      cv_target_pass2 = cv_target

      if(InitIsRandom_train or InitIsRandom_cv or InitIsRandom_test):
         torch.save([train_input_pass2, train_target_pass2, train_init, cv_input_pass2, cv_target_pass2, cv_init, test_input, test_target, test_init], DatasetPass1_path)
      else:
         torch.save([train_input_pass2, train_target_pass2, cv_input_pass2, cv_target_pass2, test_input, test_target], DatasetPass1_path)


   # Build Neural Network
   print("RTSNet pass 2 pipeline start!")
   RTSNet_model = RTSNetNN()
   RTSNet_model.NNBuild(sys_model_pass2)
   print("Number of trainable parameters for RTSNet pass 2:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
   ## Train Neural Network
   RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet_pass2")
   RTSNet_Pipeline.setssModel(sys_model_pass2)
   RTSNet_Pipeline.setModel(RTSNet_model)
   RTSNet_Pipeline.setTrainingParams(n_Epochs=4000, n_Batch=10, learningRate=1E-4, weightDecay=1E-4)
   ### Optinal: record parameters to wandb
   if wandb_switch:
      wandb.log({
      "Train_Loss_On_AllState_pass2": Train_Loss_On_AllState,
      "Test_Loss_On_AllState_pass2": Loss_On_AllState,
      "learning_rate_pass2": RTSNet_Pipeline.learningRate,
      "batch_size_pass2": RTSNet_Pipeline.N_B,
      "weight_decay_pass2": RTSNet_Pipeline.weightDecay})
   #######################################
   if (KnownRandInit_train):
      print("Train RTSNet pass 2 with Known Random Initial State")
      print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_pass2, cv_input_pass2, cv_target_pass2, train_input_pass2, train_target_pass2, path_results, MaskOnState=not Train_Loss_On_AllState, randomInit = True, cv_init=cv_init,train_init=train_init)
   else:
      print("Train RTSNet pass 2 with Unknown Initial State")
      print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_pass2, cv_input_pass2, cv_target_pass2, train_input_pass2, train_target_pass2, path_results, MaskOnState=not Train_Loss_On_AllState)
   RTSNet_Pipeline.save()
   print("RTSNet pass 2 pipeline end!")


   # load trained Neural Network
   print("Concat two RTSNets")
   RTSNet_model1 = torch.load(RTSNetPass1_path,map_location=dev)
   RTSNet_model2 = torch.load('RTSNet/best-model.pt',map_location=dev)
   ## Set up Neural Network
   RTSNet_Pipeline_2passes = Pipeline_twoRTSNets(strTime, "RTSNet", "RTSNet")
   RTSNet_Pipeline_2passes.setModel(RTSNet_model1, RTSNet_model2)
   NumofParameter = RTSNet_Pipeline_2passes.count_parameters()
   print("Number of parameters for RTSNet with 2 passes: ",NumofParameter)
   ## Test Neural Network
   if (KnownRandInit_test): 
      print("Test RTSNet(2passes)with Known Random Initial State")
      print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
      [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline_2passes.NNTest(sys_model, test_input, test_target, path_results,MaskOnState=not Loss_On_AllState,randomInit=True,test_init=test_init)
   else: 
      print("Test RTSNet(2passes) with Unknown Initial State")
      print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
      [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline_2passes.NNTest(sys_model, test_input, test_target, path_results,MaskOnState=not Loss_On_AllState)


##################
## Vanilla RNN ###
##################
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
if wandb_switch: 
   wandb.finish()  

####################
### Plot results ###
####################
# PlotfolderName = "Graphs/Linear_CA/"
# PlotfileName0 = "TrainPVA_position.png"
# PlotfileName1 = "TrainPVA_velocity.png"
# PlotfileName2 = "TrainPVA_acceleration.png"

# Plot = Plot(PlotfolderName, PlotfileName0)
# print("Plot")
# Plot.plotTraj_CA(test_target, RTS_out, rtsnet_out, dim=0, file_name=PlotfolderName+PlotfileName0)
# Plot.plotTraj_CA(test_target, RTS_out, rtsnet_out, dim=1, file_name=PlotfolderName+PlotfileName1)
# Plot.plotTraj_CA(test_target, RTS_out, rtsnet_out, dim=2, file_name=PlotfolderName+PlotfileName2)