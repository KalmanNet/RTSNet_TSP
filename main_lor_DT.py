import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from EKF_test import EKFTest
from Extended_RTS_Smoother_test import S_Test
from Extended_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T
from Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Pipeline_EKF import Pipeline_EKF
# from PF_test import PFTest

from datetime import datetime

from KalmanNet_nn import KalmanNetNN
from RTSNet_nn import RTSNetNN

from Pipeline_ERTS_2passes import Pipeline_ERTS as Pipeline_2passes
from RTSNet_nn_2passes import RTSNetNN_2passes

from Plot import Plot_extended as Plot

from filing_paths import path_model, path_session
import sys
sys.path.insert(1, path_model)
from parameters import T, T_test, m1x_0, m2x_0, m, n,H_mod, H_mod_inv
from model import f, h, fInacc, hRotate, fRotate, h_nonlinear

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

###################
###  Settings   ###
###################
offset = 0
chop = False
sequential_training = False
path_results = 'ERTSNet/'
DatafolderName = 'Simulations/Lorenz_Atractor/data/T100_Hrot1' + '/'

r2 = torch.tensor([1])
# r2 = torch.tensor([100, 10, 1, 0.1, 0.01])
r = torch.sqrt(r2)
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)

q2 = torch.mul(v,r2)
q = torch.sqrt(q2)

# q and r optimized for EKF and MB RTS
# r2optdB = torch.tensor([16.9897])
# ropt = torch.sqrt(10**(-r2optdB/10))
# q2optdB = torch.tensor([28.2391])
# qopt = torch.sqrt(10**(-q2optdB/10))

print("1/r2 [dB]: ", 10 * torch.log10(1/r[0]**2))
print("1/q2 [dB]: ", 10 * torch.log10(1/q[0]**2))

# traj_resultName = ['traj_lor_KNetFull_rq1030_T2000_NT100.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
dataFileName = ['data_lor_v20_rq020_T100.pt']#,'data_lor_v20_r1e-2_T100.pt','data_lor_v20_r1e-3_T100.pt','data_lor_v20_r1e-4_T100.pt']
# KFRTSResultName = 'KFRTS_partialh_rq3050_T2000' 

#########################################
###  Generate and load data DT case   ###
#########################################
sys_model = SystemModel(f, q[0], hRotate, r[0], T, T_test, m, n)
sys_model.InitSequence(m1x_0, m2x_0)
print("Start Data Gen")
DataGen(sys_model, DatafolderName + dataFileName[0], T, T_test,randomInit=False)
print("Data Load")
print(dataFileName[0])
[train_input_long,train_target_long, cv_input, cv_target, test_input, test_target] =  torch.load(DatafolderName + dataFileName[0],map_location=dev)  
if chop: 
   print("chop training data")    
   [train_target, train_input, train_init] = Short_Traj_Split(train_target_long, train_input_long, T)
   # [cv_target, cv_input] = Short_Traj_Split(cv_target, cv_input, T)
else:
   print("no chopping") 
   train_target = train_target_long[:,:,0:T]
   train_input = train_input_long[:,:,0:T] 
   # cv_target = cv_target[:,:,0:T]
   # cv_input = cv_input[:,:,0:T]  

print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())


# Model with partial info
sys_model_partial = SystemModel(f, q[0], h, r[0], T, T_test, m, n)
sys_model_partial.InitSequence(m1x_0, m2x_0)

########################################
### Evaluate Observation Noise Floor ###
########################################
N_T = len(test_input)
loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(N_T)# MSE [Linear]

for j in range(0, N_T): 
   reversed_target = torch.matmul(H_mod_inv, test_input[j])      
   MSE_obs_linear_arr[j] = loss_obs(reversed_target, test_target[j]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor(test dataset) - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor(test dataset) - STD:", obs_std_dB, "[dB]")


######################################
### Evaluate Filters and Smoothers ###
######################################
#Evaluate EKF true
print("Evaluate EKF true")
[MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(sys_model, test_input, test_target)
#Evaluate EKF partial (h or r)
print("Evaluate EKF partial")
[MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKFTest(sys_model_partial, test_input, test_target)
#Evaluate EKF partial optq
#  [MSE_EKF_linear_arr_partialoptq, MSE_EKF_linear_avg_partialoptq, MSE_EKF_dB_avg_partialoptq, EKF_KG_array_partialoptq, EKF_out_partialoptq] = EKFTest(sys_model_partialf_optq, test_input, test_target)
#  #Evaluate EKF partialh optr
# print("Evaluate EKF partial")
# [MSE_EKF_linear_arr_partialoptr, MSE_EKF_linear_avg_partialoptr, MSE_EKF_dB_avg_partialoptr, EKF_KG_array_partialoptr, EKF_out_partialoptr] = EKFTest(sys_model_partialh, test_input, test_target)
#Eval PF partial
# [MSE_PF_linear_arr_partial, MSE_PF_linear_avg_partial, MSE_PF_dB_avg_partial, PF_out_partial, t_PF] = PFTest(sys_model_partialh, test_input, test_target, init_cond=None)
# print(f"MSE PF H NL: {MSE_PF_dB_avg_partial} [dB] (T = {T_test})")
#Evaluate RTS true
print("Evaluate RTS true")
[MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out] = S_Test(sys_model, test_input, test_target)
#Evaluate RTS partialh optr
print("Evaluate RTS partial")
[MSE_ERTS_linear_arr_partialoptr, MSE_ERTS_linear_avg_partialoptr, MSE_ERTS_dB_avg_partialoptr, ERTS_out_partialoptr] = S_Test(sys_model_partial, test_input, test_target)


# Save results

# KFRTSfolderName = 'ERTSNet' + '/'
# torch.save({'MSE_EKF_linear_arr': MSE_EKF_linear_arr,
#             'MSE_EKF_dB_avg': MSE_EKF_dB_avg,
#             'MSE_EKF_linear_arr_partialoptr': MSE_EKF_linear_arr_partialoptr,
#             'MSE_EKF_dB_avg_partialoptr': MSE_EKF_dB_avg_partialoptr,
#             'MSE_ERTS_linear_arr': MSE_ERTS_linear_arr,
#             'MSE_ERTS_dB_avg': MSE_ERTS_dB_avg,
#             'MSE_ERTS_linear_arr_partialoptr': MSE_ERTS_linear_arr_partialoptr,
#             'MSE_ERTS_dB_avg_partialoptr': MSE_ERTS_dB_avg_partialoptr,
#             }, KFRTSfolderName+KFRTSResultName)

# # Save trajectories
# trajfolderName = 'KNet' + '/'
# DataResultName = traj_resultName[rindex]
# EKF_sample = torch.reshape(EKF_out[0,:,:],[1,m,T_test])
# EKF_Partial_sample = torch.reshape(EKF_out_partial[0,:,:],[1,m,T_test])
# target_sample = torch.reshape(test_target[0,:,:],[1,m,T_test])
# input_sample = torch.reshape(test_input[0,:,:],[1,n,T_test])
# KNet_sample = torch.reshape(KNet_test[0,:,:],[1,m,T_test])
# torch.save({
#             'KNet': KNet_test,
#             }, trajfolderName+DataResultName)

# ## Save histogram
# EKFfolderName = 'KNet' + '/'
# torch.save({'KNet_MSE_test_linear_arr': KNet_MSE_test_linear_arr,
#             'KNet_MSE_test_dB_avg': KNet_MSE_test_dB_avg,
#             }, EKFfolderName+EKFResultName)

#######################
### Evaluate RTSNet ###
#######################
## RTSNet with full info
## Build Neural Network
print("RTSNet with full model info")
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model)
# ## Train Neural Network
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model)
print("Number of trainable parameters for RTSNet:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
RTSNet_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=30, learningRate=1e-3, weightDecay=1e-6) 
# RTSNet_Pipeline.model = torch.load('ERTSNet/best-model_DTfull_rq3050_T2000.pt',map_location=dev)
if(chop):
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results,randomInit=True,train_init=train_init)
else:
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
## Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)

####################################################################################

## RTSNet_2passes with full info 
# print("RTSNet_2passes with full model info")
# RTSNet_model = RTSNetNN_2passes()
# RTSNet_model.NNBuild(sys_model)
# ## Train Neural Network
# RTSNet_Pipeline = Pipeline_2passes(strTime, "RTSNet", "RTSNet")
# RTSNet_Pipeline.setssModel(sys_model)
# RTSNet_Pipeline.setModel(RTSNet_model)
# RTSNet_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=1, learningRate=1e-4, weightDecay=1e-3)
# print("Number of trainable parameters for RTSNet_2passes:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
# if(chop):
#    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results,randomInit=True,train_init=train_init)
# else:
#    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
# ## Test Neural Network
# # RTSNet_Pipeline.model = torch.load('ERTSNet/model_KNetNew_DT_procmis_r30q50_T2000.pt',map_location=cuda0)
# [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)

####################################################################################

## RTSNet with model mismatch
## Build Neural Network
print("RTSNet with observation model mismatch")
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model_partial)
## Train Neural Network
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model_partial)
RTSNet_Pipeline.setModel(RTSNet_model)
RTSNet_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=20, learningRate=1e-4, weightDecay=1e-6)
# RTSNet_Pipeline.model = torch.load('ERTSNet/best-model_DTfull_rq3050_T2000.pt',map_location=cuda0)
if(chop):
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_partial, cv_input, cv_target, train_input, train_target, path_results,randomInit=True,train_init=train_init)
else:
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_partial, cv_input, cv_target, train_input, train_target, path_results)
## Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model_partial, test_input, test_target, path_results)

###################################################################################

print("RTSNet with estimated H")
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNetEstH_"+ dataFileName[0])
print("True Observation matrix H:", H_mod)
### Least square estimation of H
X = torch.squeeze(train_target[:,:,0]).to(dev,non_blocking = True)
Y = torch.squeeze(train_input[:,:,0]).to(dev,non_blocking = True)
for t in range(1,T):
   X_t = torch.squeeze(train_target[:,:,t])
   Y_t = torch.squeeze(train_input[:,:,t])
   X = torch.cat((X,X_t),0)
   Y = torch.cat((Y,Y_t),0)
Y_1 = torch.unsqueeze(Y[:,0],1)
Y_2 = torch.unsqueeze(Y[:,1],1)
Y_3 = torch.unsqueeze(Y[:,2],1)
H_row1 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T,X)),X.T),Y_1).to(dev,non_blocking = True)
H_row2 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T,X)),X.T),Y_2).to(dev,non_blocking = True)
H_row3 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T,X)),X.T),Y_3).to(dev,non_blocking = True)
H_hat = torch.cat((H_row1.T,H_row2.T,H_row3.T),0)
print("Estimated Observation matrix H:", H_hat)

def h_hat(x):
 return torch.matmul(H_hat,x)

# Estimated model
sys_model_esth = SystemModel(f, q[0], h_hat, r[0], T, T_test, m, n)
sys_model_esth.InitSequence(m1x_0, m2x_0)

RTSNet_Pipeline.setssModel(sys_model_esth)
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model_esth)
RTSNet_Pipeline.setModel(RTSNet_model)

RTSNet_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=30, learningRate=1E-4, weightDecay=1E-3)
# RTSNet_Pipeline.model = torch.load('RTSNet/new_architecture/linear/best-model_hrot10_linear2x2rq-1010T100.pt',map_location=dev)
[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_esth, cv_input, cv_target, train_input, train_target, path_results)
## Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model_esth, test_input, test_target, path_results)
RTSNet_Pipeline.save()

############################
###  KNet for comparison ###
############################
# KNet with model mismatch
## Build Neural Network
# KNet_model = KalmanNetNN()
# KNet_model.NNBuild(sys_model_partialf)
# ## Train Neural Network
# KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
# KNet_Pipeline.setModel(KNet_model)
# KNet_Pipeline.setssModel(sys_model_partialf)
# KNet_Pipeline.setTrainingParams(n_Epochs=100, n_Batch=10, learningRate=1e-3, weightDecay=1e-6)
# [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model_partialf, cv_input, cv_target, train_input, train_target, path_results, sequential_training)
# ## Test Neural Network
# # KNet_Pipeline.model = torch.load('KNet/model_KNetNew_DT_procmis_r30q50_T2000.pt',map_location=cuda0)
# [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, KNet_KG_array, knet_out,RunTime] = KNet_Pipeline.NNTest(sys_model_partialf, test_input, test_target, path_results)

 

   





