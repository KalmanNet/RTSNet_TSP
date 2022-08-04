import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from Linear_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T, F, H, F_rotated, H_rotated, T, T_test, m1_0, m2_0, m, n
from Pipeline_ERTS import Pipeline_ERTS as Pipeline

from datetime import datetime
from RTSNet_nn import RTSNetNN
from RNN_FWandBW import Vanilla_RNN

from KalmanFilter_test import KFTest
from RTS_Smoother_test import S_Test

from Plot import Plot_RTS as Plot

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

####################
### Design Model ###
####################
InitIsRandom = False
LengthIsRandom = False
r2 = torch.tensor([1])
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)
print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

# True model
r = torch.sqrt(r2)
q = torch.sqrt(q2)
sys_model = SystemModel(F, q, H, r, T, T_test)
sys_model.InitSequence(m1_0, m2_0)

# Mismatched model
# sys_model_partialh = SystemModel(F, q, H_rotated, r, T, T_test)
# sys_model_partialh.InitSequence(m1_0, m2_0)

###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Linear_canonical/Scaling_to_large_models' + '/'
dataFileName = '2x2_rq020_T20.pt'
# print("Start Data Gen")
# DataGen(sys_model, dataFolderName + dataFileName, T, T_test,randomInit=InitIsRandom,randomLength=LengthIsRandom)
print("Data Load")
if(InitIsRandom):
   [train_input, train_target, train_init, cv_input, cv_target, cv_init, test_input, test_target, test_init] = torch.load(dataFolderName + dataFileName,map_location=dev)
   print("trainset size:",train_target.size())
   print("cvset size:",cv_target.size())
   print("testset size:",test_target.size())
elif(LengthIsRandom):
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(dataFolderName + dataFileName,map_location=dev)
   ### Check sequence lengths
   # for sequences in train_target:
   #    print("trainset size:",sequences.size())
   # for sequences in test_target:
   #    print("testset size:",sequences.size())
else:
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName)
   print("trainset size:",train_target.size())
   print("cvset size:",cv_target.size())
   print("testset size:",test_target.size())
  


########################################
### Evaluate Observation Noise Floor ###
########################################
loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(N_T)# MSE [Linear]
for j in range(0, N_T):        
   MSE_obs_linear_arr[j] = loss_obs(test_input[j], test_target[j]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor - STD:", obs_std_dB, "[dB]")

##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter True")
if InitIsRandom:
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, test_input, test_target, randomInit = True, test_init=test_init)
else: 
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, test_input, test_target)
# print("Evaluate Kalman Filter Partial")
# [MSE_KF_linear_arr_partialh, MSE_KF_linear_avg_partialh, MSE_KF_dB_avg_partialh] = KFTest(sys_model_partialh, test_input, test_target)

#############################
### Evaluate RTS Smoother ###
#############################
print("Evaluate RTS Smoother True")
if InitIsRandom:
   [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out] = S_Test(sys_model, test_input, test_target, randomInit = True,test_init=test_init)
else:
   [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out] = S_Test(sys_model, test_input, test_target)
# print("Evaluate RTS Smoother Partial")
# [MSE_RTS_linear_arr_partialh, MSE_RTS_linear_avg_partialh, MSE_RTS_dB_avg_partialh] = S_Test(sys_model_partialh, test_input, test_target)

# PlotfolderName = 'Graphs' + '/'
# ComparedmodelName = 'Dataset'  
# Plot = Plot(PlotfolderName, ComparedmodelName)
# print("Plot")
# Plot.NNPlot_Hist(MSE_KF_linear_arr, MSE_RTS_linear_arr, MSE_obs_linear_arr)

##############################
###  Compare KF and RTS    ###
##############################
# r2 = torch.tensor([2, 1, 0.5, 0.1])
# r = torch.sqrt(r2)
# q = r
# MSE_KF_RTS_dB = torch.empty(size=[2,len(r)]).to(cuda0)
# dataFileName = ['data_2x2_r2q2_T20_Ttest20.pt','data_2x2_r1q1_T20_Ttest20.pt','data_2x2_r0.5q0.5_T20_Ttest20.pt','data_2x2_r0.1q0.1_T20_Ttest20.pt']
# for rindex in range(0, len(r)):
#     #Generate and load data
#     SysModel_design = SystemModel(F, torch.squeeze(q[rindex]), H, torch.squeeze(r[rindex]), T, T_test)  
#     SysModel_design.InitSequence(m1_0, m2_0)
#     DataGen(SysModel_design, dataFolderName + dataFileName[rindex], T, T_test)
#     [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName[rindex])
#     #Evaluate KF and RTS
#     [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(SysModel_design, test_input, test_target)
#     [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg] = S_Test(SysModel_design, test_input, test_target)
#     MSE_KF_RTS_dB[0,rindex] = MSE_KF_dB_avg
#     MSE_KF_RTS_dB[1,rindex] = MSE_RTS_dB_avg

# PlotfolderName = 'Graphs' + '/'
#PlotResultName = 'Linear_KFandRTS'  
# Plot = Plot(PlotfolderName, PlotResultName)
# print("Plot")
# Plot.KF_RTS_Plot(r, MSE_KF_RTS_dB)



#######################
### RTSNet Pipeline ###
#######################

### RTSNet with full info ##############################################################################################
# Build Neural Network
# print("RTSNet with full model info")
# RTSNet_model = RTSNetNN()
# RTSNet_model.NNBuild(sys_model)
# print("Number of trainable parameters for RTSNet:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
# ## Train Neural Network
# RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
# RTSNet_Pipeline.setssModel(sys_model)
# RTSNet_Pipeline.setModel(RTSNet_model)
# # RTSNet_Pipeline.setTrainingParams(n_Epochs=10000, n_Batch=1, learningRate=1E-5, weightDecay=1E-2)
# RTSNet_Pipeline.model = torch.load('RTSNet/new_architecture/linear_Journal/rq020_T100_randinit.pt',map_location=dev)
# if InitIsRandom:
#    # [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, randomInit = True, cv_init=cv_init,train_init=train_init)
#    ## Test Neural Network
#    [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,randomInit=True,test_init=test_init)
# else:
#    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
#    ## Test Neural Network
#    [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
# RTSNet_Pipeline.save()

### Vanilla RNN with full info ###################################################################################
## Build RNN
print("Vanilla RNN with full model info")
RNN_model = Vanilla_RNN()
RNN_model.Build(sys_model,fully_agnostic = False)
print("Number of trainable parameters for RNN:",sum(p.numel() for p in RNN_model.parameters() if p.requires_grad))
RNN_Pipeline = Pipeline(strTime, "RTSNet", "VanillaRNN")
RNN_Pipeline.setssModel(sys_model)
RNN_Pipeline.setModel(RNN_model)
RNN_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=50, learningRate=1e-3, weightDecay=1e-5)
RNN_Pipeline.model = torch.load('RNN/linear/2x2_rq020_T100.pt',map_location=dev)
if InitIsRandom:
   RNN_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, rnn=True, randomInit = True, cv_init=cv_init,train_init=train_init)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RNN_Pipeline.NNTest(sys_model, test_input, test_target, path_results, rnn=True,randomInit=True,test_init=test_init)
else:
   RNN_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, rnn=True)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RNN_Pipeline.NNTest(sys_model, test_input, test_target, path_results, rnn=True)
RNN_Pipeline.save()

### RTSNet with mismatched model #################################################################################
## Build Neural Network
# print("RTSNet with observation model mismatch")
# RTSNet_model = RTSNetNN()
# RTSNet_model.NNBuild(sys_model_partialh)
# ## Train Neural Network
# RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
# RTSNet_Pipeline.setssModel(sys_model_partialh)
# RTSNet_Pipeline.setModel(RTSNet_model)
# # RTSNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=30, learningRate=1E-3, weightDecay=1E-5)
# RTSNet_Pipeline.model = torch.load('RTSNet/new_architecture/linear/best-model_hrot10_linear2x2rq-1010T100.pt',map_location=dev)
# # [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_partialh, cv_input, cv_target, train_input, train_target, path_results)
# ## Test Neural Network
# [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,MSE_test_dB_std,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model_partialh, test_input, test_target, path_results)
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


# matlab_import = DataAnalysis()
# matlab_import.main(MSE_RTS_dB_avg)


#######################################
### Compare RTSNet and RTS Smoother ###
#######################################
# dataFolderName = 'Data' + '/'
# r2 = torch.tensor([10, 1, 0.1,0.01,0.001])
# r = torch.sqrt(r2)
# vdB = -20 # ratio v=q2/r2
# v = 10**(vdB/10)

# q2 = torch.mul(v,r2)
# q = torch.sqrt(q2)
# MSE_RTS_dB = torch.empty(size=[3,len(r)]).to(cuda0)
# dataFileName = ['data_2x2_r1q1_T50.pt','data_2x2_r2q2_T50.pt','data_2x2_r3q3_T50.pt','data_2x2_r4q4_T50.pt','data_2x2_r5q5_T50.pt']
# modelFolder = 'RTSNet' + '/'
# modelName = ['F10_2x2_r1q1','F10_2x2_r2q2','F10_2x2_r3q3','F10_2x2_r4q4','F10_2x2_r5q5']
# for rindex in range(0, len(r)):
#    print("1/r2 [dB]: ", 10 * torch.log10(1/r[rindex]**2))
#    print("1/q2 [dB]: ", 10 * torch.log10(1/q[rindex]**2))
#    SysModel_design = SystemModel(F, torch.squeeze(q[rindex]), H, torch.squeeze(r[rindex]), T, T_test,'linear', outlier_p=0) 
#    SysModel_design.InitSequence(m1_0, m2_0)
#    #Generate data
#    DataGen(SysModel_design, dataFolderName + dataFileName[rindex], T, T_test)
#    #Rotate model
#    # SysModel_rotate = SystemModel(F, torch.squeeze(q[rindex]), H_rotated, torch.squeeze(r[rindex]), T, T_test)
#    # SysModel_rotate.InitSequence(m1_0, m2_0)
#    #Load data
#    [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName[rindex])
#    #Evaluate KF with perfect SS knowledge
#    [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_RTS_dB[0,rindex]] = KFTest(SysModel_design, test_input, test_target)
#    #Evaluate RTS Smoother with perfect SS knowledge
#    [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB[1,rindex]] = S_Test(SysModel_design, test_input, test_target)
#    #Evaluate RTS Smoother with inaccurate SS knowledge
#    # [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB[1,rindex]] = S_Test(SysModel_rotate, test_input, test_target)
#    #Evaluate RTSNet with inaccurate SS knowledge
#    # RTSNet_Pipeline = Pipeline(strTime, "RTSNet", modelName[rindex])
#    # RTSNet_Pipeline.setssModel(SysModel_rotate)
#    # RTSNet_model = RTSNetNN()
#    # RTSNet_model.Build(SysModel_rotate)
#    # RTSNet_Pipeline.setModel(RTSNet_model)
#    # RTSNet_Pipeline.model = torch.load(modelFolder+"model_"+modelName[rindex]+".pt")
#    # RTSNet_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=30, learningRate=1E-2, weightDecay=5E-4)
#    # RTSNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
#    # RTSNet_Pipeline.NNTest(N_T, test_input, test_target)
#    # MSE_RTS_dB[2,rindex] = RTSNet_Pipeline.MSE_test_dB_avg
#    #Evaluate RTSNet with accurate SS knowledge
#    # RTSNet_Pipeline = Pipeline(strTime, "RTSNet", modelName[rindex])
#    # RTSNet_Pipeline.setssModel(SysModel_design)
#    # RTSNet_model = RTSNetNN()
#    # RTSNet_model.Build(SysModel_design)
#    # RTSNet_Pipeline.setModel(RTSNet_model)
#    # RTSNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=30, learningRate=1E-2, weightDecay=5E-4)
#    # RTSNet_Pipeline.NNTrain(N_E, train_input, train_target, N_CV, cv_input, cv_target)
#    # RTSNet_Pipeline.NNTest(N_T, test_input, test_target)
#    # MSE_RTS_dB[2,rindex] = RTSNet_Pipeline.MSE_test_dB_avg

# PlotfolderName = 'Graphs' + '/'
# PlotResultName = 'Opt_RTSandRTSNet_Compare' 
# torch.save(MSE_RTS_dB,PlotfolderName + PlotResultName)
# Plot = Plot(PlotfolderName, PlotResultName)
# print("Plot")
# Plot.rotate_RTS_Plot(r, MSE_RTS_dB)


