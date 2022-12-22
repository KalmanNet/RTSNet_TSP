import numpy as np
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import pickle
import torch.nn as nn
import EKF_test
from Extended_RTS_Smoother_test import S_Test
from Extended_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Pipeline_EKF import Pipeline_EKF
from Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Pipeline_concat_models import Pipeline_twoRTSNets

from KalmanNet_nn import KalmanNetNN
from RTSNet_nn import RTSNetNN
from RNN_FWandBW import Vanilla_RNN

from PF_test import PFTest
from ParticleSmoother_test import PSTest

from Plot import Plot_extended as Plot

from datetime import datetime

import wandb

from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from parameters import m1x_0, m2x_0, m, n,delta_t_gen,delta_t
from model import f, h, fInacc, hRotate, fRotate

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

######################################
###  Compare EKF, RTS and RTSNet   ###
######################################
offset = 0
chop = False
sequential_training = False
secondpass = False
path_results = 'ERTSNet/'
DatafolderName = 'Simulations/Lorenz_Atractor/data/decimation/'
DatagenfolderName = 'Simulations/Lorenz_Atractor/data/'
DatafileName = 'decimated_r0_Ttest3000.pt'
Datasecondpass = 'r0_outputoffirstpass.pt'
data_gen = 'data_gen.pt'
data_gen_file = torch.load(DatagenfolderName+data_gen, map_location=dev)
[true_sequence] = data_gen_file['All Data']

r = torch.tensor([1]) ###[1/5.9566]
lambda_q = torch.tensor([0.3873])
T = 3000
T_test = 3000
traj_resultName = ['traj_lor_dec_RTSNetJ2_r0_2pass.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
# EKFResultName = 'EKF_obsmis_rq1030_T2000_NT100' 

wandb.init(project="RTSNet_Lorenz")


for rindex in range(0, len(r)):
   print("1/r2 [dB]: ", 10 * torch.log10(1/r[rindex]**2))
   print("Search 1/q2 [dB]: ", 10 * torch.log10(1/lambda_q[rindex]**2))
   # Q_mod = (lambda_q[rindex]**2) * torch.eye(m)
   # R_mod = (r[rindex]**2) * torch.eye(n)
   # True Model
   sys_model_true = SystemModel(f, lambda_q[rindex], h, r[rindex], T, T_test,m,n)
   sys_model_true.InitSequence(m1x_0, m2x_0)

   # Model with partial Info
   sys_model = SystemModel(fInacc, lambda_q[rindex], h, r[rindex], T, T_test,m,n)
   sys_model.InitSequence(m1x_0, m2x_0)

   ##############################################
   ### Generate and load data Decimation case ###
   ##############################################
   ########################
   # print("Data Gen")
   # ########################
   # [test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_T, h, r[rindex], offset) 
   # [train_target_long, train_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_E, h, r[rindex], offset)
   # [cv_target_long, cv_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, N_CV, h, r[rindex], offset)
   # if chop:
   #    print("chop training data")  
   #    [train_target, train_input, train_init] = Short_Traj_Split(train_target_long, train_input_long, T)
   # else:
   #    print("no chopping") 
   #    train_target = train_target_long
   #    train_input = train_input_long
   # # Save dataset
   # if(chop):
   #    torch.save([train_input, train_target, train_init, cv_input_long, cv_target_long, test_input, test_target], DatafolderName+DatafileName)
   # else:
   #    torch.save([train_input, train_target, cv_input_long, cv_target_long, test_input, test_target], DatafolderName+DatafileName)

   #########################
   print("Data Load")
   #########################
   [train_input, train_target, cv_input_long, cv_target_long, test_input, test_target] = torch.load(DatafolderName+DatafileName,map_location=dev)  
   
   if(chop):
      print("chop training data")  
      [train_target, train_input, train_init] = Short_Traj_Split(train_target, train_input, T)
   
   if(secondpass):
      traj = torch.load(DatafolderName+Datasecondpass,map_location=dev) 
      train_input = traj['RTSNet']
      cv_input_long = train_input[0:5]
      test_input = train_input[5:15]

      train_input = train_input[15:]
      train_target = train_target[15:]
   print("testset size:",test_target.size())
   print("trainset size:",train_target.size())
   print("cvset size:",cv_target_long.size())
   
   ################################
   ### Load data from Welling's ###
   ################################
   # compact_path = "ERTSNet/new_arch_LA/decimation/Welling_Compare/lorenz_trainset300k.pickle"
   # with open(compact_path, 'rb') as f:
   #    data = pickle.load(f)
   # testdata = [data[0][0:T_test], data[1][0:T_test]]
   # states, meas = testdata
   # test_target =  torch.from_numpy(np.asarray(states, dtype=np.float32).transpose(1,0)).unsqueeze(0)
   # test_input = torch.from_numpy(np.asarray(meas, dtype=np.float32).transpose(1,0)).unsqueeze(0)
   # print("testset size:",test_target.size())
   # traindata = [data[0][T_test:(T_test+T*N_E)], data[1][T_test:(T_test+T*N_E)]]
   # states, meas = traindata
   # train_target =  torch.from_numpy(np.asarray(states, dtype=np.float32).transpose(1,0)).unsqueeze(0)
   # train_input = torch.from_numpy(np.asarray(meas, dtype=np.float32).transpose(1,0)).unsqueeze(0)
   # [train_target, train_input, train_init] = Short_Traj_Split(train_target, train_input, T)
   # cvdata = [data[0][(T_test+T*N_E):], data[1][(T_test+T*N_E):]]
   # states, meas = cvdata
   # cv_target_long =  torch.from_numpy(np.asarray(states, dtype=np.float32).transpose(1,0)).unsqueeze(0)
   # cv_input_long = torch.from_numpy(np.asarray(meas, dtype=np.float32).transpose(1,0)).unsqueeze(0)
   # [cv_target_long, cv_input_long, cv_init] = Short_Traj_Split(cv_target_long, cv_input_long, T)
   # print("trainset size:",train_target.size())
   # print("cvset size:",cv_target_long.size())

   ########################################
   ### Evaluate Observation Noise Floor ###
   ########################################
   N_T = len(test_input)
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

   print("Observation Noise Floor(test dataset) - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
   print("Observation Noise Floor(test dataset) - STD:", obs_std_dB, "[dB]")
   ###################################################
   N_E = len(train_input)
   MSE_obs_linear_arr = torch.empty(N_E)# MSE [Linear]
   for j in range(0, N_E):        
    MSE_obs_linear_arr[j] = loss_obs(train_input[j], train_target[j]).item()
   MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
   MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

   # Standard deviation
   MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

   # Confidence interval
   obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

   print("Observation Noise Floor(train dataset) - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
   print("Observation Noise Floor(train dataset) - STD:", obs_std_dB, "[dB]")

   ######################################
   ### Evaluate Filters and Smoothers ###
   ######################################
   ### Particle filter
   # print("Start PF test J=5")
   # [MSE_PF_linear_arr, MSE_PF_linear_avg, MSE_PF_dB_avg, PF_out, t_PF] = PFTest(sys_model_true, test_input, test_target, init_cond=None)
   # print("Start PF test J=2")
   # [MSE_PF_linear_arr_partial, MSE_PF_linear_avg_partial, MSE_PF_dB_avg_partial, PF_out_partial, t_PF] = PFTest(sys_model, test_input, test_target, init_cond=None)

   
   ### EKF
   # print("Start EKF test J=5")
   # [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKF_test.EKFTest(sys_model_true, test_input, test_target)
   # print("Start EKF test J=2")
   # [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKF_test.EKFTest(sys_model, test_input, test_target)

   # # [MSE_EKF_dB_avg, trace_dB_avg] = EKF_test.EKFTest_evol(sys_model, test_input, test_target)

   ### Particle Smoother
   # print("Start PS test J=5")
   # [MSE_PS_linear_arr, MSE_PS_linear_avg, MSE_PS_dB_avg, PS_out, t_PS] = PSTest(sys_model_true, test_input, test_target,N_FWParticles=100, M_BWTrajs=10, init_cond=None)
   # print("Start PS test J=2")
   # [MSE_PS_linear_arr_partial, MSE_PS_linear_avg_partial, MSE_PS_dB_avg_partial, PS_out_partial, t_PS] = PSTest(sys_model, test_input, test_target,N_FWParticles=100, M_BWTrajs=10, init_cond=None)

   ### MB Extended RTS
   # print("Start RTS test J=5")
   # [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out] = S_Test(sys_model_true, test_input, test_target)
   # print("Start RTS test J=2")
   # [MSE_ERTS_linear_arr_partial, MSE_ERTS_linear_avg_partial, MSE_ERTS_dB_avg_partial, ERTS_out_partial] = S_Test(sys_model, test_input, test_target)
   
   # KNet with model mismatch
   # ## Build Neural Network
#    KNet_model = KalmanNetNN()
#    KNet_model.NNBuild(sys_model)
#    ## Train Neural Network
#    KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
#    KNet_Pipeline.setModel(KNet_model)
#    KNet_Pipeline.setssModel(sys_model)
   
#   #  KNet_Pipeline.setTrainingParams(n_Epochs=100, n_Batch=10, learningRate=1e-3, weightDecay=1e-6)
#   #  [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, cv_input_long, cv_target_long, train_input, train_target, path_results, sequential_training)
#    # Test Neural Network
#    KNet_Pipeline.model = torch.load('ERTSNet/model_KNetNew_Dec_r0_noTransfer.pt',map_location=dev)
#    NumofParameter = sum(p.numel() for p in KNet_Pipeline.model.parameters() if p.requires_grad)
#    print("Number of parameters for KNet: ",NumofParameter)
#    [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, knet_out] = KNet_Pipeline.NNTest(N_T, test_input, test_target)
   # # Print MSE Cross Validation
   # print("MSE Test:", MSE_test_dB_avg, "[dB]")
   # [MSE_knet_test_dB_avg,trace_knet_dB_avg] = KNet_Pipeline.NNTest_evol(sys_model, test_input, test_target, path_results)
   # PlotfolderName = path_results
   # MSE_resultName = "error_evol"
   # error_evol = torch.load(PlotfolderName+MSE_resultName, map_location=dev)
   # print(error_evol.keys())
   # MSE_knet_test_dB_avg = error_evol['MSE_knet']
   # trace_knet_dB_avg = error_evol['trace_knet']
   # MSE_EKF_dB_avg = error_evol['MSE_EKF']
   # trace_dB_avg = error_evol['trace_EKF']
   # Plot = Plot(PlotfolderName, modelName='KNet')
   # print("Plot")
   # Plot.error_evolution(MSE_knet_test_dB_avg,trace_knet_dB_avg,MSE_EKF_dB_avg, trace_dB_avg)

   ######################
   ### Vanilla RNN ######
   ######################
   # Build RNN
   # print("Vanilla RNN with mismatched f")
   # RNN_model = Vanilla_RNN()
   # RNN_model.Build(sys_model)
   # print("Number of trainable parameters for RNN:",sum(p.numel() for p in RNN_model.parameters() if p.requires_grad))
   # RNN_Pipeline = Pipeline(strTime, "RTSNet", "VanillaRNN")
   # RNN_Pipeline.setssModel(sys_model)
   # RNN_Pipeline.setModel(RNN_model)
   # RNN_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=50, learningRate=1e-3, weightDecay=1e-5)
   # if(chop):
   #    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RNN_Pipeline.NNTrain(sys_model, cv_input_long, cv_target_long, train_input, train_target, path_results,randomInit=True,train_init=train_init)
   # else:
   #    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RNN_Pipeline.NNTrain(sys_model, cv_input_long, cv_target_long, train_input, train_target, path_results)
   # ## Test Neural Network
   # [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rnn_out,RunTime] = RNN_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
   # RNN_Pipeline.save()


   ###################################
   ### RTSNet with model mismatch  ###
   ###################################
   # ## Build Neural Network
   print("RTSNet with model mismatch")
   RTSNet_model = RTSNetNN()
   RTSNet_model.NNBuild(sys_model)
   ## Train Neural Network
   RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
   RTSNet_Pipeline.setModel(RTSNet_model)
   RTSNet_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=1, learningRate=1e-3, weightDecay=1e-4)
   NumofParameter = RTSNet_Pipeline.count_parameters()
   print("Number of parameters for RTSNet: ",NumofParameter)

   ### Optional: record parameters on wandb
   wandb.log({
  "learning_rate": RTSNet_Pipeline.learningRate,
  "batch_size": RTSNet_Pipeline.N_B,
  "weight_decay": RTSNet_Pipeline.weightDecay})
   ###

   if(chop):
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input_long, cv_target_long, train_input, train_target, path_results,randomInit=True,train_init=train_init)
   else:
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input_long, cv_target_long, train_input, train_target, path_results)
   ## Test Neural Network
   # RTSNet_Pipeline.model = torch.load('ERTSNet/new_arch_LA/decimation/model/best-model_r0_J2_NE1000_MSE-15.5.pt',map_location=dev)
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
   

   # ## Save histogram
   # MSE_ResultName = 'Partial_MSE_KNet' 
   # torch.save(MSE_test_dB_avg,trajfolderName + MSE_ResultName)


   ###############################################
   ### Concat two RTSNets with model mismatch  ###
   ###############################################
   ## load trained Neural Network
#    print("RTSNet with model mismatch")
#    RTSNet_model1 = torch.load('ERTSNet/new_arch_LA/decimation/model/best-model_r0_J2_NE1000_MSE-15.5.pt',map_location=dev)
#    RTSNet_model2 = torch.load('ERTSNet/new_arch_LA/decimation/model/second-pass-of-15.5.pt',map_location=dev)
#    ## Train Neural Network
#    RTSNet_Pipeline = Pipeline_twoRTSNets(strTime, "RTSNet", "RTSNet")
#    RTSNet_Pipeline.setModel(RTSNet_model1, RTSNet_model2)
#    NumofParameter = RTSNet_Pipeline.count_parameters()
#    print("Number of parameters for RTSNet: ",NumofParameter)
#    ## Test Neural Network
#    [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out_2pass,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)

  # Save trajectories
   # trajfolderName = 'ERTSNet' + '/'
   # DataResultName = 'traj_lor_dec_PS'
   # target_sample = torch.reshape(test_target[0,:,:],[1,m,T_test])
   # input_sample = torch.reshape(test_input[0,:,:],[1,n,T_test])
   # torch.save({#'PF J=5':PF_out,
   #             #'PF J=2':PF_out_partial,
   #             # 'True':target_sample,
   #             # 'Observation':input_sample,
   #             # 'EKF J=5':EKF_out,
   #             # 'EKF J=2':EKF_out_partial,
   #             # 'RTS J=5':ERTS_out,
   #             # 'RTS J=2':ERTS_out_partial,
   #             'PS J=5':PS_out,
   #             'PS J=2':PS_out_partial,
   #             # 'RTSNet': rtsnet_out,
   #             # 'RTSNet_2pass': rtsnet_out_2pass,
   #             # 'RNN J=2': rnn_out,
   #             }, trajfolderName+DataResultName)

#    titles = ["True Trajectory","Observation","RTSNet",]#, "Observation", "EKF J=2","EKF J=2 with optimal q"]
#    input = [target_sample,input_sample,rtsnet_out]#,EKF_sample,EKF_partial_sample,EKF_partialoptq_sample]
#    Net_Plot = Plot(trajfolderName,DataResultName)
#    Net_Plot.plotTrajectories(input,3, titles,trajfolderName+"RTSNet_twopasses.png")

# Close wandb run 
wandb.finish()  





