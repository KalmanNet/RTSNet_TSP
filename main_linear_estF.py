import torch
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import N_E, N_CV, N_T, F, H, F_rotated, H_rotated, T, T_test, m1_0, m2_0, m, n

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

from RTSNet.RTSNet_nn import RTSNetNN
from RTSNet.KalmanNet_nn import KalmanNetNN
from RNN.RNN_FWandBW import Vanilla_RNN

from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

from Plot import Plot_RTS as Plot

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
r2 = torch.tensor([10,1.,0.1,1e-2,1e-3])
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)

for index in range(0,len(r2)):

   print("1/r2 [dB]: ", 10 * torch.log10(1/r2[index]))
   print("1/q2 [dB]: ", 10 * torch.log10(1/q2[index]))

   # True model
   # r = torch.sqrt(r2[index])
   # q = torch.sqrt(q2[index])
   Q = q2[index] * torch.eye(m)
   R = r2[index] * torch.eye(n)
   sys_model = SystemModel(F_rotated, Q, H, R, T, T_test)
   sys_model.InitSequence(m1_0, m2_0)

   # Mismatched model
   sys_model_partialf = SystemModel(F, Q, H, R, T, T_test)
   sys_model_partialf.InitSequence(m1_0, m2_0)

   ###################################
   ### Data Loader (Generate Data) ###
   ###################################
   dataFolderName = 'Simulations/Linear_canonical/F_rotated' + '/'
   dataFileName = ['2x2_Frot10_rq-1010_T20.pt','2x2_Frot10_rq020_T20.pt','2x2_Frot10_rq1030_T20.pt','2x2_Frot10_rq2040_T20.pt','2x2_Frot10_rq3050_T20.pt']
   print("Start Data Gen")
   DataGen(sys_model, dataFolderName + dataFileName[index], T, T_test,randomInit=False)
   print("Data Load")
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName[index])
   print("trainset size:",train_target.size())
   print("cvset size:",cv_target.size())
   print("testset size:",test_target.size())

   ##############################
   ### Evaluate Kalman Filter ###
   ##############################
   print("Evaluate Kalman Filter True")
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, test_input, test_target)
   print("Evaluate Kalman Filter Partial")
   [MSE_KF_linear_arr_partialf, MSE_KF_linear_avg_partialf, MSE_KF_dB_avg_partialf] = KFTest(sys_model_partialf, test_input, test_target)



   # DatafolderName = 'Filters/Linear' + '/'
   # DataResultName = 'KF_HRotated'+ dataFileName[index]
   # torch.save({
   #             'MSE_KF_linear_arr': MSE_KF_linear_arr,
   #             'MSE_KF_dB_avg': MSE_KF_dB_avg,
   #             'MSE_KF_linear_arr_partialh': MSE_KF_linear_arr_partialh,
   #             'MSE_KF_dB_avg_partialh': MSE_KF_dB_avg_partialh,
   #             }, DatafolderName+DataResultName)

   #############################
   ### Evaluate RTS Smoother ###
   #############################
   print("Evaluate RTS Smoother True")
   [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out] = S_Test(sys_model, test_input, test_target)
   print("Evaluate RTS Smoother Partial")
   [MSE_RTS_linear_arr_partialf, MSE_RTS_linear_avg_partialf, MSE_RTS_dB_avg_partialf, RTS_partialF_out] = S_Test(sys_model_partialf, test_input, test_target)


   #######################
   ### RTSNet Pipeline ###
   #######################

   # RTSNet with full info
   # Build Neural Network
   # print("RTSNet with full model info")
   # RTSNet_model = RTSNetNN()
   # RTSNet_model.NNBuild(sys_model)
   # ## Train Neural Network
   # RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
   # RTSNet_Pipeline.setssModel(sys_model)
   # RTSNet_Pipeline.setModel(RTSNet_model)
   # RTSNet_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=30, learningRate=1E-3, weightDecay=1E-3)
   # [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
   # ## Test Neural Network
   # [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
   # RTSNet_Pipeline.save()
   # ##########################################################################################################################################

   # # RTSNet with mismatched model
   # # Build Neural Network
   # print("RTSNet with evolution model mismatch")
   # RTSNet_model = RTSNetNN()
   # RTSNet_model.NNBuild(sys_model_partialf)
   # ## Train Neural Network
   # RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
   # RTSNet_Pipeline.setssModel(sys_model_partialf)
   # RTSNet_Pipeline.setModel(RTSNet_model)
   # RTSNet_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=30, learningRate=1E-3, weightDecay=1E-3)
   # [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_partialf, cv_input, cv_target, train_input, train_target, path_results)
   # ## Test Neural Network
   # [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model_partialf, test_input, test_target, path_results)
   # RTSNet_Pipeline.save()
   # ##########################################################################################################################################

   print("RTSNet with estimated F")
   RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNetEstF_"+ dataFileName[index])
   print("True State Evolution Matrix F:", F_rotated)
   ### Least square estimation of F
   X = torch.squeeze(train_target[:,:,0])
   Y = torch.squeeze(train_target[:,:,1])
   for t in range(1,T-1):
      X_t = torch.squeeze(train_target[:,:,t])
      Y_t = torch.squeeze(train_target[:,:,t+1])
      X = torch.cat((X,X_t),0)
      Y = torch.cat((Y,Y_t),0)
   Y_1 = torch.unsqueeze(Y[:,0],1)
   Y_2 = torch.unsqueeze(Y[:,1],1)
   F_row1 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T,X)),X.T),Y_1)
   F_row2 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T,X)),X.T),Y_2)
   F_hat = torch.cat((F_row1.T,F_row2.T),0)
   print("Estimated State Evolution Matrix F:", F_hat)

   # Estimated model
   sys_model_estf = SystemModel(F_hat, Q, H, R, T, T_test)
   sys_model_estf.InitSequence(m1_0, m2_0)

   RTSNet_Pipeline.setssModel(sys_model_estf)
   RTSNet_model = RTSNetNN()
   RTSNet_model.NNBuild(sys_model_estf)
   RTSNet_Pipeline.setModel(RTSNet_model)
   
   RTSNet_Pipeline.setTrainingParams(n_Epochs=1000, n_Batch=30, learningRate=1E-3, weightDecay=1E-3)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_partialf, cv_input, cv_target, train_input, train_target, path_results)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model_partialf, test_input, test_target, path_results)
   RTSNet_Pipeline.save()

   ###################
   ### Vanilla RNN ###
   ###################
   ### Vanilla RNN
   # Build RNN
   print("Vanilla RNN")
   RNN_model = Vanilla_RNN()
   RNN_model.Build(sys_model,fully_agnostic = False)
   print("Number of trainable parameters for RNN:",sum(p.numel() for p in RNN_model.parameters() if p.requires_grad))
   RNN_Pipeline = Pipeline(strTime, "RTSNet", "VanillaRNN")
   RNN_Pipeline.setssModel(sys_model)
   RNN_Pipeline.setModel(RNN_model)
   RNN_Pipeline.setTrainingParams(n_Epochs=2000, n_Batch=50, learningRate=1e-3, weightDecay=1e-3)
   # RNN_Pipeline.model = torch.load('RNN/checkpoints/linear/2x2_rq020_T100.pt')  
   RNN_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, rnn=True)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RNN_Pipeline.NNTest(sys_model, test_input, test_target, path_results, rnn=True)
   RNN_Pipeline.save()

   ##########################################################################################################################################
   ### RNN with mismatched model
   # Build RNN
   print("Vanilla RNN with mismatched F")
   RNN_model = Vanilla_RNN()
   RNN_model.Build(sys_model_partialf,fully_agnostic = False)
   ## Train Neural Network
   print("Number of trainable parameters for RNN:",sum(p.numel() for p in RNN_model.parameters() if p.requires_grad))
   RNN_Pipeline = Pipeline(strTime, "RTSNet", "VanillaRNN")
   RNN_Pipeline.setssModel(sys_model_partialf)
   RNN_Pipeline.setModel(RNN_model)
   RNN_Pipeline.setTrainingParams(n_Epochs=2000, n_Batch=50, learningRate=1e-3, weightDecay=1e-3)

   RNN_Pipeline.NNTrain(sys_model_partialf, cv_input, cv_target, train_input, train_target, path_results, rnn=True)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RNN_Pipeline.NNTest(sys_model_partialf, test_input, test_target, path_results, rnn=True)
   RNN_Pipeline.save()

