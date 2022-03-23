import torch
import torch.nn as nn
import time
from Linear_KF import KalmanFilter

def KFTest(SysModel, test_input, test_target, kitti=False):

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    N_T = len(test_input)
    MSE_KF_linear_arr = torch.empty(N_T)
    start = time.time()
    KF = KalmanFilter(SysModel)
    if(~kitti):
        KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)       
    j=0
    mask = torch.tensor([True,True,True,False,False,False])# for kitti
    for sequence_target,sequence_input in zip(test_target,test_input):
        if(kitti):
            KF.InitSequence(sequence_target[:,0], SysModel.m2x_0)   
        KF.GenerateSequence(sequence_input, sequence_input.size()[-1])
        if(kitti):
            MSE_KF_linear_arr[j] = loss_fn(KF.x[mask], sequence_target[mask]).item()     
        else:
            MSE_KF_linear_arr[j] = loss_fn(KF.x, sequence_target).item()
        j = j+1
        #MSE_KF_linear_arr[j] = loss_fn(test_input[j, :, :], test_target[j, :, :]).item()
    end = time.time()
    t = end - start

    # Average
    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)

    # Standard deviation
    MSE_KF_dB_std = torch.std(MSE_KF_linear_arr, unbiased=True)
    MSE_KF_dB_std = 10 * torch.log10(MSE_KF_dB_std)

    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    print("Kalman Filter - MSE STD:", MSE_KF_dB_std, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg]



