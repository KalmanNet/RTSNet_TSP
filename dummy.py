from cmath import nan
import numpy as np
import torch
from Pipeline_ERTS import Pipeline_ERTS as Pipeline
from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
# from model import gt_data
from model import h_nonlinear,hRotate, getJacobian, toSpherical, toCartesian
from parameters import H_mod

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

# mask = torch.tensor([True,True,True,False,False,False])
# for sequence in gt_data:
#     print(sequence[:,0][mask])

############################################################
num_of_seq = 200
m = 3
n = 3
T = 20
T_test = 20
r = 1
q = 1
print("1/r2 [dB]: ", 10 * np.log10(1/r**2))
print("1/q2 [dB]: ", 10 * np.log10(1/q**2))
# # loss_fn = torch.nn.MSELoss(reduction='mean')
# # input = torch.empty(num_of_seq,m, T)
# # x_out_training = torch.ones_like(input)
# # train_target = torch.zeros_like(input)
# # LOSS = loss_fn(x_out_training, train_target)
# # print(LOSS)

# input = torch.empty(num_of_seq,m)
# init = torch.ones_like(input)

# print(torch.unsqueeze(init[1,:],1).size())
######################################################
# a = torch.tensor([float('nan')])
# print(~torch.isnan(a))
######################################################
# a = torch.tensor([0])
# print(a.item())
######################################################
# data_target = torch.rand(num_of_seq,m,T_test)
# data_input = torch.rand(num_of_seq,n,T_test)
# print("orginal:",data_target)
# data_target = list(torch.split(data_target,T+1,2))# +1 to reserve for init
# print("after split length:",len(data_target))
# print("after split:",data_target)
# data_input = list(torch.split(data_input,T+1,2))
# data_target.pop()# Remove the last one which may not fullfill length T
# data_input.pop()
# print("after pop:",len(data_target))
# data_target = torch.squeeze(torch.cat(list(data_target), dim=0))
# data_input = torch.squeeze(torch.cat(list(data_input), dim=0))
# # Split init
# target = data_target[:,:,1:]
# input = data_input[:,:,1:]
# init = data_target[:,:,0]
# print("output target:",target.size())
# print("output input:",input.size())
# print("output init:",init.size())
############################################################
# cv_init = None
# train_init = None
# dataFolderName = 'Simulations/Linear_canonical/Scaling_to_large_models' + '/'
# dataFileName = '40x40_rq020_T20.pt'
# [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(dataFolderName+dataFileName,map_location=dev)
# train_input = train_input.numpy().astype(np.float64)
# print(type(train_input[0,0,0]))

# state = np.array([[1, 2,3], [4,5,6]])
# print(state.shape)
# position = state[:, [0, 2]]
# print(position)

# m1x_0 = torch.ones(m, 1) 
# # P_0 = np.diag([1] * 3) * 0
# print(m1x_0)
# Sx = torch.squeeze(toSpherical(m1x_0))
# print(Sx)
# xafter = toCartesian(Sx)
# print(xafter)

# m1x_0 = torch.ones(3, 2) 
# m1x_0[2,1] = 0
# # P_0 = np.diag([1] * 3) * 0
# print(m1x_0)
# Sx = m1x_0[:,-1]
# print(Sx)

loss_fn = torch.nn.MSELoss(reduction='mean')
## Load x and y linear H
# DatafolderName = 'Simulations/Lorenz_Atractor/data/T100_Hrot1' + '/'
# dataFileName = ['data_lor_v20_rq020_T100.pt']
# h_function = hRotate
## Load x and y NL H
DatafolderName = 'Simulations/Lorenz_Atractor/data/T20_hNL' + '/'
dataFileName = ['data_lor_v0_rq00_T20.pt']
h_function = h_nonlinear
print("Data Load")
print(dataFileName[0])
[_,_,_,_, test_input, test_target] =  torch.load(DatafolderName + dataFileName[0],map_location=dev)
y = test_input
x = test_target
print("x size:",x.size())
print("y size:",y.size())
# print(H_mod)
# Htest = torch.zeros(n,n)
# Htest[0] = torch.ones(1,n)
# for i in range(n):
#     Htest[i,n-1-i] = 1
# Htest = torch.ones(n,n)
# print(Htest)
# def h_test(x):
#    return torch.squeeze(torch.matmul(Htest,x))
# def h_nonlinear(x):
#    return torch.squeeze(torch.sin(x))
# for i in range(num_of_seq):
#    for t in range(T):
#       y[i,:,t] = h_nonlinear(x[i,:,t]) 
#       mean = torch.zeros([n])
#       er = torch.normal(mean, r)
#       y[i,:,t] = torch.add(y[i,:,t],er)
# MSE_linear_arr = torch.empty(num_of_seq)
# for j in range(num_of_seq):
#    MSE_linear_arr[j] = loss_fn(y[j,:,:], x[j,:,:]).item()
# MSE_linear_avg = torch.mean(MSE_linear_arr)
# MSE_dB_avg = 10 * torch.log10(MSE_linear_avg)

# print("Obs LOSS:", MSE_dB_avg, "[dB]")

### LMMSE method
# x_out = torch.empty((num_of_seq,m,T))
# for i in range(num_of_seq):
#    for t in range(T):
#       ### Compute Jac at prior_x = true_x + q
#       mean = torch.zeros([m])
#       eq = torch.normal(mean, q)
#       priorx = torch.add(x[i,:,t],eq)
#       Ht = getJacobian(priorx, h_function)
#       # print(Ht)
#       Ht_T = torch.transpose(Ht,0,1)
#       H_lmmse = torch.matmul(torch.inverse(torch.matmul(Ht_T,Ht)),Ht_T)
#       x_out[i,:,t] = torch.squeeze(torch.matmul(H_lmmse, y[i,:,t]))

### Invert h method
# x_out = torch.empty((num_of_seq,m,T))
# for i in range(num_of_seq):
#    for t in range(T):
#       x_out[i,:,t] = toCartesian(y[i,:,t])

### Linear opt RTS


# MSE_linear_arr = torch.empty(num_of_seq)
# for j in range(num_of_seq):
#    MSE_linear_arr[j] = loss_fn(x_out[j,:,:], x[j,:,:]).item()
# MSE_linear_avg = torch.mean(MSE_linear_arr)
# MSE_dB_avg = 10 * torch.log10(MSE_linear_avg)

#  # Standard deviation
# MSE_linear_std = torch.std(MSE_linear_arr, unbiased=True)

# # Confidence interval
# std_dB = 10 * torch.log10(MSE_linear_std + MSE_linear_avg) - MSE_dB_avg

# print("MSE LOSS:", MSE_dB_avg, "[dB]")
# print("STD:", std_dB, "[dB]")

#######################
# torch.equal
Q1 = torch.zeros(3,3)
Q2 = torch.ones(3,3)
panding1 = torch.equal(Q1,torch.zeros(3,3))
panding2 = torch.equal(Q2,torch.zeros(3,3))
print(panding1,panding2)