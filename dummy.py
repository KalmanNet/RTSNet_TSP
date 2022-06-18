from cmath import nan
import numpy as np
import torch
from Pipeline_ERTS import Pipeline_ERTS as Pipeline
from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
# from model import gt_data
from model import toSpherical, toCartesian


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
num_of_seq = 1
m = 3
n = 3
T = 2
T_test = 10
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

m1x_0 = torch.ones(m, 1) 
# P_0 = np.diag([1] * 3) * 0
print(m1x_0)
Sx = torch.squeeze(toSpherical(m1x_0))
print(Sx)
xafter = toCartesian(Sx)
print(xafter)