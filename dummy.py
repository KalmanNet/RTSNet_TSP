from cmath import nan
import torch
from filing_paths import path_model
# import sys
# sys.path.insert(1, path_model)
# from model import gt_data
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
cv_init = None
print(cv_init==None)