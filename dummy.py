import torch
from filing_paths import path_model
# import sys
# sys.path.insert(1, path_model)
# from model import gt_data
# mask = torch.tensor([True,True,True,False,False,False])
# for sequence in gt_data:
#     print(sequence[:,0][mask])


num_of_seq = 200
m = 3
T = 100
# loss_fn = torch.nn.MSELoss(reduction='mean')
# input = torch.empty(num_of_seq,m, T)
# x_out_training = torch.ones_like(input)
# train_target = torch.zeros_like(input)
# LOSS = loss_fn(x_out_training, train_target)
# print(LOSS)

input = torch.empty(num_of_seq,m)
init = torch.ones_like(input)

print(torch.unsqueeze(init[1,:],1).size())