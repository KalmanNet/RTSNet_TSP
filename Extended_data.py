from signal import Handlers
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import math
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

########################
### Basic Parameters ###
########################

# Number of Training Examples
N_E = 1000

# Number of Cross Validation Examples
N_CV = 5

N_T = 10

# Sequence Length for Linear Case
T = 20
T_test = 20


############
## 2 x 2 ###
############
# m = 2
# n = 2
# # F in canonical form
# F = torch.eye(m)
# F[0] = torch.ones(1,m) 

# # H = I
# H = torch.eye(2)

# m1_0 = torch.tensor([[0.0], [0.0]]).to(dev)
# # m1x_0_design = torch.tensor([[10.0], [-10.0]])
# m2_0 = 0 * 0 * torch.eye(m).to(dev)


################################
### 5 x 5, 10 x 10 and so on ###
################################
m = 40
n = 40
# F in canonical form
F = torch.eye(m)
F[0] = torch.ones(1,m) 

# H in reverse canonical form
H = torch.zeros(n,n)
H[0] = torch.ones(1,n)
for i in range(n):
    H[i,n-1-i] = 1

m1_0 = torch.zeros(m, 1).to(dev)
# m1x_0_design = torch.tensor([[1.0], [-1.0], [2.0], [-2.0], [0.0]]).to(dev)
m2_0 = 0 * 0 * torch.eye(m).to(dev)



# Inaccurate model knowledge based on matrix rotation
F_rotated = torch.zeros_like(F)
H_rotated = torch.zeros_like(H)
if(m==2):
    alpha_degree = 10
    rotate_alpha = torch.tensor([alpha_degree/180*torch.pi]).to(dev)
    cos_alpha = torch.cos(rotate_alpha)
    sin_alpha = torch.sin(rotate_alpha)
    rotate_matrix = torch.tensor([[cos_alpha, -sin_alpha],
                                [sin_alpha, cos_alpha]]).to(dev)
    # print(rotate_matrix)
    F_rotated = torch.mm(F,rotate_matrix) #inaccurate process model
    H_rotated = torch.mm(H,rotate_matrix) #inaccurate observation model

def DataGen_True(SysModel_data, fileName, T):

    SysModel_data.GenerateBatch(1, T, randomInit=False)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target

    # torch.save({"True Traj":[test_target],
    #             "Obs":[test_input]},fileName)
    torch.save([test_input, test_target], fileName)

def DataGen(SysModel_data, fileName, T, T_test,randomInit=False,randomLength=False):

    ##################################
    ### Generate Training Sequence ###
    ##################################
    SysModel_data.GenerateBatch(N_E, T, randomInit=randomInit,randomLength=randomLength)
    training_input = SysModel_data.Input
    training_target = SysModel_data.Target
    if(randomInit):
        training_init = SysModel_data.m1x_0_rand

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    SysModel_data.GenerateBatch(N_CV, T, randomInit=randomInit,randomLength=randomLength)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target
    if(randomInit):
        cv_init = SysModel_data.m1x_0_rand

    ##############################
    ### Generate Test Sequence ###
    ##############################
    SysModel_data.GenerateBatch(N_T, T_test, randomInit=randomInit,randomLength=randomLength)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target
    if(randomInit):
        test_init = SysModel_data.m1x_0_rand

    #################
    ### Save Data ###
    #################
    if(randomInit):
        torch.save([training_input, training_target, training_init, cv_input, cv_target, cv_init, test_input, test_target, test_init], fileName)
    else:
        torch.save([training_input, training_target, cv_input, cv_target, test_input, test_target], fileName)

def DataLoader(fileName):

    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.load(fileName,map_location=dev)
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]

def DataLoader_GPU(fileName):
    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.utils.data.DataLoader(torch.load(fileName,map_location=dev),pin_memory = False)
    training_input = training_input.squeeze().to(dev)
    training_target = training_target.squeeze().to(dev)
    cv_input = cv_input.squeeze().to(dev)
    cv_target =cv_target.squeeze().to(dev)
    test_input = test_input.squeeze().to(dev)
    test_target = test_target.squeeze().to(dev)
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]

def DecimateData(all_tensors, t_gen,t_mod, offset=0):
    
    # ratio: defines the relation between the sampling time of the true process and of the model (has to be an integer)
    ratio = round(t_mod/t_gen)

    i = 0
    all_tensors_out = all_tensors
    for tensor in all_tensors:
        tensor = tensor[:,(0+offset)::ratio]
        if(i==0):
            all_tensors_out = torch.cat([tensor], dim=0).view(1,all_tensors.size()[1],-1)
        else:
            all_tensors_out = torch.cat([all_tensors_out,tensor], dim=0)
        i += 1

    return all_tensors_out

def Decimate_and_perturbate_Data(true_process, delta_t, delta_t_mod, N_examples, h, lambda_r, offset=0):
    
    # Decimate high resolution process
    decimated_process = DecimateData(true_process, delta_t, delta_t_mod, offset)

    noise_free_obs = getObs(decimated_process,h)

    # Replicate for computation purposes
    decimated_process = torch.cat(int(N_examples)*[decimated_process])
    noise_free_obs = torch.cat(int(N_examples)*[noise_free_obs])


    # Observations; additive Gaussian Noise
    observations = noise_free_obs + torch.randn_like(decimated_process) * lambda_r

    return [decimated_process, observations]

def getObs(sequences, h):
    i = 0
    sequences_out = torch.zeros_like(sequences)
    # sequences_out = torch.zeros_like(sequences)
    for sequence in sequences:
        for t in range(sequence.size()[1]):
            sequences_out[i,:,t] = h(sequence[:,t])
    i = i+1

    return sequences_out


def Short_Traj_Split(data_target, data_input, T):### Random Init is automatically incorporated
    data_target = list(torch.split(data_target,T+1,2)) # +1 to reserve for init
    data_input = list(torch.split(data_input,T+1,2)) # +1 to reserve for init

    data_target.pop()# Remove the last one which may not fullfill length T
    data_input.pop()# Remove the last one which may not fullfill length T

    data_target = torch.squeeze(torch.cat(list(data_target), dim=0))#Back to tensor and concat together
    data_input = torch.squeeze(torch.cat(list(data_input), dim=0))#Back to tensor and concat together
    # Split out init
    target = data_target[:,:,1:]
    input = data_input[:,:,1:]
    init = data_target[:,:,0]
    return [target, input, init]
