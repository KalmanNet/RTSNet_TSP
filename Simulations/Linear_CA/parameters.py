import torch
import math

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

####################################
### Generative Parameters For CA ###
####################################
m = 3 # dim of state for CA model
m_cv = 2 # dim of state for CV model
n = 1 # dim of observation
std = 1
m1x_0 = torch.zeros(m) # Initial State
m1x_0_cv = torch.zeros(m_cv) # Initial State for CV
m2x_0 = std * std * torch.eye(m) # Initial Covariance
m2x_0_cv = std * std * torch.eye(m_cv) # Initial Covariance for CV

delta_t_gen =  1e-2
# Length of Time Series Sequence
T_gen = 100
T_test_gen = 100

#########################################################
### state evolution matrix F and observation matrix H ###
#########################################################

F_gen = torch.tensor([[1, delta_t_gen,0.5*delta_t_gen**2],
                  [0,       1,       delta_t_gen],
                  [0,       0,         1]]).float()

F_CV = torch.tensor([[1, delta_t_gen],
                     [0,           1]]).float()              

# Full observation
H_identity = torch.eye(3)
# Observe only the postion
H_onlyPos = torch.tensor([[1, 0, 0]]).float()

###############################################
### process noise Q and observation noise R ###
###############################################
# Noise Parameters
r2 = torch.tensor([1]).float()
q2 = torch.tensor([1]).float()

Q_gen = q2 * torch.tensor([[1/20*delta_t_gen**5, 1/8*delta_t_gen**4,1/6*delta_t_gen**3],
                           [ 1/8*delta_t_gen**4, 1/3*delta_t_gen**3,1/2*delta_t_gen**2],
                           [ 1/6*delta_t_gen**3, 1/2*delta_t_gen**2,       delta_t_gen]]).float()

Q_CV = q2 * torch.tensor([[1/3*delta_t_gen**3, 1/2*delta_t_gen**2],
                          [1/2*delta_t_gen**2,        delta_t_gen]]).float()  

R_3 = r2 * torch.eye(3)
R_2 = r2 * torch.eye(2)

R_onlyPos = r2

##########################
### Further Decimation ###
##########################
delta_t = 1e-1
# Decimation ratio
ratio = delta_t_gen/delta_t
T = int(T_gen*ratio)
T_test = int(T_test_gen*ratio)

F = torch.tensor([[1, delta_t,0.5*delta_t**2],
                  [0,       1,       delta_t],
                  [0,       0,         1]]).float()

q2_tuned = torch.tensor([0.1]) # can be tuned
Q =     q2_tuned * torch.tensor([[1/20*delta_t**5, 1/8*delta_t**4,1/6*delta_t**3],
                           [ 1/8*delta_t**4, 1/6*delta_t**3,1/2*delta_t**2],
                           [ 1/6*delta_t**3, 1/2*delta_t**2,       delta_t]]).float()