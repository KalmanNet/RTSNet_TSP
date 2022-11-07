import torch
import math

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

#########################
### Design Parameters ###
#########################
m = 2
n = 2

m1x_0 = torch.FloatTensor([1,0]) 
m1x_0_design_test = torch.ones(m, 1)
variance = 0
m2x_0 = variance * torch.eye(m)

##########################################
### Generative Parameters For Pendulum ###
##########################################
g = 9.81 # Gravitational Acceleration
L = 1 # Radius of pendulum

delta_t_gen =  1e-5
delta_t = 0.02

# Decimation ratio
ratio = delta_t_gen/delta_t

# Length of Time Series Sequence
T = 100
T_test = 100
T_gen = math.ceil(T_test / ratio)
# T_test_gen = math.ceil(T_test / ratio)

H_id = torch.eye(m)

H_id_inv = H_id

# Observe only the postion
H_onlyPos = torch.tensor([[1, 0]]).float()

# Noise Parameters
r2 = torch.tensor([1]).float()
q2 = torch.tensor([1]).float()

# Noise Matrices
Q_gen = q2 * torch.tensor([[1/3*delta_t_gen**3, 1/2*delta_t_gen**2],
                          [1/2*delta_t_gen**2,        delta_t_gen]]).float()  

Q = q2 * torch.tensor([[1/3*delta_t**3, 1/2*delta_t**2],
                       [1/2*delta_t**2,        delta_t]]).float() 

R = r2 * torch.eye(n)
R_onlyPos = r2