import math
import torch

from Simulations.Linear_CA.parameters import H_onlyPos
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch import autograd
from parameters import g, L, m, n, delta_t, delta_t_gen, H_id, H_onlyPos

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

def f_gen(x):    
    result = [x[0]+x[1]*delta_t_gen-(g/L * torch.sin(x[0])*0.5*(delta_t_gen**2)), x[1]-(g/L * torch.sin(x[0]))*delta_t_gen]
    result = torch.squeeze(torch.tensor(result))
    # print(result.size())
    return result

def f(x):
    result = [x[0]+x[1]*delta_t-(g/L * torch.sin(x[0])*0.5*(delta_t**2)), x[1]-(g/L * torch.sin(x[0]))*delta_t]
    result = torch.squeeze(torch.tensor(result))
    # print(result.size())
    return result

def h(x):
    return torch.matmul(H_id,x).to(cuda0)
    #return toSpherical(x)


def h_onlyPos(x):
    return torch.matmul(H_onlyPos,x)
    #return toSpherical(x)

def h_NL(x):
    return L*torch.sin(x[0])

def getJacobian(x, g):
    
    # if(x.size()[1] == 1):
    #     y = torch.reshape((x.T),[x.size()[0]])
    
    y = torch.reshape((x.permute(*torch.arange(x.ndim - 1, -1, -1))),[x.size()[0]])

    Jac = autograd.functional.jacobian(g, y)
    Jac = Jac.view(-1,m)
    return Jac


'''
x = torch.tensor([[1],[1],[1]]).float() 
H = getJacobian(x, 'ObsAcc')
print(H)
print(h(x))

F = getJacobian(x, 'ModAcc')
print(F)
print(f(x))
'''