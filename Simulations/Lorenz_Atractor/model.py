import math
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch import autograd
from filing_paths import path_model
import sys
sys.path.insert(1, path_model)
from parameters import m, n, J, delta_t,delta_t_test,delta_t_gen, H_design, B, C, B_mod, C_mod, delta_t_mod, J_mod, H_mod, H_design_inv, H_mod_inv,RotMatrix

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")


def f_test(x):
    
    BX = torch.reshape(torch.matmul(B, x),(m,m))
    A = torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)),C)
    
    # Taylor Expansion for F    
    F = torch.eye(m)
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t_test, j)/math.factorial(j)).to(dev)
        F = torch.add(F, F_add).to(dev)

    return torch.matmul(F, x)

def f_gen(x):
    BX = torch.reshape(torch.matmul(B, x),(m,m))
    A = torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)),C)  
    # Taylor Expansion for F    
    F = torch.eye(m)
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t_gen, j)/math.factorial(j)).to(dev)
        F = torch.add(F, F_add).to(dev)

    return torch.matmul(F, x)

def f(x):
    BX = torch.reshape(torch.matmul(B, x),(m,m))
    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    A = (torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)),C)).to(dev)
    
    # Taylor Expansion for F    
    F = torch.eye(m)
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j)).to(dev)
        F = torch.add(F, F_add).to(dev)
    x_out = torch.matmul(F, x)
    return x_out

def h(x):
    y = torch.matmul(H_design,x).to(dev)
    return y
    #return toSpherical(x)

def fInacc(x):
    BX = torch.reshape(torch.matmul(B_mod, x),(m,m))
    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    A = torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)),C_mod)
    
    # Taylor Expansion for F    
    F = torch.eye(m)
    for j in range(1,J_mod+1):
        F_add = (torch.matrix_power(A*delta_t_mod, j)/math.factorial(j)).to(dev)
        F = torch.add(F, F_add).to(dev)

    return torch.matmul(F, x)

def fRotate(x):
    BX = torch.reshape(torch.matmul(B, x),(m,m)) 
    A = (torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)),C)).to(dev)  
    # Taylor Expansion for F    
    F = torch.eye(m)
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j)).to(dev)
        F = torch.add(F, F_add).to(dev)
    F_rotated = torch.mm(RotMatrix,F)
    return torch.matmul(F_rotated, x)

def h_nonlinear(x):
    return toSpherical(x)


def f_interpolate(x, n=2):
    
    for _ in range(n):
        BX = torch.reshape(torch.matmul(B_mod, x),(m,m))
        A = torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)),C_mod)#.to(dev, non_blocking=True)
   
        # Taylor Expansion for F    
        F = torch.eye(m)#.to(dev, non_blocking=True)
        for j in range(1,J+1):
            F_add = torch.matrix_power(A*delta_t/n, j)/math.factorial(j)
            F = torch.add(F, F_add)
        x = torch.matmul(F, x)
        return x

def f_interpolate_approx(x, n=2):

    for _ in range(n):
        BX = torch.reshape(torch.matmul(B_mod, x),(m,m))
        A = torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)),C_mod)#.to(dev, non_blocking=True)
   
        # Taylor Expansion for F    
        F = torch.eye(m)#.to(dev, non_blocking=True)
        for j in range(1,J_mod+1):
            F_add = torch.matrix_power(A*delta_t/n, j)/math.factorial(j)
            F = torch.add(F, F_add)
        x = torch.matmul(F, x)
    return x


def hRotate(x):
    return torch.matmul(H_mod,x)
    #return toSpherical(x)

def getJacobian(x, g):
    
    # if(x.size()[1] == 1):
    #     y = torch.reshape((x.T),[x.size()[0]])
    
    y = torch.reshape((x.permute(*torch.arange(x.ndim - 1, -1, -1))),[x.size()[0]])

    Jac = autograd.functional.jacobian(g, y)
    Jac = Jac.view(-1,m)
    return Jac

def toSpherical(cart):

    rho = torch.norm(cart, p=2).view(1,1)
    phi = torch.atan2(cart[1, ...], cart[0, ...]).view(1, 1)
    phi = phi + (phi < 0).type_as(phi) * (2 * torch.pi)

    theta = torch.acos(cart[2, ...] / rho).view(1, 1)

    spher = torch.cat([rho, theta, phi], dim=0)

    return spher

def toCartesian(sphe):

    rho = sphe[0,:]
    theta = sphe[1,:]
    phi = sphe[2,:]

    x = (rho * torch.sin(theta) * torch.cos(phi)).view(1,-1)
    y = (rho * torch.sin(theta) * torch.sin(phi)).view(1,-1)
    z = (rho * torch.cos(theta)).view(1,-1)

    cart = torch.cat([x,y,z],dim=0)

    return cart

def hInv(y):
    return torch.matmul(H_design_inv,y)
    #return toCartesian(y)


def hInaccInv(y):
    return torch.matmul(H_mod_inv,y)
    #return toCartesian(y)

