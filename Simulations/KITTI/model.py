import torch
import math
import pykitti
import numpy as np


if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

basedir = 'Simulations/KITTI/KITTI_dataset/'
date = '2011_09_26'
drive = tuple()
drive = ('0002','0005','0009','0011','0013','0014','0017','0018','0048','0051','0056','0057','0059','0060','0084','0091','0093','0095','0096','0104','0106','0113','0117')

def load_kittidata(basedir, date, drive):
    data = pykitti.raw(basedir, date, drive)

    point_imu = np.array([0,0,0,1])
    point_xyz = [np.delete(o.T_w_imu.dot(point_imu),-1) for o in data.oxts]
    array_xyz = np.vstack(point_xyz).transpose()

    point_vf = np.hstack([o.packet.vf for o in data.oxts])
    point_vl = np.hstack([o.packet.vl for o in data.oxts])
    point_vu =  np.hstack([o.packet.vu for o in data.oxts])
    array_vel = np.vstack((point_vf,point_vl,point_vu))

    # point_roll = np.hstack([o.packet.roll for o in data.oxts])
    # point_pitch = np.hstack([o.packet.pitch for o in data.oxts])
    # point_yaw =  np.hstack([o.packet.yaw for o in data.oxts])
    # array_rpy = np.vstack((point_roll,point_pitch,point_yaw))

    # point_wf = np.hstack([o.packet.wf for o in data.oxts])
    # point_wl = np.hstack([o.packet.wl for o in data.oxts])
    # point_wu =  np.hstack([o.packet.wu for o in data.oxts])
    # array_w = np.vstack((point_wf,point_wl,point_wu))


    return torch.from_numpy(np.vstack((array_xyz,array_vel)))

gt_data = load_kittidata(basedir, date, '0001')
gt_data = torch.unsqueeze(gt_data,0)
for item in drive:
    ground_truth = torch.unsqueeze(load_kittidata(basedir, date, item),0)
    gt_data = torch.cat((gt_data,ground_truth),0)
print(gt_data.size())

#########################
### Design Parameters ###
#########################
m = 6 # [x y z x' y' z']
n = 3 # [x y z]
variance = 0
m2_0 = variance * torch.eye(m)

# Sampling period of KITTI
delta_t = 1/100

#####################################
### Process and Observation model ###
#####################################

F_kitti =torch.tensor([[1.0, 0.0, 0.0, delta_t, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, delta_t, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0, delta_t],
                       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).double()

H_kitti = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).double()





## Angle of rotation in the 3 axes
roll_deg = yaw_deg = pitch_deg = 1

roll = roll_deg * (math.pi/180)
yaw = yaw_deg * (math.pi/180)
pitch = pitch_deg * (math.pi/180)

RX = torch.tensor([
                [1, 0, 0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)]])
RY = torch.tensor([
                [math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)]])
RZ = torch.tensor([
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1]])

RotMatrix = torch.mm(torch.mm(RZ, RY), RX)
H_mod = torch.mm(RotMatrix,torch.eye(n))
H_mod = torch.cat([H_mod, torch.zeros(3,3)],dim=1)

#########################
### Noise Parameters ####
#########################
# Noise Parameters
lambda_r2 = torch.tensor([1], dtype=torch.float32)
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
lambda_q2 = torch.mul(v,lambda_r2)
lambda_r = torch.sqrt(lambda_r2)
lambda_q = torch.sqrt(lambda_q2)

# Noise Matrices
q_1d = (lambda_q**2) * delta_t
q_2d = (lambda_q**2) * (delta_t**2) /2
q_3d = (lambda_q**2) * (delta_t**3) /3
Q = torch.tensor([[q_3d, 0.0, 0.0, q_2d, 0.0, 0.0],
                  [0.0, q_3d, 0.0, 0.0, q_2d, 0.0],
                  [0.0, 0.0, q_3d, 0.0, 0.0, q_2d],
                  [q_2d, 0.0, 0.0, q_1d, 0.0, 0.0],
                  [0.0, q_2d, 0.0, 0.0, q_1d, 0.0],
                  [0.0, 0.0, q_2d, 0.0, 0.0, q_1d]])
R = (lambda_r**2) * torch.eye(n)