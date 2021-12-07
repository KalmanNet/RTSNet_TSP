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

basedir = 'Simulations/KITTI/KITTI_dataset/Person'
date = '2011_09_28'
drive = tuple()
### City category
# drive = ('0001','0002','0005','0009','0011','0013','0014','0017','0018','0048','0051','0056','0057','0059','0060','0084','0091','0093','0095','0096','0104','0106','0113','0117')
### Person category
drive = ('0053','0054','0057','0065','0066','0068','0070','0071','0075','0077','0078','0080','0082','0086','0087','0089','0090','0094','0095','0096','0098','0100','0102','0103','0104','0106','0108','0110','0113','0117','0119','0121','0122','0125','0126','0128','0132','0134','0135','0136','0138','0141','0143','0145','0146','0149','0153','0154','0155','0156','0160','0161','0162','0165','0166','0167','0168','0171','0174','0177','0179','0183','0184','0185','0186','0187','0191','0192','0195','0198','0199','0201','0204','0205','0208','0209','0214','0216','0220','0222')

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

gt_data = []
for item in drive:
    ground_truth = load_kittidata(basedir, date, item).float()
    gt_data.append(ground_truth)
print("Total number of trajectories:",len(gt_data))
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
                       [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

H_kitti = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])





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

lambda_r =  torch.tensor([0.1], dtype=torch.float32)
lambda_q =  torch.tensor([1], dtype=torch.float32)

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