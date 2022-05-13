"""# **Class: Deep Unfolded RTSNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func

from RTSNet_nn import RTSNetNN

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")


class RTSNetNN_multipass(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self, iterations):
        super().__init__()
        self.iterations = iterations
        self.RTSNet_passes = []

        for i in range(self.iterations):            
            self.RTSNet_passes.append(RTSNetNN())
    
    #############
    ### Build ###
    #############
    def NNBuild_multipass(self, ssModel, KNet_in_mult = 5, KNet_out_mult = 40, RTSNet_in_mult = 5, RTSNet_out_mult = 40):
        # Set State Evolution Function
        self.f = ssModel.f
        self.m = ssModel.m

        # Set Observation Function
        self.h = ssModel.h
        self.n = ssModel.n

        self.InitSystemDynamics_multipass(self.f,self.h,self.m,self.n)

        for i in range(self.iterations):
            self.RTSNet_passes[i].InitKGainNet(ssModel.prior_Q, ssModel.prior_Sigma, ssModel.prior_S, KNet_in_mult, KNet_out_mult)
            self.RTSNet_passes[i].InitRTSGainNet(ssModel.prior_Q, ssModel.prior_Sigma, RTSNet_in_mult, RTSNet_out_mult)
    
    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics_multipass(self, f, h, m, n):
        for i in range(self.iterations):
            # Set State Evolution Function
            self.RTSNet_passes[i].f = f
            self.RTSNet_passes[i].m = m

            # Set Observation Function
            self.RTSNet_passes[i].h = h
            self.RTSNet_passes[i].n = n

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence_multipass(self, i, M1_0, T):
        self.T = T
        
        self.RTSNet_passes[i].m1x_posterior = torch.squeeze(M1_0).to(dev, non_blocking=True)
        self.RTSNet_passes[i].m1x_posterior_previous = self.RTSNet_passes[i].m1x_posterior.to(dev, non_blocking=True)
        self.RTSNet_passes[i].m1x_prior_previous = self.RTSNet_passes[i].m1x_posterior.to(dev, non_blocking=True)
        self.RTSNet_passes[i].y_previous = self.h(self.RTSNet_passes[i].m1x_posterior).to(dev, non_blocking=True)

    ####################################
    ### Initialize Backward Sequence ###
    ####################################
    def InitBackward_multipass(self, i, filter_x):
        self.RTSNet_passes[i].s_m1x_nexttime = torch.squeeze(filter_x)

    ###############
    ### Forward ###
    ###############
    def forward(self, iteration, yt, filter_x, filter_x_nexttime, smoother_x_tplus2):

        if yt is None:
            # BW pass
            return self.RTSNet_passes[iteration].RTSNet_step(filter_x, filter_x_nexttime, smoother_x_tplus2)
        else:
            # FW pass
            yt = yt.to(dev, non_blocking=True)
            return self.RTSNet_passes[iteration].KNet_step(yt)
    
    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden_multipass(self):
        for i in range(self.iterations):
            self.RTSNet_passes[i].init_hidden()


       



# from Linear_sysmdl import SystemModel
# from Extended_data import N_E, N_CV, N_T, F, H, F_rotated, H_rotated, T, T_test, m1_0, m2_0, m, n
# if __name__ == '__main__':
#     iterations = 5
#     r2 = torch.tensor([1])
#     vdB = -20 # ratio v=q2/r2
#     v = 10**(vdB/10)
#     q2 = torch.mul(v,r2)
#     r = torch.sqrt(r2)
#     q = torch.sqrt(q2)
#     sys_model = SystemModel(F, q, H, r, T, T_test)
#     sys_model.InitSequence(m1_0, m2_0)

#     DU_RTSNet = RTSNetNN_multipass(iterations)
#     DU_RTSNet.NNBuild_multipass(sys_model)
#     print("Number of trainable parameters for RTSNet:",sum(p.numel() for p in DU_RTSNet.RTSNet_passes[1].parameters() if p.requires_grad))







    
        
