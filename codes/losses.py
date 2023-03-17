import torch
import torch.nn.functional as F
import numpy as np
import math



class MSE:
    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)



class Grad:
    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]) 
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])         

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad



class NCC:
    def __init__(self, scale=1, win=None, GPU="0"):
        self.win = win
        self.scale = scale
        self.GPU = GPU

    def loss(self, y_true, y_pred):

        if self.scale==1:
            I = y_true
        else:
            I = F.interpolate(y_true, scale_factor=1/self.scale, mode='bicubic', align_corners=False, recompute_scale_factor=False)
        J = y_pred

        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        if self.win==None:
            win = [9] * ndims             
        else:
            win = [self.win] * ndims             

        sum_filt = torch.ones([1, 1, *win]).to("cuda:"+self.GPU)

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        conv_fn = getattr(F, 'conv%dd' % ndims)

        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
