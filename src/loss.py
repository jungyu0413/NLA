import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import numpy as np

# Noise-aware-Adaptive Weight 
class NAW(nn.Module):
    def __init__(self, args, mu_x_t, mu_y_t=0.50, mu_x_f=0.30, mu_y_f=0.15, f_minor=0.80, f_major=0.85, t_minor=-0.50, t_major=0.80, t_lambda=1.0):
        super(NAW, self).__init__()
        t_minor = args.eps * t_minor
        self.t_lambda = t_lambda
        self.t_mu = torch.tensor([mu_x_t, mu_y_t],dtype=torch.float).unsqueeze(1).to('cuda')
        self.f_mu = torch.tensor([mu_x_f, mu_y_f],dtype=torch.float).unsqueeze(1).to('cuda')
        self.t_cov = torch.tensor([[t_major, t_minor], [t_minor, t_major]],dtype=torch.float).to('cuda')
        self.f_cov = torch.tensor([[f_major, f_minor], [f_minor, f_major]],dtype=torch.float).to('cuda')

        
        
    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        P_GT = probs.gather(1, targets.unsqueeze(1)).squeeze()

        top2_probs = probs.topk(2, dim=1).values
        max_probs = top2_probs[:, 0]
        second_max_probs = top2_probs[:, 1]

        P_MAX_is_GT = max_probs == P_GT
        
        P_NN = torch.where(P_MAX_is_GT, second_max_probs, max_probs)
        SV_flat = torch.stack([P_GT, P_NN], axis=0)
        
        t_value = self.t_lambda*self.co_gau(SV_flat, self.t_cov, self.t_mu) 
        f_value = self.co_gau(SV_flat, self.f_cov, self.f_mu)

        value = torch.where(P_MAX_is_GT, t_value, f_value)
        Adaptive_loss = ce_loss * value
        
        return Adaptive_loss
    
    
    def co_gau(self, SV_flat, cov, mu):
        diff = SV_flat - mu  # [2, N]
        inv_cov = torch.linalg.inv(cov)  # [2, 2]
        det_cov = torch.det(cov)  # scalar
        
        exponent = -0.5 * torch.sum(diff * (inv_cov @ diff), dim=0)
    
        k = SV_flat.shape[0]  
        norm_factor = 1.0 / torch.sqrt((2 * torch.tensor(math.pi)) ** k * det_cov)
    
        value = norm_factor * torch.exp(exponent)  
        return value
    




# JSD regularizer
def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    kl_div_p = F.kl_div(F.log_softmax(p, dim=1), F.softmax(m, dim=1), reduction='sum')
    kl_div_q = F.kl_div(F.log_softmax(q, dim=1), F.softmax(m, dim=1), reduction='sum')
    js_divergence = 0.5 * (kl_div_p + kl_div_q)
    return js_divergence


# sigma scheduler
def exponential_scheduler(epoch, epochs, slope=-15, sch_bool=True):
    if sch_bool:
        return (1 - np.exp(slope * epoch / epochs))
    else:
        return 1
    
