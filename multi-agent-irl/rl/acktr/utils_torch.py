import torch
import torch.nn as nn

def sample(logits, axis=1):
    noise = torch.rand_like(logits)
    return torch.argmax(logits - torch.log(-torch.log(noise)), dim=axis)

def fc(ns, nh, init_scale=1.0):
    linear = nn.Linear(ns, nh)
    nn.init.orthogonal_(linear.weight, gain=init_scale)
    nn.init.constant_(linear.bias, 0.0)
    return linear