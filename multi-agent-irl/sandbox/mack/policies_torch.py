import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rl.acktr.utils_torch import fc, sample


class CategoricalPolicy(nn.Module):
    def __init__(self, ob_space, ac_space, ob_spaces, ac_spaces, nstack, device):
        print('inputs:')
        print(ob_space)
        print(ac_space)
        print(ob_spaces)
        print(ac_spaces)
        
        print(ob_space.shape)
        super(CategoricalPolicy, self).__init__()
        self.device = device
        ob_shape = ob_space.shape[0] * nstack
        all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        nact = ac_space.n
        all_ac_shape = (sum([ac.n for ac in ac_spaces]) - nact) * nstack

        self.fc1 = fc(ob_shape, nh=128, init_scale=np.sqrt(2))
        self.fc2 = fc(128, nh=128, init_scale=np.sqrt(2))
        self.pi_f = fc(128, nact)

        if len(ob_spaces) > 1:
            in_space = all_ob_shape + all_ac_shape
        else:
            in_space = all_ob_shape
            
        self.fc3 = fc(in_space, nh=256, init_scale=np.sqrt(2))
        self.fc4 = fc(256, nh=256, init_scale=np.sqrt(2))
        self.vf_f = fc(256, 1)
        
        self.initial_state = [] # not stateful

    def forward(self, ob, obs, a_v):
        pi = self.compute_pi(ob)
        vf = self.compute_vf(obs, a_v)
        return pi, vf

    def compute_pi(self, ob):
        ob = torch.tensor(ob, dtype=torch.float32).to(self.device)
        h1 = F.relu(self.fc1(ob))
        h2 = F.relu(self.fc2(h1))
        pi = F.softmax(self.pi_f(h2), dim=1)
        return pi
    
    def compute_vf(self, obs, a_v):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        if a_v is not None:
            a_v = torch.tensor(a_v, dtype=torch.float32).to(self.device)
            Y = torch.cat([obs, a_v], dim=1)
        else:
            Y = obs
        h3 = F.relu(self.fc3(Y))
        h4 = F.relu(self.fc4(h3))
        vf = self.vf_f(h4)
        return vf


    def step_log_prob(self, ob, acts):
        pi = self.compute_pi(ob)
        loss = -1 * nn.CrossEntropyLoss()(pi, acts)
        return loss

    def step(self, ob, obs, a_v):
        pi, v = self.forward(ob, obs, a_v)
        a = sample(pi)
        return a, v, [] # dummy state

    def value(self, ob, a_v):
        vf = self.compute_vf(ob, a_v)
        return vf[:, 0]
    
    def policy_params(self):
        combined_params = list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.pi_f.parameters())
        return combined_params 
