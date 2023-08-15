import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def fc(ns, nh, init_scale=1.0):
    linear = nn.Linear(ns, nh)
    nn.init.orthogonal_(linear.weight, gain=init_scale)
    nn.init.constant_(linear.bias, 0.0)
    return linear

class CategoricalPolicy(nn.Module):
    def __init__(self, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack):
        super(CategoricalPolicy, self).__init__()
        ob_shape = ob_space.shape[0] * nstack
        all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        nact = ac_space.n
        all_ac_shape = (sum([ac.n for ac in ac_spaces]) - nact) * nstack

        self.fc1 = fc(ob_shape, nh=128, init_scale=np.sqrt(2))
        self.fc2 = fc(128, nh=128, init_scale=np.sqrt(2))
        self.pi_f = fc(128, nact)

        if len(nenv * nsteps) > 1:
            in_space = all_ob_shape + all_ac_shape
        else:
            in_space = all_ob_shape
            
        self.fc3 = fc(in_space, nh=256, init_scale=np.sqrt(2))
        self.fc4 = fc(256, nh=256, init_scale=np.sqrt(2))
        self.vf_f = fc(256, 1)

    def forward(self, ob, obs, a_v):
        h1 = F.relu(self.fc1(ob))
        h2 = F.relu(self.fc2(h1))
        pi = self.pi_f(h2)

        if a_v is not None:
            Y = torch.cat([obs, a_v], dim=1)
        else:
            Y = obs
        h3 = F.relu(self.fc3(Y))
        h4 = F.relu(self.fc4(h3))
        vf = self.vf_f(h4)

        return pi, vf

    def step_log_prob(self, ob, acts):
        pi, _ = self.forward(self.X, self.X_v, self.A_v)
        log_prob = F.log_softmax(pi, dim=1)
        selected_log_prob = log_prob.gather(1, acts.unsqueeze(1))
        return selected_log_prob

    def step(self, ob, obs, a_v):
        pi, vf = self.forward(ob, obs, a_v)
        a = torch.multinomial(F.softmax(pi, dim=1), 1).squeeze(1)
        return a, vf, []

    def value(self, ob, a_v):
        _, vf = self.forward(ob, None, a_v)
        return vf.squeeze(1)

