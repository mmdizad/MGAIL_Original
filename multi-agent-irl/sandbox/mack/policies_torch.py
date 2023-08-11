import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalPolicy(nn.Module):
    def __init__(self, ob_space, ac_space, ob_spaces, ac_spaces,
                 nenv, nsteps, nstack, name='model'):
        super(CategoricalPolicy, self).__init__()
        nbatch = nenv * nsteps
        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        all_ob_shape = (nbatch, sum([obs.shape[0] for obs in ob_spaces]) * nstack)
        nact = ac_space.n
        self.actions = torch.empty(nbatch, dtype=torch.int32)
        all_ac_shape = (nbatch, (sum([ac.n for ac in ac_spaces]) - nact) * nstack)
        self.X = torch.empty(ob_shape, dtype=torch.int32)  # obs
        self.X_v = torch.empty(all_ob_shape, dtype=torch.int32)
        self.A_v = torch.empty(all_ac_shape, dtype=torch.int32)

        self.fc1 = nn.Linear(ob_shape[1], 128)
        self.fc2 = nn.Linear(128, 128)
        self.pi = nn.Linear(128, nact)

        if len(ob_spaces) > 1:
            Y = torch.cat([self.X_v, self.A_v], dim=1)
        else:
            Y = self.X_v
        self.fc3 = nn.Linear(Y.shape[1], 256)
        self.fc4 = nn.Linear(256, 256)
        self.vf = nn.Linear(256, 1)

    def forward(self, ob, obs, a_v):
        h1 = F.relu(self.fc1(ob))
        h2 = F.relu(self.fc2(h1))
        pi = self.pi(h2)

        if a_v is not None:
            Y = torch.cat([obs, a_v], dim=1)
        else:
            Y = obs
        h3 = F.relu(self.fc3(Y))
        h4 = F.relu(self.fc4(h3))
        vf = self.vf(h4)

        return pi, vf

    def step_log_prob(self, ob, acts):
        pi, _ = self.forward(self.X, self.X_v, self.A_v)
        log_prob = F.log_softmax(pi, dim=1)
        selected_log_prob = log_prob.gather(1, acts.unsqueeze(1))
        return selected_log_prob

    def step(self, ob, obs, a_v, *_args, **_kwargs):
        pi, vf = self.forward(ob, obs, a_v)
        a = torch.multinomial(F.softmax(pi, dim=1), 1).squeeze(1)
        return a, vf, []

    def value(self, ob, a_v, *_args, **_kwargs):
        _, vf = self.forward(ob, None, a_v)
        return vf.squeeze(1)