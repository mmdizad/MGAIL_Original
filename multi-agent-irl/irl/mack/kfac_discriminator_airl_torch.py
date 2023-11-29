import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl.acktr.utils_torch import fc
from rl.acktr.utils_torch import Scheduler

disc_types = ['decentralized', 'centralized', 'single']

class Discriminator(object):
    def __init__(self, ob_spaces, ac_spaces, state_only, discount,
                 nstack, index, disc_type='decentralized', hidden_size=128,
                 lr_rate=0.01, total_steps=50000, kfac_clip=0.001, max_grad_norm=0.5,
                 l2_loss_ratio=0.01):
        self.lr = Scheduler(v=lr_rate, nvalues=total_steps, schedule='linear')
        self.disc_type = disc_type
        self.l2_loss_ratio = l2_loss_ratio
        if disc_type not in disc_types:
            assert False
        self.state_only = state_only
        self.gamma = discount
        self.index = index
        self.sess = sess
        ob_space = ob_spaces[index]
        ac_space = ac_spaces[index]
        self.ob_shape = ob_space.shape[0] * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        try:
            nact = ac_space.n
        except:
            nact = ac_space.shape[0]
        self.ac_shape = nact * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        try:
            self.all_ac_shape = sum([ac.n for ac in ac_spaces]) * nstack
        except:
            self.all_ac_shape = sum([ac.shape[0] for ac in ac_spaces]) * nstack
        self.hidden_size = hidden_size


    def relu_net(self, x, layers=2, dout=1, hidden_size=128):
        pass
       
    def tanh_net(self, x, layers=2, dout=1, hidden_size=128):
        pass
        
    def get_variables(self):
        pass

    def get_trainable_variables(self):
        pass

    def get_reward(self, obs, acs, obs_next, path_probs, discrim_score=False):
        pass

    def train(self, g_obs, g_acs, g_nobs, g_probs, e_obs, e_acs, e_nobs, e_probs):
        pass

    def restore(self, path):
        pass

    def save(self, save_path):
        pass

    def load(self, load_path):
        pass