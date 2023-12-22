import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl.acktr.utils_torch import fc
from rl.acktr.utils_torch import Scheduler

disc_types = ['decentralized', 'decentralized-all']

class Discriminator(nn.Module):
    def __init__(self, ob_spaces, ac_spaces, state_only, discount,
                 nstack, index, device, disc_type='decentralized', hidden_size=128,
                 lr_rate=0.01, total_steps=50000, kfac_clip=0.001, max_grad_norm=0.5,
                 weight_decay=1e-4):
        super(Discriminator, self).__init__()
        self.lr = Scheduler(v=lr_rate, nvalues=total_steps, schedule='linear')
        self.disc_type = disc_type
        if disc_type not in disc_types:
            assert False
        self.state_only = state_only
        self.gamma = discount
        self.index = index
        self.device = device
        self.max_grad_norm = max_grad_norm

        ob_space = ob_spaces[index]
        ac_space = ac_spaces[index]
        self.ob_shape = ob_space.shape[0] * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        try:
            nact = ac_space.n
        except:
            nact = ac_space.shape[0]
        self.ac_shape = nact * nstack
        
        try:
            self.all_ac_shape = sum([ac.n for ac in ac_spaces]) * nstack
        except:
            self.all_ac_shape = sum([ac.shape[0] for ac in ac_spaces]) * nstack
        self.hidden_size = hidden_size
        
        self.labels_d = 1
        self.lprobs_d = 1
        
        if disc_type == 'decentralized':
            self.obs_d = self.ob_shape
            self.nobs_d = self.ob_shape
            self.act_d = self.ac_shape     
        elif disc_type == 'decentralized-all':
            self.obs_d = self.all_ob_shape
            self.nobs_d = self.all_ob_shape
            self.act_d = self.all_ac_shape     
        else:
            assert False
        
        input_d = self.obs_d
        if not self.state_only:
            input_d = self.obs_d, self.act_d
            
        
        self.reward_fn = self.relu_net(input_d, dout=1).to(device)
        # self.reward = self.tanh_net(rew_input, dout=1)
    
        self.value_fn_n = self.relu_net(self.nobs_d, dout=1).to(device)
        # self.value_fn_n = self.tanh_net(self.nobs, dout=1)
    
        self.value_fn = self.relu_net(self.obs_d, dout=1).to(device)
        # self.value_fn = self.tanh_net(self.obs, dout=1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr_rate, weight_decay=weight_decay) # TODO scheduler support


    def relu_net(self, din, layers=2, dout=1, hidden_size=128):
        seq_layers = []
        for i in range(layers):
            if i == 0:
                seq_layers.append(nn.Linear(din, hidden_size))
                seq_layers.append(nn.ReLU())
            else:
                seq_layers.append(nn.Linear(hidden_size, hidden_size))
                seq_layers.append(nn.ReLU())
        seq_layers.append(nn.Linear(hidden_size, dout))
        seq = torch.nn.Sequential(*seq_layers)
        return seq
       
    def tanh_net(self, x, layers=2, dout=1, hidden_size=128):
        pass

    def get_reward(self, obs, acs, obs_next, path_probs, discrim_score=False):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
               
        obs = self.to_torch(obs)
        acs = self.to_torch(acs)
        obs_next = self.to_torch(obs_next)
        
        raw_input = obs
        if not self.state_only:
            raw_input = torch.cat([obs, acs], axis=1)
        
        if discrim_score:
            log_q_tau = path_probs
            log_p_tau = self.reward_fn(raw_input) + self.gamma * self.value_fn_n(obs_next) - self.value_fn(obs)
            
            log_pq = torch.logsumexp(torch.cat([log_p_tau, log_q_tau]), dim=0)
            scores = torch.exp(log_p_tau - log_pq)  
            eps=1e-20        
            score = torch.log(scores + eps) - torch.log(1 - scores + eps)
        else:
            score = self.reward_fn(raw_input)
        return score.detach().cpu().numpy()
            
        

    def train(self, g_obs, g_acs, g_nobs, g_probs, e_obs, e_acs, e_nobs, e_probs):
        labels = np.concatenate((np.zeros([g_obs.shape[0], 1]), np.ones([e_obs.shape[0], 1])), axis=0)
        
        labels = self.to_torch(labels)
        obs = self.to_torch(np.concatenate([g_obs, e_obs], axis=0))
        act = self.to_torch(np.concatenate([g_acs, e_acs], axis=0))
        nobs = self.to_torch(np.concatenate([g_nobs, e_nobs], axis=0))
        lprobs = self.to_torch(np.concatenate([g_probs, e_probs], axis=0))
        print(f'g_obs: {g_obs.shape}')
        print(f'e_obs: {e_obs.shape}')
        print(f'g_acs: {g_acs.shape}')
        print(f'e_acs: {e_acs.shape}')
        print(f'g_nobs: {g_nobs.shape}')
        print(f'e_nobs: {e_nobs.shape}')
        print(f'g_probs: {g_probs.shape}')
        print(f'e_probs: {e_probs.shape}')
        raw_input = obs
        if not self.state_only:
            raw_input = torch.cat([obs, act], axis=1)
        
        log_q_tau = lprobs
        log_p_tau = self.reward_fn(raw_input) + self.gamma * self.value_fn_n(nobs) - self.value_fn(obs)
        log_pq = torch.logsumexp(torch.cat([log_p_tau, log_q_tau]), dim=0)
        print(f'labels: {labels.shape}')
        print(f'log_q_tau: {log_q_tau.shape}')
        print(f'log_p_tau: {log_p_tau.shape}')
        print(f'log_pq: {log_pq.shape}')
        loss = -torch.mean(labels * (log_p_tau - log_pq) + (1 - labels) * (log_q_tau - log_pq))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return loss.detach().cpu()

    def save(self, save_path):
        torch.save(self.state_dict(), save_path) # TODO save optimizer parameters
        
    def load(self, load_path):
        self.load_state_dict(torch.load(load_path)) # TODO load optimizer parameters
        
    def to_torch(self, param):
        return torch.tensor(param, dtype=torch.float32, requires_grad=True).to(self.device)