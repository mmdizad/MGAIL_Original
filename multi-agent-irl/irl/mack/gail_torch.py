import os.path as osp
import random
import time

import joblib
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from rl.acktr.utils_torch import Scheduler, discount_with_dones
from rl.acktr.utils_torch import cat_entropy, mse, onehot, multionehot

from rl import logger
from rl.acktr import kfac
from rl.common.torch_util import set_global_seeds, explained_variance
from irl.mack.kfac_discriminator_torch import Discriminator
# from irl.mack.kfac_discriminator_wgan import Discriminator
from irl.dataset import Dset

class GeneralModel():
    def __init__(self, policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=2, nsteps=200,
                 nstack=1, ent_coef=0.00, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                kfac_clip=0.001, lrschedule='linear', identical=None):
        
        nbatch = nenvs * nsteps
        self.num_agents = num_agents = len(ob_space)
        self.n_actions = [ac_space[k].n for k in range(self.num_agents)]
        if identical is None:
            identical = [False for _ in range(self.num_agents)]
            
        scale = [1 for _ in range(num_agents)]
        pointer = [i for i in range(num_agents)]
        h = 0
        for k in range(num_agents):
            if identical[k]:
                scale[h] += 1
            else:
                pointer[h] = k
                h = k
        pointer[h] = num_agents
        
        
    def train(self, obs, states, rewards, masks, actions, values):
        advs = [rewards[k] - values[k] for k in range(self.num_agents)]
        for _ in range(len(obs)):
            cur_lr = self.lr.value()
        
        ob = np.concatenate(obs, axis=1)
        for k in range(self.num_agents):
            if self.identical[k]:
                continue
            new_map = {}
            if self.num_agents > 1:
                action_v = []
                for j in range(k, self.pointer[k]):
                    action_v.append(np.concatenate([multionehot(actions[i], self.n_actions[i])
                                               for i in range(self.num_agents) if i != k], axis=1))
                action_v = np.concatenate(action_v, axis=0)
                new_map.update({self.train_model[k].A_v: action_v})
                td_map.update({self.train_model[k].A_v: action_v})
                
            new_map.update({
                self.train_model[k].X: np.concatenate([obs[j] for j in range(k, self.pointer[k])], axis=0),
                self.train_model[k].X_v: np.concatenate([ob.copy() for j in range(k, self.pointer[k])], axis=0),
                A[k]: np.concatenate([actions[j] for j in range(k, self.pointer[k])], axis=0),
                ADV[k]: np.concatenate([advs[j] for j in range(k, self.pointer[k])], axis=0),
                R[k]: np.concatenate([rewards[j] for j in range(k, self.pointer[k])], axis=0),
                PG_LR[k]: cur_lr / float(self.scale[k])
            })
            sess.run(self.train_ops[k], feed_dict=new_map)
            td_map.update(new_map)
             

def learn(policy, expert, env, env_id, seed, total_timesteps=int(40e6), gamma=0.99, lam=0.95, log_interval=1, nprocs=32,
          nsteps=20, nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=100, lrschedule='linear', dis_lr=0.001, disc_type='decentralized',
          bc_iters=500, identical=None):
    set_global_seeds(seed)
    buffer = None

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_agents = (len(ob_space))
    model = GeneralModel(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps=nsteps,
                               nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
                               lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                               lrschedule=lrschedule, identical=identical)