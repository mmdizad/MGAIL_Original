import os.path as osp
import random
import time

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from irl.dataset import Dset
from irl.mack.kfac_discriminator_torch import Discriminator
from rl import logger
from rl.acktr.utils_torch import (Scheduler, cat_entropy, discount_with_dones,
                                  mse, multionehot, onehot)
from rl.common.torch_util import explained_variance, set_global_seeds
from scipy.stats import pearsonr, spearmanr


class GeneralModel():
    def __init__(self, policy, ob_space, ac_space, nenvs, total_timesteps, device, nprocs=2, nsteps=200,
                 nstack=1, ent_coef=0.00, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                kfac_clip=0.001, lrschedule='linear', identical=None):
        
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.device = device
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
        
        self.step_model = []
        self.train_model = []
        for k in range(num_agents):
            if identical[k]:
                self.step_model.append(self.step_model[-1])
                self.train_model.append(self.train_model[-1])
            else:
                self.train_model.append(policy(ob_space[k], ac_space[k], ob_space, ac_space,
                                            nstack, self.device).to(self.device))
                self.step_model.append(self.train_model[-1])
                       
        self.optim = []
        self.clones = []
        for k in range(num_agents):
            if identical[k]:
                self.optim.append(self.optim[-1])
                self.clones.append(self.clones[-1])
            else:
                self.optim.append(torch.optim.Adam(self.train_model[k].parameters(), lr=lr)) # TODO
                self.clones.append(torch.optim.Adam(self.train_model[k].policy_params(), lr=lr))
        
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        self.clone_lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        
        
    def train(self, obs, states, rewards, masks, actions, values):
        advs = [rewards[k] - values[k] for k in range(self.num_agents)]
        for _ in range(len(obs)):
            cur_lr = self.lr.value()
        
        ob = np.concatenate(obs, axis=1)
        
        policy_loss = value_loss = policy_entropy = []
        for k in range(self.num_agents):
            if self.identical[k]:
                continue
            A_v = None
            if self.num_agents > 1:
                action_v = []
                for _ in range(k, self.pointer[k]):
                        action_v.append(np.concatenate([multionehot(actions[i], self.n_actions[i])
                                                   for i in range(self.num_agents) if i != k], axis=1))
                action_v = np.concatenate(action_v, axis=0)
                A_v = torch.tensor(action_v)
            X = torch.tensor(np.concatenate([obs[j] for j in range(k, self.pointer[k])], axis=0))
            X_v = torch.tensor(np.concatenate([ob.copy() for _ in range(k, self.pointer[k])], axis=0))
            A = torch.tensor(np.concatenate([actions[j] for j in range(k, self.pointer[k])], axis=0))
            ADV =  torch.tensor(np.concatenate([advs[j] for j in range(k, self.pointer[k])], axis=0))
            R =  torch.tensor(np.concatenate([rewards[j] for j in range(k, self.pointer[k])], axis=0))


            pi, vf = self.train_model[k](X, X_v, A_v)
            logpac = F.cross_entropy(pi, A)
            entropy = torch.mean(cat_entropy(pi))
            pg_loss = torch.mean(ADV * logpac)
            pg_loss = pg_loss - self.ent_coef * entropy
            vf_loss = torch.mean(mse(vf, R))
            loss = pg_loss + self.vf_coef * vf_loss # TODO use cur_lr
            self.optim[k].zero_grad()
            loss.backward()
            self.optim[k].step()
            policy_loss.append(pg_loss.detach().cpu())
            value_loss.append(vf_loss.detach().cpu())
            policy_entropy.append(entropy.detach().cpu())
        
        return policy_loss, value_loss, policy_entropy
        
    def clone(self, obs, actions):
        cur_lr = self.clone_lr.value()
        lld_loss = []
        for k in range(self.num_agents):
            if self.identical[k]:
                continue

            X = torch.tensor(np.concatenate([obs[j] for j in range(k, self.pointer[k])], axis=0))
            A = torch.tensor(np.concatenate([actions[j] for j in range(k, self.pointer[k])], axis=0))
            pi = self.train_model[k].compute_pi(X)
            logpac = F.cross_entropy(pi, A)
            lld = torch.mean(logpac)
            self.clones[k].zero_grad() # TODO use cur_lr
            lld.backward()
            self.clones[k].step()
            lld_loss.append(lld.detach().cpu())
            
        return lld_loss

    def save(self, save_path):
        models_dict = {}
        for k in range(self.num_agents):
            key1 = f'train_model{k+1}'
            models_dict[key1] = self.train_model[k].state_dict()
        torch.save(models_dict, save_path)

    def load(self, load_path):
        models_dict = torch.load(load_path)
        for k in range(self.num_agents):
            key1 = f'train_model{k+1}'
            self.train_model[k].load_state_dict(models_dict[key1])
            self.step_model[k] = self.train_model[k]
        
    def step(self, ob, av):
        a, v, s = [], [], []
        obs = torch.tensor(np.concatenate(ob, axis=1))
        ob = torch.tensor(ob)
        for k in range(self.num_agents):
            a_v = torch.tensor(np.concatenate([multionehot(av[i], self.n_actions[i])
                                  for i in range(self.num_agents) if i != k], axis=1))
            with torch.no_grad():
                a_, v_, s_ = self.step_model[k].step(ob[k], obs, a_v)
            a.append(a_.detach().cpu())
            v.append(v_.detach().cpu())
            s.append(s_)
        return a, v, s
             
    def value(self, obs, av):
        v = []
        ob = torch.tensor(np.concatenate(obs, axis=1))
        for k in range(self.num_agents):
            a_v = torch.tensor(np.concatenate([multionehot(av[i], self.n_actions[i])
                                  for i in range(self.num_agents) if i != k], axis=1))
            with torch.no_grad():
                v_ = self.step_model[k].value(ob, a_v)
            v.append(v_.detach().cpu())
        return v
    
    

class Runner(object):
    def __init__(self, env, model, discriminator, nsteps, nstack, gamma, lam, disc_type):
        self.env = env
        self.model = model
        self.discriminator = discriminator
        self.disc_type = disc_type
        self.num_agents = len(env.observation_space)
        self.nenv = env.num_envs
        self.batch_ob_shape = [
            (self.nenv * nsteps, nstack * env.observation_space[k].shape[0]) for k in range(self.num_agents)
        ]
        self.obs = [
            np.zeros((self.nenv, nstack * env.observation_space[k].shape[0])) for k in range(self.num_agents)
        ]
        self.actions = [np.zeros((self.nenv, )) for _ in range(self.num_agents)]
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.n_actions = [env.action_space[k].n for k in range(self.num_agents)]
        self.dones = [np.array([False for _ in range(self.nenv)]) for k in range(self.num_agents)]

    def update_obs(self, obs):
        self.obs = obs

    def run(self):
        mb_obs = [[] for _ in range(self.num_agents)]
        mb_true_rewards = [[] for _ in range(self.num_agents)]
        mb_rewards = [[] for _ in range(self.num_agents)]
        mb_actions = [[] for _ in range(self.num_agents)]
        mb_values = [[] for _ in range(self.num_agents)]
        mb_dones = [[] for _ in range(self.num_agents)]
        mb_masks = [[] for _ in range(self.num_agents)]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.actions)
            if self.disc_type == 'decentralized':
                rewards = [np.squeeze(self.discriminator[k].get_reward(
                    self.obs[k], multionehot(self.actions[k], self.n_actions[k])))
                    for k in range(self.num_agents)]
            elif self.disc_type == 'centralized':
                mul = [multionehot(self.actions[k], self.n_actions[k]) for k in range(self.num_agents)]
                rewards = self.discriminator.get_reward(np.concatenate(self.obs, axis=1), np.concatenate(mul, axis=1))
                rewards = rewards.swapaxes(1, 0)
            elif self.disc_type == 'single':
                mul = [multionehot(self.actions[k], self.n_actions[k]) for k in range(self.num_agents)]
                rewards = self.discriminator.get_reward(np.concatenate(self.obs, axis=1), np.concatenate(mul, axis=1))
                rewards = np.repeat(rewards, self.num_agents).reshape(len(rewards), self.num_agents)
                rewards = rewards.swapaxes(1, 0)
            else:
                assert False

            self.actions = actions
            for k in range(self.num_agents):
                mb_obs[k].append(np.copy(self.obs[k]))
                mb_actions[k].append(actions[k])
                mb_values[k].append(values[k])
                mb_dones[k].append(self.dones[k])
                mb_rewards[k].append(rewards[k])
            actions_list = []
            for i in range(self.nenv):
                actions_list.append([onehot(actions[k][i], self.n_actions[k]) for k in range(self.num_agents)])
            obs, true_rewards, dones, _ = self.env.step(actions_list)
            self.states = states
            self.dones = dones
            for k in range(self.num_agents):
                for ni, done in enumerate(dones[k]):
                    if done:
                        self.obs[k][ni] = self.obs[k][ni] * 0.0
            self.update_obs(obs)
            for k in range(self.num_agents):
                mb_true_rewards[k].append(true_rewards[k])
        for k in range(self.num_agents):
            mb_dones[k].append(self.dones[k])

        # batch of steps to batch of rollouts
        for k in range(self.num_agents):
            mb_obs[k] = np.asarray(mb_obs[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])
            mb_true_rewards[k] = np.asarray(mb_true_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_rewards[k] = np.asarray(mb_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_actions[k] = np.asarray(mb_actions[k], dtype=np.int32).swapaxes(1, 0)
            mb_values[k] = np.asarray(mb_values[k], dtype=np.float32).swapaxes(1, 0)
            mb_dones[k] = np.asarray(mb_dones[k], dtype=np.bool).swapaxes(1, 0)
            mb_masks[k] = mb_dones[k][:, :-1]
            mb_dones[k] = mb_dones[k][:, 1:]


        mb_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        mb_true_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        last_values = self.model.value(self.obs, self.actions)
        # discount/bootstrap off value fn
        for k in range(self.num_agents):
            for n, (rewards, true_rewards, dones, value) in enumerate(zip(mb_rewards[k], mb_true_rewards[k], mb_dones[k], last_values[k].tolist())):
                rewards = rewards.tolist()
                dones = dones.tolist()
                true_rewards = true_rewards.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                    true_rewards = discount_with_dones(true_rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)
                    true_rewards = discount_with_dones(true_rewards, dones, self.gamma)
                mb_returns[k][n] = rewards
                mb_true_returns[k][n] = true_rewards

        for k in range(self.num_agents):
            mb_returns[k] = mb_returns[k].flatten()
            mb_masks[k] = mb_masks[k].flatten()
            mb_values[k] = mb_values[k].flatten()
            mb_actions[k] = mb_actions[k].flatten()

        mh_actions = [multionehot(mb_actions[k], self.n_actions[k]) for k in range(self.num_agents)]
        mb_all_obs = np.concatenate(mb_obs, axis=1)
        mh_all_actions = np.concatenate(mh_actions, axis=1)
        return mb_obs, mb_states, mb_returns, mb_masks, mb_actions,\
               mb_values, mb_all_obs, mh_actions, mh_all_actions, mb_rewards, mb_true_rewards, mb_true_returns

def learn(policy, expert, env, env_id, seed, total_timesteps=int(40e6), gamma=0.99, lam=0.95, log_interval=1, nprocs=32,
          nsteps=20, nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=100, lrschedule='linear', dis_lr=0.001, disc_type='decentralized',
          bc_iters=500, identical=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_global_seeds(seed)
    buffer = None

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_agents = (len(ob_space))
    make_model = lambda: GeneralModel(policy, ob_space, ac_space, nenvs, total_timesteps, device, nprocs=nprocs, nsteps=nsteps,
                               nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
                               lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                               lrschedule=lrschedule, identical=identical)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    
    if disc_type == 'decentralized':
        discriminator = [
            Discriminator(ob_space, ac_space, nstack, k, device, disc_type=disc_type).to(device) for k in range(num_agents)
        ]
    elif disc_type == 'centralized':
        discriminator = Discriminator(ob_space, ac_space, nstack, 0, device, disc_type=disc_type).to(device)
    elif disc_type == 'single':
        discriminator = Discriminator(ob_space, ac_space, nstack, 0, device, disc_type=disc_type).to(device)
    else:
        assert False
        
    runner = Runner(env, model, discriminator, nsteps=nsteps, nstack=nstack, gamma=gamma, lam=lam, disc_type=disc_type)
    nbatch = nenvs * nsteps
    tstart = time.time()
    
    for update in range(1, total_timesteps // nbatch + 1):
        obs, states, rewards, masks, actions, values, all_obs,\
        mh_actions, mh_all_actions, mh_rewards, mh_true_rewards, mh_true_returns = runner.run()
        
        d_iters = 1
        g_loss, e_loss = np.zeros((num_agents, d_iters)), np.zeros((num_agents, d_iters))
        idx = 0
        idxs = np.arange(len(all_obs))
        random.shuffle(idxs)
        all_obs = all_obs[idxs]
        mh_actions = [mh_actions[k][idxs] for k in range(num_agents)]
        mh_obs = [obs[k][idxs] for k in range(num_agents)]
        mh_values = [values[k][idxs] for k in range(num_agents)]
        
        if buffer:
            buffer.update(mh_obs, mh_actions, None, all_obs, mh_values)
        else:
            buffer = Dset(mh_obs, mh_actions, None, all_obs, mh_values, randomize=True, num_agents=num_agents)

        d_minibatch = nenvs * nsteps
        
        for d_iter in range(d_iters):
            e_obs, e_actions, e_all_obs, _ = expert.get_next_batch(d_minibatch)
            g_obs, g_actions, g_all_obs, _ = buffer.get_next_batch(batch_size=d_minibatch)
            if disc_type == 'decentralized':
                for k in range(num_agents):
                    g_loss[k, d_iter], e_loss[k, d_iter] = discriminator[k].train(
                        g_obs[k],
                        g_actions[k],
                        e_obs[k],
                        e_actions[k]
                    )
                    
            elif disc_type == 'centralized':
                g_loss_t, e_loss_t = discriminator.train(
                    g_all_obs,
                    np.concatenate(g_actions, axis=1),
                    e_all_obs, np.concatenate(e_actions, axis=1))
                g_loss[:, d_iter] = g_loss_t
                e_loss[:, d_iter] = e_loss_t
                
            elif disc_type == 'single':
                g_loss_t, e_loss_t = discriminator.train(
                    g_all_obs,
                    np.concatenate(g_actions, axis=1),
                    e_all_obs, np.concatenate(e_actions, axis=1))
                g_loss[:, d_iter] = g_loss_t
                e_loss[:, d_iter] = e_loss_t
            else:
                assert False
            idx += 1
            
        if update > 10:
            policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
            
        model.old_obs = obs
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = [explained_variance(values[k], rewards[k]) for k in range(model.num_agents)]
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)

            for k in range(model.num_agents):
                logger.record_tabular("explained_variance %d" % k, float(ev[k]))
                if update > 10:
                    logger.record_tabular("policy_entropy %d" % k, float(policy_entropy[k]))
                    logger.record_tabular("policy_loss %d" % k, float(policy_loss[k]))
                    logger.record_tabular("value_loss %d" % k, float(value_loss[k]))
                    try:
                        logger.record_tabular('pearson %d' % k, float(
                            pearsonr(rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
                        logger.record_tabular('reward %d' % k, float(np.mean(rewards[k])))
                        logger.record_tabular('spearman %d' % k, float(
                            spearmanr(rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
                    except:
                        pass
            if update > 10 and env_id == 'simple_tag':
                try:
                    logger.record_tabular('in_pearson_0_2', float(
                        pearsonr(rewards[0].flatten(), rewards[2].flatten())[0]))
                    logger.record_tabular('in_pearson_1_2', float(
                        pearsonr(rewards[1].flatten(), rewards[2].flatten())[0]))
                    logger.record_tabular('in_spearman_0_2', float(
                        spearmanr(rewards[0].flatten(), rewards[2].flatten())[0]))
                    logger.record_tabular('in_spearman_1_2', float(
                        spearmanr(rewards[1].flatten(), rewards[2].flatten())[0]))
                except:
                    pass

            g_loss_m = np.mean(g_loss, axis=1)
            e_loss_m = np.mean(e_loss, axis=1)
            # g_loss_gp_m = np.mean(g_loss_gp, axis=1)
            # e_loss_gp_m = np.mean(e_loss_gp, axis=1)
            for k in range(num_agents):
                logger.record_tabular("g_loss %d" % k, g_loss_m[k])
                logger.record_tabular("e_loss %d" % k, e_loss_m[k])
                # logger.record_tabular("g_loss_gp %d" % k, g_loss_gp_m[k])
                # logger.record_tabular("e_loss_gp %d" % k, e_loss_gp_m[k])

            logger.dump_tabular()
        
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'm_%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
            if disc_type == 'decentralized':
                for k in range(num_agents):
                    savepath = osp.join(logger.get_dir(), 'd_%d_%.5i' % (k, update))
                    discriminator[k].save(savepath)
            elif disc_type == 'centralized':
                savepath = osp.join(logger.get_dir(), 'd_%.5i' % update)
                discriminator.save(savepath)
            elif disc_type == 'single':
                savepath = osp.join(logger.get_dir(), 'd_%.5i' % update)
                discriminator.save(savepath)  
    env.close()