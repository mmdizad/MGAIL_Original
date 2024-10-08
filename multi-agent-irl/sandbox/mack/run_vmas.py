#!/usr/bin/env python
import itertools
import logging
import os
import sys

import click

# relative_path = sys.path[0]
# relative_path = relative_path.replace('multi-agent-irl', 'multi-agent-particle-envs')
# sys.path.append(relative_path)
from rl import bench, logger
from rl.common import set_global_seeds
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
from sandbox.mack.acktr_disc import learn
from sandbox.mack.policies import GaussianPolicy
from vmas import make_env

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu):
    def create_env(rank):
        def _thunk():
            env = make_env(
                scenario=env_id,
                num_envs=1,
                device='cpu',
                continuous_actions=True,
                seed=(seed + rank),
            )
            # env.seed(seed + rank)
            print(env)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            return env
        return _thunk

    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])
    
    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)
    policy_fn = GaussianPolicy
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.00, identical=make_env.get_identical(env_id), save_interval=1000)
    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='./results/target_model')
# @click.option('--env', type=click.STRING, default='simple_world_comm')
@click.option('--env', type=click.STRING, default='waterfall')
@click.option('--lr', type=click.FLOAT, default=0.1)
@click.option('--seed', type=click.INT, default=1)
@click.option('--batch_size', type=click.INT, default=1000)
def main(logdir, env, lr, seed, batch_size):
    env_ids = [env]
    lrs = [lr]
    seeds = [seed]
    batch_sizes = [batch_size]

    print('logging to: ' + logdir)

    for env_id, seed, lr, batch_size in itertools.product(env_ids, seeds, lrs, batch_sizes):
        train(logdir + '/' + env_id + '/l-{}-b-{}/seed-{}'.format(lr, batch_size, seed),
              env_id, 5e7, lr, batch_size, seed, batch_size // 250)


if __name__ == "__main__":
    main()
