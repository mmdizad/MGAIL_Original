import logging
import os
import itertools
import click
import gym
import sys
relative_path = sys.path[0]
relative_path = relative_path.replace('multi-agent-irl', 'multi-agent-particle-envs')
sys.path.append(relative_path)
import make_env
from rl import bench
from rl import logger
from rl.common.torch_util import set_global_seeds
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
from irl.dataset import MADataSet
from irl.mack.gail_torch import learn
from sandbox.mack.policies_torch import CategoricalPolicy

def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu, expert_path,
          traj_limitation, ret_threshold, dis_lr, disc_type='decentralized', bc_iters=500, weight_decay=1e-4,
          d_iters=1, ent_coef=0.005):
    def create_env(rank):
        def _thunk():
            env = make_env.make_env(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])

    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)
    print(num_cpu)
    policy_fn = CategoricalPolicy
    expert = MADataSet(expert_path, ret_threshold=ret_threshold, traj_limitation=traj_limitation)
    learn(policy_fn, expert, env, env_id, seed, total_timesteps=int(num_timesteps * 1.1),
     nprocs=num_cpu, nsteps=timesteps_per_batch // num_cpu, lr=lr, vf_coef=1,
        ent_coef=ent_coef, dis_lr=dis_lr, disc_type=disc_type, bc_iters=bc_iters,
        identical=make_env.get_identical(env_id), d_iters=d_iters, weight_decay=weight_decay)
    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='./atlas/tmp')
@click.option('--env', type=click.STRING, default='simple_spread')
@click.option('--expert_path', type=click.STRING,
              default='./atlas/model/exps/mack/simple_spread/l-0.1-b-1000/seed-1/checkpoint50000-50000tra.pkl')
@click.option('--atlas', is_flag=True, flag_value=True)
@click.option('--seed', type=click.INT, default=1)
@click.option('--traj_limitation', type=click.INT, default=200)
@click.option('--ret_threshold', type=click.FLOAT, default=-10)
@click.option('--dis_lr', type=click.FLOAT, default=1e-4)
@click.option('--disc_type', type=click.Choice(['decentralized', 'centralized', 'single']), default='decentralized')
@click.option('--bc_iters', type=click.INT, default=500)
def main(logdir, env, expert_path, atlas, seed, traj_limitation, ret_threshold, dis_lr, disc_type, bc_iters):
    env_ids = [env]
    lrs = [1e-4, 1e-5]
    dis_lrs = [1e-4, 1e-5]
    seeds = [1]
    batch_sizes = [1000]
    weight_decays = [1e-4, 1e-5]
    bc_iters = 500
    d_iters = 1
    num_timesteps = 5e7
    ent_coef = 0.005

    for env_id, seed, lr, dis_lr, batch_size, weight_decay in itertools.product(env_ids, seeds, lrs, dis_lrs, batch_sizes, weight_decays):
        train(logdir + '/gail/' + env_id + '/' + disc_type + '/s-{}/l-{}-b-{}-d-{}-c-{}-w-{}/seed-{}'.format(
              traj_limitation, lr, batch_size, dis_lr, bc_iters, seed, weight_decay),
              env_id, num_timesteps, lr, batch_size, seed, batch_size // 250, expert_path,
              traj_limitation, ret_threshold, dis_lr, disc_type=disc_type, bc_iters=bc_iters, 
              weight_decay=weight_decay, d_iters=d_iters, ent_coef=ent_coef)


if __name__ == "__main__":
    main()

