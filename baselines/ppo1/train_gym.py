import pathlib
import sys
import datetime
import os
from collections import OrderedDict

import yaml
from mpi4py import MPI

from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple
import gym


def train(num_iters, seed, id_name, layers_val, layers_pol, saved_model,
          eval_at, save_at, timesteps_per_actorbatch, entropy_pen, opt_epochs,
          lr_sched, optim_batchsize, clip_param, gamma, lam, optim_stepsize,
          experiment_folder_fpath, distribution, gae_kstep, normalize_atarg, **kwargs):

    U.make_session(num_cpu=1).__enter__()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = gym.make('MountainCarContinuous-v0')

    env_eval = None

    experiment_spec = {
        'experiment_name': id_name,
        'experiment_folder': experiment_folder_fpath,
    }

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    layers_val=layers_val, layers_pol=layers_pol,
                                    gaussian_fixed_var=False, dist=distribution)

    pposgd_simple.learn(env, policy_fn, env_eval=env_eval, max_iters=num_iters,
                        timesteps_per_actorbatch=timesteps_per_actorbatch,
                        clip_param=clip_param, entcoeff=entropy_pen, normalize_atarg=normalize_atarg,
                        optim_epochs=opt_epochs, optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
                        gamma=gamma, lam=lam, schedule=lr_sched, gae_kstep=gae_kstep,
                        saved_model=saved_model, eval_at=eval_at,
                        save_at=save_at, experiment_spec=experiment_spec, **kwargs)


def main(config):
    datenow = datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S')
    extra = ''

    id_name = 'PPO__'
    id_name += extra
    id_name = datenow + '_' + id_name

    config['id_name'] = id_name

    pathlib.Path(os.path.join(config['experiment_folder_fpath'], 'configs')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(config['experiment_folder_fpath'], 'logs')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(config['experiment_folder_fpath'], 'models')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(config['experiment_folder_fpath'], 'tensorboard')).mkdir(parents=True, exist_ok=True)

    path_configs = os.path.join(config['experiment_folder_fpath'], 'configs')

    with open(os.path.join(path_configs, '{}.yaml'.format(id_name)), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    train(**config)


if __name__ == '__main__':

    config = OrderedDict()
    # Environment Config
    config['wind_farms'] = [30, 25, 18]
    config['pv_plants'] = []
    config['use_FRR'] = True
    config['reward_func'] = 'penalty'
    config['reward_scale'] = 200
    # PPO Config
    config['num_iters'] = 5000
    config['seed'] = 0
    config['normalize_atarg'] = True
    config['layers_val'] = [64, 64]
    config['layers_pol'] = config['layers_val']
    config['optim_batchsize'] = 64
    config['timesteps_per_actorbatch'] = config['optim_batchsize'] * 20
    config['lr_sched'] = 'exp__0.3'  # exp__xx, constant, linear
    config['optim_stepsize'] = 1e-4
    config['entropy_pen'] = 0.0
    config['opt_epochs'] = 5
    config['gae_kstep'] = 1
    config['clip_param'] = 0.2
    config['gamma'] = 0.99
    config['lam'] = 0.95
    # Miscellaneous Config
    config['eval_at'] = 20
    config['save_at'] = 20
    config['logging'] = False
    config['distribution'] = 'beta'
    config['description'] = ""
    config['saved_model'] = None  # r"C:\Users\CPES\PycharmProjects\restable\RL\experiments_restable\models\2018-03-15 18_02_17_PPO__[30, 25, 18]\model-14980"
    config['experiment_folder'] = 'experiments_autoconsumo'
    config['experiment_folder_fpath'] = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))),
                                                     config['experiment_folder'])
    for key, value in config.items():
        print("{}: {}".format(key, value))

    main(config)
