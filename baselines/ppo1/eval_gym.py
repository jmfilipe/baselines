import os

import numpy as np
import yaml
import gym

from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1.pposgd_simple import load_policy
from baselines.ppo1 import mlp_policy


def evaluation(seed, model_path, checkpoint, animate, n_episodes,
               layers_val, layers_pol, distribution, entropy_pen, clip_param,
               **kwargs):

    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)

    env = gym.make('MountainCarContinuous-v0')

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    layers_val=layers_val, layers_pol=layers_pol,
                                    gaussian_fixed_var=False, dist=distribution)

    policy = load_policy(env, policy_fn, entcoeff=entropy_pen,
                         model_path=model_path,
                         checkpoint=checkpoint, clip_param=clip_param)

    paths_eval = []
    path = {}
    for ep in range(n_episodes):
        ob = env.reset()
        rews = []
        while True:
            ac = policy.act(stochastic=False, ob=ob)[0]
            ob, rew, done, _ = env.step(ac)
            if animate:
                env.render()
            rews.append(rew)
            if done:
                break
        path['reward'] = np.array(rews)
        paths_eval.append(path)


if __name__ == "__main__":

    configs_folder = r"C:\Users\CPES\PycharmProjects\restable\RL\experiments_restable\configs"
    file_name = "2018-03-16 13_18_20_PPO__[30, 25, 18]"
    CHECKPOINT = 620

    with open(os.path.join(configs_folder, file_name + '.yaml'), 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    models_folder = os.path.abspath(os.path.join(configs_folder, '..', 'models'))
    MODEL_PATH = os.path.join(models_folder, file_name)
    SEED = 1337
    ANIMATE = False
    LOGGING = True
    N_EPISODES = 10
    TEST_SPECIFIC_EPISODES = False

    NAME = "__" + 'chk_{}'.format(CHECKPOINT)

    evaluation(seed=SEED, model_path=MODEL_PATH, checkpoint=CHECKPOINT,
               animate=ANIMATE, n_episodes=N_EPISODES, **config)
