import os

from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf
import numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque


def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    done = np.zeros(horizon, 'int32')
    stat1 = np.array([ac for _ in range(horizon)])
    stat2 = np.array([ac for _ in range(horizon)])
    stat3 = np.array([ac for _ in range(horizon)])
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, _stat1, _stat2, _stat3 = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "done": done,
                   "stat1": stat1, "stat2": stat2, "stat3": stat3}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        stat1[i] = _stat1
        stat2[i] = _stat2
        stat3[i] = _stat3

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def calculate_advantage_and_vtarg(seg, gamma, lam, k_step):
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    done = seg["done"]
    T = len(seg["rew"]) - k_step
    rew = seg["rew"]
    vtarg = np.zeros(T, 'float32')
    seg["adv"] = gae = np.zeros(T, 'float32')

    for t in range(T):
        _sum = 0
        for k in range(k_step):
            if (k == (k_step - 1)) | (done[t + k] != 0):
                nonterminal = 0
            else:
                nonterminal = 1
            delta = rew[t + k] + gamma * vpred[t + k + 1] * nonterminal - vpred[t + k]
            _sum += (gamma * lam) ** k * delta
            if nonterminal == 0:
                break
        gae[t] = _sum
        vtarg[t] = _sum + vpred[t]

    seg["tdlamret"] = seg["adv"] + seg["vpred"][:T]


def learn(env, policy_fn, *,
          timesteps_per_actorbatch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
          gae_kstep=None,
          env_eval=None,
          saved_model=None,
          eval_at=50,
          save_at=50,
          normalize_atarg=True,
          experiment_spec=None,  # dict with: experiment_name, experiment_folder
          **extra_args
          ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space)  # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])  # learning rate multiplier, updated with schedule
    entromult = tf.placeholder(name='entromult', dtype=tf.float32, shape=[])  # entropy penalty multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    MPI_n_ranks = MPI.COMM_WORLD.Get_size()

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult, entromult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult, entromult], losses)

    acts = list()
    stats1 = list()
    stats2 = list()
    stats3 = list()
    for p in range(ac_space.shape[0]):
        acts.append(tf.placeholder(tf.float32, name="act_{}".format(p + 1)))
        stats1.append(tf.placeholder(tf.float32, name="stats1_{}".format(p + 1)))
        stats2.append(tf.placeholder(tf.float32, name="stats2_{}".format(p + 1)))
        stats3.append(tf.placeholder(tf.float32, name="stats3_{}".format(p + 1)))
        tf.summary.histogram("act_{}".format(p), acts[p])
        if pi.dist == 'gaussian':
            tf.summary.histogram("pd_mean_{}".format(p), stats1[p])
            tf.summary.histogram("pd_std_{}".format(p), stats2[p])
            tf.summary.histogram("pd_logstd_{}".format(p), stats3[p])
        else:
            tf.summary.histogram("pd_beta_{}".format(p), stats1[p])
            tf.summary.histogram("pd_alpha_{}".format(p), stats2[p])
            tf.summary.histogram("pd_alpha_beta_{}".format(p), stats3[p])

    rew = tf.placeholder(tf.float32, name="rew")
    tf.summary.histogram("rew", rew)
    summaries = tf.summary.merge_all()
    gather_summaries = U.function([ob, *acts, *stats1, *stats2, *stats3, rew], summaries)

    U.initialize()
    adam.sync()
    if saved_model is not None:
        U.load_state(saved_model)

    if (MPI.COMM_WORLD.Get_rank() == 0) & (experiment_spec is not None):
        # TensorBoard & Saver
        # ----------------------------------------
        if experiment_spec['experiment_folder'] is not None:
            path_tb = os.path.join(experiment_spec['experiment_folder'], 'tensorboard')
            path_logs = os.path.join(experiment_spec['experiment_folder'], 'logs')
            exp_name = '' if experiment_spec['experiment_name'] is not None else experiment_spec['experiment_name']
            summary_file = tf.summary.FileWriter(os.path.join(path_tb, exp_name), U.get_session().graph)
            saver = tf.train.Saver(max_to_keep=None)
            logger.configure(dir=os.path.join(path_logs, exp_name))
    else:
        logger.configure(format_strs=[])

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0, max_seconds > 0]) == 1, "Only one time constraint permitted"

    while True:
        if callback:
            callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(iters_so_far) / max_iters, 0)
        elif 'exp' in schedule:
            current_lr = schedule.strip()
            _, d = current_lr.split('__')
            cur_lrmult = float(d) ** (float(iters_so_far) / max_iters)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)

        seg = seg_gen.__next__()
        if gae_kstep is None:
            add_vtarg_and_adv(seg, gamma, lam)
            T = len(seg["rew"])
        else:
            calculate_advantage_and_vtarg(seg, gamma, lam, k_step=gae_kstep)
            T = len(seg["rew"]) - gae_kstep

        ob, ac, atarg, tdlamret = seg["ob"][:T], seg["ac"][:T], seg["adv"][:T], seg["tdlamret"][:T]
        vpredbefore = seg["vpred"][:T]  # predicted value function before udpate

        if normalize_atarg:
            eps = 1e-9
            atarg = (atarg - atarg.mean()) / (atarg.std() + eps)  # standardized advantage function estimate

        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"):
            pi.ob_rms.update(ob)  # update running mean/std for policy

        assign_old_eq_new()  # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        g_max = -np.Inf
        g_min = np.Inf
        g_mean = []
        for _ in range(optim_epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, cur_lrmult)
                g_max = g.max() if g.max() > g_max else g_max
                g_min = g.min() if g.min() < g_min else g_min
                g_mean.append(g.mean())
                if np.isnan(np.sum(g)):
                    print('NaN in Gradient, skipping this update')
                    continue
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
                # logger.log(fmt_row(13, np.mean(losses, axis=0)))

        summary = tf.Summary()
        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, cur_lrmult)
            losses.append(newlosses)
        meanlosses, _, _ = mpi_moments(losses, axis=0)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, loss_names):
                logger.record_tabular("loss_" + name, lossval)
                summary.value.add(tag="loss_" + name, simple_value=lossval)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("ItersSoFar (%)", iters_so_far / max_iters * 100)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("TimePerIter", (time.time() - tstart) / (iters_so_far + 1))

        if MPI.COMM_WORLD.Get_rank() == 0:
            # Saves model
            if ((iters_so_far % save_at) == 0) & (iters_so_far != 0):
                if experiment_spec['experiment_folder'] is not None:
                    path_models = os.path.join(experiment_spec['experiment_folder'], 'models')
                    dir_path = os.path.join(path_models, exp_name)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    saver.save(U.get_session(), os.path.join(dir_path, 'model'), global_step=iters_so_far)

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()

            summ = gather_summaries(ob,
                                    *np.split(ac, ac_space.shape[0], axis=1),
                                    *np.split(seg['stat1'], ac_space.shape[0], axis=1),
                                    *np.split(seg['stat2'], ac_space.shape[0], axis=1),
                                    *np.split(seg['stat3'], ac_space.shape[0], axis=1),
                                    seg['rew'])

            summary.value.add(tag="total_loss", simple_value=meanlosses[:3].sum())
            summary.value.add(tag="explained_variance", simple_value=explained_variance(vpredbefore, tdlamret))
            summary.value.add(tag='EpRewMean', simple_value=np.mean(rewbuffer))
            summary.value.add(tag='EpLenMean', simple_value=np.mean(lenbuffer))
            summary.value.add(tag='EpThisIter', simple_value=len(lens))
            summary.value.add(tag='atarg_max', simple_value=atarg.max())
            summary.value.add(tag='atarg_min', simple_value=atarg.min())
            summary.value.add(tag='atarg_mean', simple_value=atarg.mean())
            summary.value.add(tag='GMean', simple_value=np.mean(g_mean))
            summary.value.add(tag='GMax', simple_value=g_max)
            summary.value.add(tag='GMin', simple_value=g_min)
            summary.value.add(tag='learning_rate', simple_value=cur_lrmult * optim_stepsize)
            summary.value.add(tag='AcMAX', simple_value=np.mean(seg["ac"].max()))
            summary.value.add(tag='AcMIN', simple_value=np.mean(seg["ac"].min()))
            summary_file.add_summary(summary, iters_so_far)
            summary_file.add_summary(summ, iters_so_far)

        iters_so_far += 1


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def load_policy(env, policy_func, *,
                clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
                adam_epsilon=1e-5,
                model_path, checkpoint):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space)  # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
    pol_surr = - U.mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = U.mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    U.load_state(os.path.join(model_path, 'model-{}'.format(checkpoint)))

    return pi
