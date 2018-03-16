from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, layers_val, layers_pol, gaussian_fixed_var=True,
              dist='gaussian', ):
        assert isinstance(ob_space, gym.spaces.Box)

        self.dist = dist
        self.pdtype = pdtype = make_pdtype(ac_space, dist=dist)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i, size in enumerate(layers_val):
                last_out = tf.nn.relu(tf.layers.dense(last_out, size, name="fc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:, 0]

        with tf.variable_scope('pol'):
            last_out = obz
            for i, size in enumerate(layers_pol):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, size, name='fc%i' % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0] // 2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        if dist == 'gaussian':
            self._act = U.function([stochastic, ob], [ac, self.vpred, self.pd.std, self.pd.mean, self.pd.logstd])
        elif dist == 'beta':
            self._act = U.function([stochastic, ob], [ac, self.vpred, self.pd.alpha, self.pd.beta, self.pd.alpha_beta])

    def act(self, stochastic, ob):
        ac1, vpred1, stat1, stat2, stat3 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0], stat1[0], stat2[0], stat3[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
