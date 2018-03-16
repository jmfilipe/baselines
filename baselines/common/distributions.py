from math import log

import tensorflow as tf
import numpy as np
import baselines.common.tf_util as U
from tensorflow.python.ops import math_ops


class Pd(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return - self.neglogp(x)


class PdType(object):
    """
    Parametrized family of probability distributions
    """

    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape + self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(), name=name)


class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat

    def pdclass(self):
        return CategoricalPd

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int32


class MultiCategoricalPdType(PdType):
    def __init__(self, nvec):
        self.ncats = nvec

    def pdclass(self):
        return MultiCategoricalPd

    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.ncats, flat)

    def param_shape(self):
        return [sum(self.ncats)]

    def sample_shape(self):
        return [len(self.ncats)]

    def sample_dtype(self):
        return tf.int32


class BetaPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return BetaPd

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return DiagGaussianPd

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return BernoulliPd

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.int32


# WRONG SECOND DERIVATIVES
# class CategoricalPd(Pd):
#     def __init__(self, logits):
#         self.logits = logits
#         self.ps = tf.nn.softmax(logits)
#     @classmethod
#     def fromflat(cls, flat):
#         return cls(flat)
#     def flatparam(self):
#         return self.logits
#     def mode(self):
#         return U.argmax(self.logits, axis=-1)
#     def logp(self, x):
#         return -tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, x)
#     def kl(self, other):
#         return tf.nn.softmax_cross_entropy_with_logits(other.logits, self.ps) \
#                 - tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def entropy(self):
#         return tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def sample(self):
#         u = tf.random_uniform(tf.shape(self.logits))
#         return U.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=one_hot_actions)

    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class MultiCategoricalPd(Pd):
    def __init__(self, nvec, flat):
        self.flat = flat
        self.categoricals = list(map(CategoricalPd, tf.split(flat, nvec, axis=-1)))

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)

    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])

    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])

    def sample(self):
        return tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)

    @classmethod
    def fromflat(cls, flat):
        raise NotImplementedError


class BetaPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        alpha, beta = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)

        # Softplus to ensure alpha and beta >= 1
        # epsilon < 1.0, hence negative
        epsilon = 1e-6
        max_a_b = np.int32(25)
        log_eps = log(epsilon)

        alpha = tf.clip_by_value(t=alpha, clip_value_min=log_eps, clip_value_max=max_a_b)
        self.alpha = tf.log(x=(tf.exp(x=alpha) + 1.0)) + 1.0

        beta = tf.clip_by_value(t=beta, clip_value_min=log_eps, clip_value_max=max_a_b)
        self.beta = tf.log(x=(tf.exp(x=beta) + 1.0)) + 1.0

        # shape = (-1,) + self.shape
        # alpha = tf.reshape(tensor=alpha, shape=shape)
        # beta = tf.reshape(tensor=beta, shape=shape)

        self.alpha_beta = tf.maximum(x=(self.alpha + self.beta), y=epsilon)
        self.log_norm = tf.lgamma(x=self.alpha) + tf.lgamma(x=self.beta) - tf.lgamma(x=self.alpha_beta)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.beta / self.alpha_beta

    def logp(self, x):
        action = tf.minimum(x=x, y=(1.0 - 1e-6))
        logps = (self.beta - 1.0) * tf.log(x=tf.maximum(x=action, y=1e-6)) + \
                (self.alpha - 1.0) * tf.log1p(x=-action) - self.log_norm
        return tf.reduce_mean(input_tensor=logps, axis=1)

    def kl(self, other):
        return other.log_norm - self.log_norm - tf.digamma(x=self.beta) * (other.beta - self.beta) - \
               tf.digamma(x=self.alpha) * (other.alpha - self.alpha) + tf.digamma(x=self.alpha_beta) * \
               (other.alpha_beta - self.alpha_beta)

    def entropy(self):
        return self.log_norm - (self.beta - 1.0) * tf.digamma(x=self.beta) - (self.alpha - 1.0) * tf.digamma(x=self.alpha) + \
               (self.alpha_beta - 2.0) * tf.digamma(x=self.alpha_beta)

    def sample(self):
        # Non-deterministic: sample action using gamma distribution
        alpha_sample = tf.random_gamma(shape=(), alpha=self.alpha)
        beta_sample = tf.random_gamma(shape=(), alpha=self.beta)

        sampled = beta_sample / tf.maximum(x=(alpha_sample + beta_sample), y=1e-6)

        return sampled

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class BernoulliPd(Pd):
    def __init__(self, logits):
        self.logits = logits
        self.ps = tf.sigmoid(logits)

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.round(self.ps)

    def neglogp(self, x):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.to_float(x)), axis=-1)

    def kl(self, other):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits, labels=self.ps), axis=-1) - tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=-1)

    def entropy(self):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.ps))
        return tf.to_float(math_ops.less(u, self.ps))

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


def make_pdtype(ac_space, dist='gaussian'):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        if dist == 'gaussian':
            return DiagGaussianPdType(ac_space.shape[0])
        elif dist == 'beta':
            return BetaPdType(ac_space.shape[0])
        else:
            raise NotImplemented('Distribution "{}" not implemented'.format(dist))
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPdType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.n)
    else:
        raise NotImplementedError


def shape_el(v, i):
    maybe = v.get_shape()[i]
    if maybe is not None:
        return maybe
    else:
        return tf.shape(v)[i]


@U.in_session
def test_probtypes():
    np.random.seed(0)

    pdparam_diag_gauss = np.array([-.2, .3, .4, -.5, .1, -.5, .1, 0.8])
    diag_gauss = DiagGaussianPdType(pdparam_diag_gauss.size // 2)  # pylint: disable=E1101
    validate_probtype(diag_gauss, pdparam_diag_gauss)

    pdparam_categorical = np.array([-.2, .3, .5])
    categorical = CategoricalPdType(pdparam_categorical.size)  # pylint: disable=E1101
    validate_probtype(categorical, pdparam_categorical)

    nvec = [1, 2, 3]
    pdparam_multicategorical = np.array([-.2, .3, .5, .1, 1, -.1])
    multicategorical = MultiCategoricalPdType(nvec)  # pylint: disable=E1101
    validate_probtype(multicategorical, pdparam_multicategorical)

    pdparam_bernoulli = np.array([-.2, .3, .5])
    bernoulli = BernoulliPdType(pdparam_bernoulli.size)  # pylint: disable=E1101
    validate_probtype(bernoulli, pdparam_bernoulli)


def validate_probtype(probtype, pdparam):
    N = 100000
    # Check to see if mean negative log likelihood == differential entropy
    Mval = np.repeat(pdparam[None, :], N, axis=0)
    M = probtype.param_placeholder([N])
    X = probtype.sample_placeholder([N])
    pd = probtype.pdfromflat(M)
    calcloglik = U.function([X, M], pd.logp(X))
    calcent = U.function([M], pd.entropy())
    Xval = tf.get_default_session().run(pd.sample(), feed_dict={M: Mval})
    logliks = calcloglik(Xval, Mval)
    entval_ll = - logliks.mean()  # pylint: disable=E1101
    entval_ll_stderr = logliks.std() / np.sqrt(N)  # pylint: disable=E1101
    entval = calcent(Mval).mean()  # pylint: disable=E1101
    assert np.abs(entval - entval_ll) < 3 * entval_ll_stderr  # within 3 sigmas

    # Check to see if kldiv[p,q] = - ent[p] - E_p[log q]
    M2 = probtype.param_placeholder([N])
    pd2 = probtype.pdfromflat(M2)
    q = pdparam + np.random.randn(pdparam.size) * 0.1
    Mval2 = np.repeat(q[None, :], N, axis=0)
    calckl = U.function([M, M2], pd.kl(pd2))
    klval = calckl(Mval, Mval2).mean()  # pylint: disable=E1101
    logliks = calcloglik(Xval, Mval2)
    klval_ll = - entval - logliks.mean()  # pylint: disable=E1101
    klval_ll_stderr = logliks.std() / np.sqrt(N)  # pylint: disable=E1101
    assert np.abs(klval - klval_ll) < 3 * klval_ll_stderr  # within 3 sigmas
    print('ok on', probtype, pdparam)
