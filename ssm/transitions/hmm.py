from functools import partial
from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd.scipy.stats import dirichlet
from autograd import hessian

from ssm.util import one_hot, logistic, relu, rle, ensure_args_are_lists
from ssm.regression import fit_multiclass_logistic_regression, fit_negative_binomial_integer_r, fit_linear_regression_given_expectations
from ssm.stats import multivariate_normal_logpdf
from ssm.optimizers import adam, bfgs, lbfgs, rmsprop, sgd


class HMMTransitions(object):
    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def permute(self, perm):
        pass

    def log_transition_matrices(self, data, input, mask, tag):
        raise NotImplementedError

    def transition_matrices(self, data, input, mask, tag):
        return np.exp(self.log_transition_matrices(data, input, mask, tag))

    def sample(self, states, data, input, tag, with_noise=True):
        P = self.transition_matrices(data, input, None, tag)[-1]
        if with_noise:
            return npr.choice(self.K, p=P[states[-1]])
        else:
            return np.argmax(P[states[-1]])

    def m_step(self, expectations, datas, inputs, masks, tags,
               optimizer="lbfgs", num_iters=100, **kwargs):
        """
        If M-step cannot be done in closed form for the transitions, default to BFGS.
        """
        optimizer = dict(sgd=sgd, adam=adam, rmsprop=rmsprop, bfgs=bfgs, lbfgs=lbfgs)[optimizer]

        # Maximize the expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, e \
                in zip(datas, inputs, masks, tags, expectations):
                log_Ps = self.log_transition_matrices(data, input, mask, tag)
                elbo += np.sum(e["Ezzp1"] * log_Ps)
            return elbo

        # Normalize and negate for minimization
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        # Call the optimizer. Persist state (e.g. SGD momentum) across calls to m_step.
        optimizer_state = self.optimizer_state if hasattr(self, "optimizer_state") else None
        self.params, self.optimizer_state = \
            optimizer(_objective, self.params, num_iters=num_iters,
                      state=optimizer_state, full_output=True, **kwargs)

    def hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        warn("Analytical Hessian is not implemented for this transition class. \
              Optimization via Laplace-EM may be slow. Consider using an \
              alternative posterior and inference method.")
        T, D = data.shape
        obj = lambda x, E_zzp1: np.sum(E_zzp1 * self.log_transition_matrices(x, input, mask, tag))
        hess = hessian(obj)
        terms = np.array([hess(x[None,:], Ezzp1) for x, Ezzp1 in zip(data, expected_joints)])
        return terms


class StationaryTransitions(HMMTransitions):
    """
    Standard Hidden Markov Model with fixed initial distribution and transition matrix.
    """
    def __init__(self, K, D, M=0):
        super(StationaryTransitions, self).__init__(K, D, M=M)
        Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

    @property
    def params(self):
        return (self.log_Ps,)

    @params.setter
    def params(self, value):
        self.log_Ps = value[0]

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]

    @property
    def transition_matrix(self):
        return np.exp(self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True))

    def log_transition_matrices(self, data, input, mask, tag):
        T = len(data)
        log_Ps = self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True)
        # return np.tile(log_Ps[None, :, :], (T-1, 1, 1))
        return log_Ps[None, :, :]

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        P = sum([np.sum(e["Ezzp1"], axis=0) for e in expectations]) + 1e-16
        P /= P.sum(axis=-1, keepdims=True)
        self.log_Ps = np.log(P)

    def hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))


class StickyTransitions(StationaryTransitions):
    """
    Upweight the self transition prior.

    pi_k ~ Dir(alpha + kappa * e_k)
    """
    def __init__(self, K, D, M=0, alpha=1, kappa=100):
        super(StickyTransitions, self).__init__(K, D, M=M)
        self.alpha = alpha
        self.kappa = kappa

    def log_prior(self):
        K = self.K
        Ps = np.exp(self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True))

        lp = 0
        for k in range(K):
            alpha = self.alpha * np.ones(K) + self.kappa * (np.arange(K) == k)
            lp += dirichlet.logpdf(Ps[k], alpha)
        return lp

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        expected_joints = sum([np.sum(e["Ezzp1"], axis=0) for e in expectations]) + 1e-8
        expected_joints += self.kappa * np.eye(self.K)
        P = expected_joints / expected_joints.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(P)

    def hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))


class InputDrivenTransitions(StickyTransitions):
    """
    Hidden Markov Model whose transition probabilities are
    determined by a generalized linear model applied to the
    exogenous input.
    """
    def __init__(self, K, D, M, alpha=1, kappa=0, l2_penalty=0.0):
        super(InputDrivenTransitions, self).__init__(K, D, M=M, alpha=alpha, kappa=kappa)

        # Parameters linking input to state distribution
        self.Ws = npr.randn(K, M)

        # Regularization of Ws
        self.l2_penalty = l2_penalty

    @property
    def params(self):
        return self.log_Ps, self.Ws

    @params.setter
    def params(self, value):
        self.log_Ps, self.Ws = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.Ws = self.Ws[perm]

    def log_prior(self):
        lp = super(InputDrivenTransitions, self).log_prior()
        lp = lp + np.sum(-0.5 * self.l2_penalty * self.Ws**2)
        return lp

    def log_transition_matrices(self, data, input, mask, tag):
        T = data.shape[0]
        assert input.shape[0] == T
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

    def hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))


class RecurrentTransitions(InputDrivenTransitions):
    """
    Generalization of the input driven HMM in which the observations serve as future inputs
    """
    def __init__(self, K, D, M=0, alpha=1, kappa=0):
        super(RecurrentTransitions, self).__init__(K, D, M, alpha=alpha, kappa=kappa)

        # Parameters linking past observations to state distribution
        self.Rs = np.zeros((K, D))

    @property
    def params(self):
        return super(RecurrentTransitions, self).params + (self.Rs,)

    @params.setter
    def params(self, value):
        self.Rs = value[-1]
        super(RecurrentTransitions, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        super(RecurrentTransitions, self).permute(perm)
        self.Rs = self.Rs[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        # Past observations effect
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

    def hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        for k in range(self.K):
            vtilde = vtildes[:,k,:] # normalized probabilities given state k
            Rv = vtilde @ self.Rs
            hess += Ez[:,k][:,None,None] * \
                    ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs, self.Rs) \
                    + np.einsum('ti, tj -> tij', Rv, Rv))
        return hess


class RecurrentOnlyTransitions(HMMTransitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0):
        super(RecurrentOnlyTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = npr.randn(K, M)
        self.Rs = npr.randn(K, D)
        self.r = npr.randn(K)

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs, self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape
        log_Ps = np.dot(input[1:], self.Ws.T)[:, None, :]              # inputs
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]     # past observations
        log_Ps = log_Ps + self.r                                       # bias
        log_Ps = np.tile(log_Ps, (1, self.K, 1))                       # expand
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        HMMTransitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

    def hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        v = np.dot(input[1:], self.Ws.T) + np.dot(data[:-1], self.Rs.T) + self.r
        shifted_exp = np.exp(v - np.max(v,axis=1,keepdims=True))
        vtilde = shifted_exp / np.sum(shifted_exp,axis=1,keepdims=True) # normalized probabilities
        Rv = vtilde@self.Rs
        return np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs, self.Rs) \
               + np.einsum('ti, tj -> tij', Rv, Rv)


class RBFRecurrentTransitions(InputDrivenTransitions):
    """
    Recurrent transitions with radial basis functions for parameterizing
    the next state probability given current continuous data. We have,

    p(z_{t+1} = k | z_t, x_t)
        \propto N(x_t | \mu_k, \Sigma_k) \times \pi_{z_t, z_{t+1})

    where {\mu_k, \Sigma_k, \pi_k}_{k=1}^K are learned parameters.
    Equivalently,

    log p(z_{t+1} = k | z_t, x_t)
        = log N(x_t | \mu_k, \Sigma_k) + log \pi_{z_t, z_{t+1}) + const
        = -D/2 log(2\pi) -1/2 log |Sigma_k|
          -1/2 (x - \mu_k)^T \Sigma_k^{-1} (x-\mu_k)
          + log \pi{z_t, z_{t+1}}

    The difference between this and the recurrent model above is that the
    log transition matrices are quadratic functions of x rather than linear.

    While we're at it, there's no harm in adding a linear term to the log
    transition matrices to capture input dependencies.
    """
    def __init__(self, K, D, M=0, alpha=1, kappa=0):
        super(RBFRecurrentTransitions, self).__init__(K, D, M=M, alpha=alpha, kappa=kappa)

        # RBF parameters
        self.mus = npr.randn(K, D)
        self._sqrt_Sigmas = npr.randn(K, D, D)

    @property
    def params(self):
        return self.log_Ps, self.mus, self._sqrt_Sigmas, self.Ws

    @params.setter
    def params(self, value):
        self.log_Ps, self.mus, self._sqrt_Sigmas, self.Ws = value

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Fit a GMM to the data to set the means and covariances
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(self.K, covariance_type="full")
        gmm.fit(np.vstack(datas))
        self.mus = gmm.means_
        self._sqrt_Sigmas = np.linalg.cholesky(gmm.covariances_)

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.mus = self.mus[perm]
        self.sqrt_Sigmas = self.sqrt_Sigmas[perm]
        self.Ws = self.Ws[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        assert np.all(mask), "Recurrent models require that all data are present."

        T = data.shape[0]
        assert input.shape[0] == T
        K, D = self.K, self.D

        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))

        # RBF recurrent function
        rbf = multivariate_normal_logpdf(data[:-1, None, :], self.mus, self.Sigmas)
        log_Ps = log_Ps + rbf[:, None, :]

        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        HMMTransitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)


# Allow general nonlinear emission models with neural networks
class NeuralNetworkRecurrentTransitions(HMMTransitions):
    def __init__(self, K, D, M=0, hidden_layer_sizes=(50,), nonlinearity="relu"):
        super(NeuralNetworkRecurrentTransitions, self).__init__(K, D, M=M)

        # Baseline transition probabilities
        Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

        # Initialize the NN weights
        layer_sizes = (D + M,) + hidden_layer_sizes + (K,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

        nonlinearities = dict(
            relu=relu,
            tanh=np.tanh,
            sigmoid=logistic)
        self.nonlinearity = nonlinearities[nonlinearity]

    @property
    def params(self):
        return self.log_Ps, self.weights, self.biases

    @params.setter
    def params(self, value):
        self.log_Ps, self.weights, self.biases = value

    def permute(self, perm):
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.weights[-1] = self.weights[-1][:,perm]
        self.biases[-1] = self.biases[-1][perm]

    def log_transition_matrices(self, data, input, mask, tag):
        # Pass the data and inputs through the neural network
        x = np.hstack((data[:-1], input[1:]))
        for W, b in zip(self.weights, self.biases):
            y = np.dot(x, W) + b
            x = self.nonlinearity(y)

        # Add the baseline transition biases
        log_Ps = self.log_Ps[None, :, :] + y[:, None, :]

        # Normalize
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=100, **kwargs):
        # Default to adam instead of bfgs for the neural network model.
        HMMTransitions.m_step(self, expectations, datas, inputs, masks, tags,
            optimizer=optimizer, num_iters=num_iters, **kwargs)
