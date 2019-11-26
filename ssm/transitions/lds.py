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


class ContinuousStateTransitions(object):
    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def sample(self, states, data, input, tag, with_noise=True):
        raise NotImplementedError

    def log_transition_probs(self, states, data, input, tag):
        raise NotImplementedError

    def m_step(self, posteriors, datas, inputs, masks, tags, weights=None,
               optimizer="adam", **kwargs):
        """
        Perform an approximate m-step by maximizing the expected log likelihood
        with stochastic gradient ascent.
        """
        optimizer = dict(sgd=sgd, adam=adam, rmsprop=rmsprop, bfgs=bfgs, lbfgs=lbfgs)[optimizer]

        # Maximize the expected log joint
        def _expected_log_prob(expectations):
            lp = self.log_prior()
            for posterior, data, input, mask, tag, weight \
                in zip(posteriors, datas, inputs, masks, tags, weights):

                states = posterior.sample()
                lp += np.sum(weight * self.log_transition_probs(states, data, input, mask, tag))

            return lp

        # Normalize and negate for minimization
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_prob(expectations)
            return -obj / T

        # Call the optimizer. Persist state (e.g. SGD momentum) across calls to m_step.
        optimizer_state = self.optimizer_state if hasattr(self, "optimizer_state") else None
        self.params, self.optimizer_state = \
            optimizer(_objective, self.params, num_iters=num_iters,
                      state=optimizer_state, full_output=True, **kwargs)


class LinearGaussianTransitions(GenericTransitions):
    def __init__(self, D, M=0):
        self.D, self.M = D, M

        # Initialize model parameters
        self.A = npr.randn(D, D)
        self.B = npr.randn(D, M)
        self.b = npr.randn(D)
        self._Q_sqrt = np.eye(D)

    @property
    def Q(self):
        return np.dot(self._Q_sqrt, self._Q_sqrt.T)

    @Q.setter
    def Q(self, value):
        self._Q_sqrt = np.linalg.cholesky(value)

    @property
    def params(self):
        return (self.A, self.B, self.b, self._Q_sqrt)

    @params.setter
    def params(self, value):
        self.A , self.B, self.b, self._Q_sqrt = value

    def sample(self, states, data, input, tag, with_noise=True):
        mean = self.A @ states[-1] + self.B @ input[-1] + b
        return npr.multivariate_normal(mean, self.Q)

    def log_transition_probs(self, states, data, input, tag):
        mean = self.A @ states[-1] + self.B @ input[-1] + b
        return multivariate_normal_logpdf(states[1:], mean, self.Q)

    def m_step(self, expectations, datas, inputs, masks, tags, weights=None, **kwargs):
        """
        Perform an exact m-step given the following expectations:

        Ex     = E[(x_1, ..., x_T)]                           a (T, D) array
        ExxT   = E[(x_1 x_1^T, ..., x_T x_T^T)]               a (T, D, D) array
        Exxp1T = E[(x_1 x_2^T, ..., x_{T-1} x_T^T)]           a (T-1, D, D) array

        We cast this as a linear regression where x = x_{t-1} and y = x_t.
        """
        # Extract expectations for m-step
        Exs = [e['Ex'][:-1] for e in expectations]
        Eys = [e['Ex'][1:] for e in expectations]
        ExxTs = [e['ExxT'][:-1] for e in expectations]
        ExyTs = [e['Exxp1T'] for e in expectations]
        EyyTs = [e['ExxT'][1:] for e in expectations]

        self.A, self.B, self.b, self.Q = \
            fit_linear_regression_given_expectations(Exs, Eys, ExxTs, ExyTs, EyyTs,
                inputs=inputs, weights=weights)


class SwitchingLinearGaussianTransitions(GenericTransitions):
    """
    Combine discrete and linear Gaussian transitions.

    Note: "recurrent" transition classes typically take in the observed data,
          but here they take in the continuous latent state instead.  If you
          want to pass the data to the discrete transition model, pass it as
          an additional input.
    """
    def __init__(self, K, D, M=0,
                 discrete_transitions="standard",
                 discrete_transition_kwargs=None,
                 continuous_dynamics="gaussian",
                 continuous_dynamics_kwargs=None):

        self.K = K
        self.D = D
        self.M = M

        # Make the discrete transition model
        discrete_transition_classes = dict(
            standard=StationaryTransitions,
            stationary=StationaryTransitions,
            sticky=StickyTransitions,
            inputdriven=InputDrivenTransitions,
            recurrent=RecurrentTransitions,
            recurrent_only=RecurrentOnlyTransitions,
            rbf_recurrent=RBFRecurrentTransitions,
            nn_recurrent=NeuralNetworkRecurrentTransitions
            )

        if isinstance(discrete_transitions, str):
            if discrete_transitions not in discrete_transition_classes:
                raise Exception("Invalid discrete transition model: {}. Must be one of {}".
                    format(discrete_transitions, list(discrete_transition_classes.keys())))
            discrete_transition_kwargs = discrete_transition_kwargs or {}
            self.discrete_transitions = \
                discrete_transition_classes[discrete_transitions](
                    K, D, M=M, **discrete_transition_kwargs)

        if not isinstance(discrete_transitions, Transitions):
            raise TypeError("'discrete_transitions' must be a subclass of"
                            " ssm.transitions.Transitions")

        # Make the continuous dynamics models, one for each discrete state
        continuous_dynamics_classes = dict(
            gaussian=LinearGaussianTransitions,
            )

        assert isinstance(continuous_dynamics, str):
        if continuous_dynamics not in continuous_dynamics_classes:
            raise Exception("Invalid continuous dynamics model: {}. Must be one of {}".
                format(continuous_dynamics, list(continuous_dynamics_classes.keys())))
        continuous_dynamics_kwargs = continuous_dynamics_kwargs or {}
        self.continuous_dynamics = \
            [continuous_dynamics_classes[continuous_dynamics](
                D, M=M, **continuous_dynamics_kwargs)
             for _ in range(K)]

    def sample(self, states, data, input, tag, with_noise=True):

        discrete_states = [int(s[0]) for s in states]
        continuous_states = [s[1:] for s in states]

        next_discrete_state = self.discrete_transitions.sample(
            discrete_states, continuous_states, input, tag, with_noise=with_noise)
        next_continuous_state = self.continuous_dynamics[next_discrete_state].sample(
            continuous_states, data, input, tag, with_noise=with_noise)

        return np.concatenate([[next_discrete_state], next_continuous_state])

    def m_step(self, posterior, datas, inputs, masks, tags, **kwargs):
        self.discrete_transitions.m_step(expectations, datas, inputs, masks, tags, **kwargs)
        for k in range(self.K):
            weights = [e["Ez"][:, k] for e in expectations]
            self.continuous_dynamics[k].m_step(expectations, datas, inputs, masks, tags, weights=weights, **kwargs)
