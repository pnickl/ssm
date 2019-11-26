import autograd.numpy as np
import autograd.numpy.random as npr

import ssm.messages
from ssm.util import replicate, collapse
from ssm.optimizers import lbfgs

class Posterior(object):
    """
    Base class for a posterior distribution over latent states given data x
    and parameters theta.

        p(z | x; theta) = p(z, x; theta) / p(x; theta)

    where z is a latent variable and x is the observed data.
    """
    def __init__(self, model, data, input=None, mask=None, tag=None):
        """
        Initialize the posterior with a ref to the model and datas,
        where datas is a list of data arrays.
        """
        self.model = model
        self.data = data
        self.input = input
        self.mask = mask
        self.tag = tag
        self.T = data.shape[0]

    @property
    def expectations(self):
        """
        Return posterior expectations of the latent states given the data
        """
        raise NotImplementedError

    @property
    def mode(self):
        """
        Return posterior mode of the latent states given the data
        """
        raise NotImplementedError

    @property
    def marginal_likelihood(self):
        """
        Compute (or approximate) the marginal likelihood of the data p(x; theta).
        For simple models like HMMs and LDSs, this will be exact.  For more
        complex models, like SLDS and rSLDS, this will be approximate.
        """
        raise NotImplementedError

    def sample(self, num_samples=1):
        """
        Return samples from p(z | x; theta)
        """
        raise NotImplemented

    def update(self):
        """
        Update the posterior distribution given the model parameters.
        """
        raise NotImplementedError


class HMMExactPosterior(Posterior):
    """
    Exact posterior distribution for a hidden Markov model found via message passing.
    """
    def __init__(self, model, data, input=None, mask=None, tag=None):
        super(HMMExactPosterior, self).__init__(model, data, input, mask, tag)

    @property
    def _model_params(self):
        model = self.model
        data = self.data
        input = self.input
        mask = self.mask
        tag = self.tag

        pi0 = model.initial_state_distn.initial_state_distn(data, input, mask, tag)
        Ps = model.transition_distn.transition_matrices(data, input, mask, tag)
        log_likes = model.observation_distn.log_likelihoods(data, input, mask, tag)
        return pi0, Ps, log_likes

    @property
    def expectations(self):
        Ez, Ezzp1, ll = ssm.messages.hmm_expected_states(*self._model_params)
        return dict(Ez=Ez, Ezzp1=Ezzp1)

    @property
    def marginal_likelihood(self):
        return ssm.messages.hmm_normalizer(*self._model_params)

    @property
    def mode(self):
        return ssm.messages.viterbi(*self._model_params)

    def sample(self, num_samples=1):
        if num_samples == 1:
            return ssm.messages.hmm_sample(*self._model_params)
        else:
            params = self._model_params
            return np.array([ssm.messages.hmm_sample(*params) for _ in range(num_samples)])

    def filter(self):
        return ssm.messages.hmm_filter(*self._model_params)

    def denoise(self):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return self.model.observation_distn.smooth(self.expectations["Ez"], self.data, self.input, self.tag)

    def E_step(self):
        Ez, Ezzp1, ll = ssm.messages.hmm_expected_states(*self._model_params)
        return dict(Ez=Ez, Ezzp1=Ezzp1), ll


class HMMGibbsPosterior(HMMExactPosterior):
    """
    Inherit all the nice message passing routines from the exact posterior, but
    replace the update step with a sample from the conditional distribution over
    latent states.  Upstream models can use these samples for parameter updates.
    """
    def __init__(self, model, data, input=None, mask=None, tag=None):
        super(HMMGibbsPosterior, self).__init__(model, data, input, mask, tag)

        # Save samples of the posterior distribution
        self._samples = []

    @property
    def samples(self):
        return self._samples

    def clear_caches(self):
        super(HMMGibbsPosterior, self).clear_caches()
        self._samples = []

    def update(self):
        # Draw one sample of the state sequence and append it to the list
        self._samples.append(self.sample())



class HSMMExactPosterior(Posterior):
    """
    Exact posterior distribution for a hidden semi-Markov model found via message passing.
    """
    def __init__(self, model, data, input=None, mask=None, tag=None):
        super(HSMMExactPosterior, self).__init__(model, data, input, mask, tag)

    @property
    def state_map(self):
        return self.model.state_map

    @property
    def _model_params(self):
        model = self.model
        data = self.data
        input = self.input
        mask = self.mask
        tag = self.tag
        state_map = self.state_map

        pi0 = model.initial_state_distn.initial_state_distn(data, input, mask, tag)
        Ps = model.transition_distn.transition_matrices(data, input, mask, tag)
        log_likes = model.observation_distn.log_likelihoods(data, input, mask, tag)
        return replicate(pi0, state_map), Ps, replicate(log_likes, state_map)

    @property
    def expectations(self):
        # Run message passing on the replicated state space, then collapse
        state_map = self.state_map
        Ez, Ezzp1, ll = ssm.messages.hmm_expected_states(*self._model_params)
        Ez = collapse(Ez, state_map)
        Ezzp1 = collapse(collapse(Ezzp1, state_map, axis=2), state_map, axis=1)
        return dict(Ez=Ez, Ezzp1=Ezzp1)

    @property
    def marginal_likelihood(self):
        return ssm.messages.hmm_normalizer(*self._model_params)

    @property
    def mode(self):
        return self.state_map[ssm.messages.viterbi(*self._model_params)]

    def sample(self, num_samples=1):
        if num_samples == 1:
            return self.state_map[ssm.messages.hmm_sample(*self._model_params)]
        else:
            params = self._model_params
            state_map = self.state_map
            return np.array([state_map[ssm.messages.hmm_sample(*params)] for _ in range(num_samples)])

    def filter(self):
        return ssm.messages.hmm_filter(*self._model_params)

    def denoise(self):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return self.model.observations.smooth(self.expectations["Ez"], self.data, self.input, self.tag)

    def E_step(self):
        state_map = self.state_map
        Ez, Ezzp1, ll = ssm.messages.hmm_expected_states(*self._model_params)
        z_smpl = self.sample()

        # Collapse the expectations and samples
        Ez = collapse(Ez, state_map)
        Ezzp1 = collapse(collapse(Ezzp1, state_map, axis=2), state_map, axis=1)
        z_smpl = state_map[z_smpl]
        return dict(Ez=Ez, Ezzp1=Ezzp1, z_smpl=z_smpl), ll


class GaussianLDSExactPosterior(Posterior):
    """
    Exact posterior distribution for a linear dynamical system with Gaussian emissions.
    """
    def __init__(self, model, data, input=None, mask=None, tag=None):
        super(GaussianLDSExactPosterior, self).__init__(model, data, input, mask, tag)
        self._augmented_input = np.column_stack((self.input, np.ones(self.T)))

    @property
    def _model_params(self):
        model = self.model
        mu0 = model.initial_state_distn.mu_init
        S0 = model.initial_state_distn.Sigma_init
        A = model.transition_distn.A
        B = np.column_stack((model.transition_distn.B, model.transition_distn.b))
        Q = model.transition_distn.Q
        C = model.observation_distn.C
        D = np.column_stack((model.observation_distn.D, model.observation_distn.d))
        R = model.observation_distn.R
        u = self._augmented_input
        y = self.data

        # TODO: Account for missing data by changing R

        return mu0, S0, A, B, Q, C, D, R, u, y

    @property
    def expectations(self):
        ll, Ex, Covx, Exxp1T = ssm.messages.kalman_smoother(*self._model_params)
        ExxT = Covx + Ex[:, :, None] * Ex[:, None, :]
        return dict(Ex=Ex, ExxT=ExxT, Covx=Covx, Exxp1T=Exxp1T)

    @property
    def marginal_likelihood(self):
        return ssm.messages.kalman_filter(*self._model_params)[0]

    @property
    def mode(self):
        _, Ex, _, _ = ssm.messages.kalman_smoother(*self._model_params)
        return Ex

    def sample(self, num_samples=1):
        if num_samples == 1:
            return ssm.messages.kalman_sample(*self._model_params)
        else:
            params = self._model_params
            return np.array([ssm.messages.kalman_sample(*params) for _ in range(num_samples)])

    def filter(self):
        return ssm.messages.kalman_filter(*self._model_params)

    def denoise(self):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return self.model.observations.smooth(self.expectations["Ex"], self.data, self.input, self.tag)

    def E_step(self):
        ll, Ex, Covx, Exxp1T = ssm.messages.kalman_smoother(*self._model_params)
        ExxT = Covx + Ex[:, :, None] * Ex[:, None, :]
        return dict(Ex=Ex, ExxT=ExxT, Covx=Covx, Exxp1T=Exxp1T), ll


class LDSLaplaceApproxPosterior(Posterior):
    """
    Laplace approximation to the posterior distribution in a linear dynamical system.
    This should work particularly well when the observation likelihood is log concave
    in the continuous latent states.
    """
    def __init__(self, model, data, input=None, mask=None, tag=None):
        super(LDSLaplaceApproxPosterior, self).__init__(model, data, input, mask, tag)
        self._augmented_input = np.column_stack((self.input, np.ones(self.T)))

    @property
    def expectations(self):
        ll, Ex, Covx, Exxp1T = ssm.messages.kalman_smoother(*self._model_params)
        ExxT = Covx + Ex[:, :, None] * Ex[:, None, :]
        return dict(Ex=Ex, ExxT=ExxT, Covx=Covx, Exxp1T=Exxp1T)

    @property
    def marginal_likelihood(self):
        return ssm.messages.kalman_filter(*self._model_params)[0]

    @property
    def mode(self):
        _, Ex, _, _ = ssm.messages.kalman_smoother(*self._model_params)
        return Ex

    def sample(self, num_samples=1):
        if num_samples == 1:
            return ssm.messages.kalman_sample(*self._model_params)
        else:
            params = self._model_params
            return np.array([ssm.messages.kalman_sample(*params) for _ in range(num_samples)])

    def filter(self):
        return ssm.messages.kalman_filter(*self._model_params)

    def denoise(self):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return self.model.observations.smooth(self.expectations["Ex"], self.data, self.input, self.tag)

    def E_step(self, optimizer="newton", maxiter=100, tolerance=1e-4):
        """
        First update the Laplace approximation, then run the Kalman smoother to get
        the posterior expectations.
        """

        # Compute the log joint
        def negative_log_joint(x, scale=1):
            # The "mask" for x is all ones
            x_mask = np.ones_like(x, dtype=bool)
            lp = self.model.transition_distn.log_likelihoods(x, self.input, x_mask, self.tag)
            lp += self.model.observation_distn.log_likelihoods(self.data, self.input, self.mask, self.tag, x)
            return -1 * lp / scale

        # We'll need the gradient of the expected log joint wrt x
        grad_negative_log_joint = grad(negative_log_joint)

        # We also need the hessian of the of the expected log joint
        def hessian_negative_log_joint(x, scale=1):
            T, D = np.shape(x)
            x_mask = np.ones((T, D), dtype=bool)
            hessian_diag, hessian_lower_diag = self.dynamics.hessian_expected_log_dynamics_prob(Ez, x, input, x_mask, tag)
            hessian_diag[:-1] += self.transitions.hessian_expected_log_trans_prob(x, input, x_mask, tag, Ezzp1)
            hessian_diag += self.emissions.hessian_log_emissions_prob(data, input, mask, tag, x)

            # The Hessian of the log probability should be *negative* definite since we are *maximizing* it.
            hessian_diag -= 1e-8 * np.eye(D)

            # Return the scaled negative hessian, which is positive definite
            return -1 * hessian_diag / scale, -1 * hessian_lower_diag / scale

        # Run Newton's method for each data array to find a Laplace approximation for q(x)
        # Use Newton's method or LBFGS to find the argmax of the expected log joint
        x0 = self.mean_continuous_states
        scale = x0.size

        if optimizer == "newton":
            # Run Newtons method
            x = newtons_method_block_tridiag_hessian(
                x0, negative_log_joint,
                grad_negative_log_joint,
                hessian_negative_log_joint,
                tolerance=tolerance,
                maxiter=maxiter)

        elif optimizer == "lbfgs":
            # use LBFGS
            _objective = lambda params, itr: negative_log_joint(params, scale=scale)
            x = lbfgs(_objective, x0, num_iters=maxiter, tol=tolerance)

        else:
            raise Exception("Invalid optimizer {}. Valid options are 'newton' and 'lbfgs'.".format(optimizer))


        # Evaluate the Hessian at the mode
        assert np.all(np.isfinite(obj(x)))
        J_diag, J_lower_diag = hessian_neg_expected_log_joint(x, Ez, Ezzp1)

        # Compute the Hessian vector product h = J * x = -H * x
        # We can do this without instantiating the full matrix
        h = symm_block_tridiag_matmul(J_diag, J_lower_diag, x)

        # Run the smoother to compute expectations
        ll, Ex, Covx, Exxp1T = ssm.messages.kalman_info_smoother(J, h)
        ExxT = Covx + Ex[:, :, None] * Ex[:, None, :]
        return dict(Ex=Ex, ExxT=ExxT, Covx=Covx, Exxp1T=Exxp1T), ll
