from functools import partial
from tqdm.auto import trange

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad

from ssm.optimizers import adam_step, rmsprop_step, sgd_step, convex_combination
from ssm.util import ensure_args_are_lists, ensure_args_not_none, \
    ensure_slds_args_not_none, ensure_variational_args_are_lists, \
    replicate, collapse

import ssm.observations as obs
import ssm.transitions as trans
import ssm.init_state_distns as isd
import ssm.hierarchical as hier
import ssm.emissions as emssn
import ssm.posterior as post

__all__ = ['HMM', 'HSMM']


class _SSM(object):
    """
    Abstract base class for temporal state space models (SSMs).

    SSMs specify a joint distribution over latent states and observed data.
    The observations are typically assumed to be conditionally independent
    of one another given their underlying latent states. The latent states,
    in turn, are linked by temporal dependencies.  Finally, the latent states
    may also have an initial distribution for the first time step.

    This base class implements generic fitting code for state space models.
    It supports:

    1.  Stochastic gradient ascent of the marginal likelihood, as long
        as it is possible to obtain an estimate of or lower bound on that
        quantity.

    2.  (Generalized) expectation maximization, which alternates
        between updating an (approximate) posterior distribution and then
        updating global parameters to maximize the expected log probability
        under that posterior.  The family of generalized EM algorithms includes
        variational EM, Monte Carlo EM, particle EM, and so on.

    3.  Gibbs sampling, which alternates between sampling the latent states
        and sampling the model parameters from their respective conditional
        distributions.
    """

    # Specify the set of available (approximate) posterior distributions
    _posterior_classes = {}

    @property
    def initial_state_distn(self):
        raise NotImplementedError

    @property
    def transition_distn(self):
        raise NotImplementedError

    @property
    def observation_distn(self):
        raise NotImplementedError

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """
        Initialize parameters given data.
        """
        pass

    def sample(self, num_timesteps,
               inputs=None,
               preceding_states=None,
               preceding_data=None,
               preceding_inputs=None,
               tag=None,
               with_noise=True):
        """
        Sample synthetic data from the model. Optionally, condition on a given
        prefix (preceding discrete states and data).

        Parameters
        ----------
        num_timesteps : int
            number of time steps to sample

        inputs : iterable of length num_timesteps

        preceding_states : interable
            Optional set of preceding latente states.  Must match the type of
            the output states.

        preceding_data : interable of same length as preceding_states
            Optional set of preceding observed data.  Must match the type of
            the output states.

        preceding_inputs : interable of same length as preceding_states
            Optional set of preceding observed data.  Must match the type of
            the output states.

        Returns
        -------
        states : list
            Sequence of sampled discrete states

        data : list
            Array of sampled data
        """
        covariates, states, data = [], [], []

        # Make dummy inputs if necessary
        inputs = [None] * num_timesteps if inputs is None else inputs

        # Initialize the states and data
        has_preceding = None not in [preceding_states, preceding_data]
        if has_preceding:
            assert len(preceding_data) == len(preceding_states)
            preceding_inputs = [None] * len(preceding_states) if preceding_inputs is None else preceding_inputs
            assert len(preceding_inputs) == len(preceding_states)

            covariates.extend(preceding_inputs)
            states.extend(preceding_states)
            data.extend(preceding_data)

            # Sample the first state given the preceding states and data
            covariates.append(inputs[0])
            states.append(self.transition_distn.sample(states, data, covariates, tag, with_noise=with_noise))

        else:
            # Sample the initial state from its initial distribution
            covariates.append(inputs[0])
            states.append(self.initial_state_distn.sample(data, covariates, tag, with_noise=with_noise))

        # Sample the first data point, then fill in the rest of the data
        data.append(self.observation_distn.sample(states, data, covariates, tag, with_noise=with_noise))
        for t in range(1, num_timesteps):
            covariates.append(inputs[t])
            states.append(self.transition_distn.sample(states, data, covariates, tag, with_noise=with_noise))
            data.append(self.observation_distn.sample(states, data, covariates, tag, with_noise=with_noise))

        # Return the sampled data
        return states[-num_timesteps:], data[-num_timesteps:]

    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """
        return self.initial_state_distn.log_prior() + \
               self.transition_distn.log_prior() + \
               self.observation_distn.log_prior()

    @ensure_args_are_lists
    def log_likelihood(self, datas, inputs=None, masks=None, tags=None, posterior="exact"):
        """
        Compute the log probability of the data under the current
        model parameters.

        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        ll = 0
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            p = self._posterior_classes[posterior](self, data, input, mask, tag)
            ll += p.marginal_likelihood
            assert np.isfinite(ll)
        return ll

    @ensure_args_are_lists
    def log_probability(self, datas, inputs=None, masks=None, tags=None):
        return self.log_likelihood(datas, inputs, masks, tags) + self.log_prior()

    # Model fitting
    def _fit_sgd(self, optimizer, posteriors, datas, inputs, masks, tags, num_iters=1000, **kwargs):
        """
        Fit the model with maximum marginal likelihood.
        """
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.initial_state_distn.params = params[0]
            self.transition_distn.params = params[1]
            self.observation_distn.params = params[2]

            obj = self.log_prior()
            for posterior in posteriors:
                obj += posterior.marginal_likelihood
            return -obj / T

        # Set up the progress bar
        params = (self.initial_state_distn.params,
                  self.transition_distn.params,
                  self.observation_distn.params)
        lls = [-_objective(params, 0) * T]
        pbar = trange(num_iters)
        pbar.set_description("Epoch {} Itr {} LP: {:.1f}".format(0, 0, lls[-1]))

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            params, val, g, state = step(value_and_grad(_objective), params, itr, state, **kwargs)
            self.initial_state_distn.params = params[0]
            self.transition_distn.params = params[1]
            self.observation_distn.params = params[2]

            lls.append(-val * T)
            pbar.set_description("LP: {:.1f}".format(lls[-1]))
            pbar.update(1)

        return lls

    def _fit_em(self, posteriors, datas, inputs, masks, tags,
                num_em_iters=100, tolerance=0,
                init_state_mstep_kwargs={},
                transitions_mstep_kwargs={},
                observations_mstep_kwargs={}):
        """
        Fit the parameters with expectation maximization.

        E step: compute E[z_t] and E[z_t, z_{t+1}] with message passing;
        M-step: analytical maximization of E_{p(z | x)} [log p(x, z; theta)].
        """
        # Run one E step to start
        expectations, lls = list(zip(*[posterior.E_step() for posterior in posteriors]))
        logprobs = [self.log_prior() + sum(lls)]

        pbar = trange(num_em_iters)
        pbar.set_description("LP: {:.1f}".format(lls[-1]))
        for itr in pbar:
            # M step: maximize expected log joint wrt parameters
            self.initial_state_distn.m_step(expectations, datas, inputs, masks, tags, **init_state_mstep_kwargs)
            self.transition_distn.m_step(expectations, datas, inputs, masks, tags, **transitions_mstep_kwargs)
            self.observation_distn.m_step(expectations, datas, inputs, masks, tags, **observations_mstep_kwargs)

            # E step: compute expected latent states under the posterior
            expectations, lls = list(zip(*[posterior.E_step() for posterior in posteriors]))
            logprobs.append(self.log_prior() + sum(lls))

            # Show progress
            pbar.set_description("LP: {:.1f}".format(logprobs[-1]))

            # Check for convergence
            if itr > 0 and abs(logprobs[-1] - logprobs[-2]) < tolerance:
                pbar.set_description("Converged to LP: {:.1f}".format(logprobs[-1]))
                break

        return logprobs

    def _fit_gibbs(self, posteriors, datas, inputs, masks, tags,
                num_em_iters=100, tolerance=0,
                init_state_mstep_kwargs={},
                transitions_mstep_kwargs={},
                observations_mstep_kwargs={}):

        # Initialize the posteriors with a draw from their conditional
        states = [posterior.sample() for posterior in posteriors]
        lls = [self.log_prior() + sum(p.log_joint for p in posteriors)]

        pbar = trange(num_em_iters)
        pbar.set_description("LP: {:.1f}".format(lls[-1]))
        for itr in pbar:
            # Update the parameters
            self.initial_state_distn.gibbs_step(states, datas, inputs, masks, tags, **init_state_mstep_kwargs)
            self.transition_distn.gibbs_step(states, datas, inputs, masks, tags, **transitions_mstep_kwargs)
            self.observation_distn.gibbs_step(states, datas, inputs, masks, tags, **observations_mstep_kwargs)

            # Update the latent states
            states = [posterior.sample() for posterior in posteriors]
            lls.append(self.log_prior() + sum(p.log_joint for p in posteriors))

            # Print progress
            pbar.set_description("LP: {:.1f}".format(lls[-1]))

            # Check for convergence
            if itr > 0 and abs(lls[-1] - lls[-2]) < tolerance:
                pbar.set_description("Converged to LP: {:.1f}".format(lls[-1]))
                break

        return lls

    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, tags=None,
            method="em", posterior="exact", initialize=True, **kwargs):
        _fitting_methods = \
            dict(sgd=partial(self._fit_sgd, "sgd"),
                 adam=partial(self._fit_sgd, "adam"),
                 em=self._fit_em,
                 gibbs=self._fit_gibbs
                 )

        # Initialize the parameters
        if initialize:
            self.initialize(datas, inputs=inputs, masks=masks, tags=tags)

        # Construct the posterior objects
        if posterior not in self._posterior_classes:
            raise Exception("Invalid posterior: {}. Options are {}".
                            format(posterior, self._posterior_classes.keys()))

        posteriors = [self._posterior_classes[posterior](self, data, input, mask, tag)
                      for data, input, mask, tag in
                      zip(datas, inputs, masks, tags)]

        # Run the appropriate fitting method
        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".
                            format(method, _fitting_methods.keys()))

        lls = _fitting_methods[method](posteriors, datas, inputs=inputs, masks=masks, tags=tags, **kwargs)

        # Return training likelihoods and posterior(s)
        if len(posteriors) == 1:
            return lls, posteriors[0]
        else:
            return lls, posteriors


    @ensure_args_are_lists
    def infer(self, datas, inputs=None, masks=None, tags=None, posterior="exact", **kwargs):
        """
        Infer the posterior distribution over data given fixed model parameters.
        """
        # Construct the posterior objects
        if posterior not in self._posterior_classes:
            raise Exception("Invalid posterior: {}. Options are {}".
                            format(posterior, self._posterior_classes.keys()))

        posteriors = []
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            p = self._posterior_classes[posterior](self, data, input, mask, tag)
            posteriors.append(p)

        # Return posterior(s)
        if len(posteriors) == 1:
            return posteriors[0]
        else:
            return posteriors


class HMM(_SSM):
    """
    A Hidden Markov Model.

    Notation:
    K: number of discrete latent states
    D: dimensionality of observations
    M: dimensionality of inputs

    In the code we will sometimes refer to the discrete
    latent state sequence as z and the data as x.
    """
    _posterior_classes = dict(exact=post.HMMExactPosterior)

    def __init__(self, K, D, M=0, init_state_distn=None,
                 transitions='standard',
                 transition_kwargs=None,
                 hierarchical_transition_tags=None,
                 observations="gaussian", observation_kwargs=None,
                 hierarchical_observation_tags=None, **kwargs):

        # Make the initial state distribution
        if init_state_distn is None:
            init_state_distn = isd.InitialStateDistribution(K, D, M=M)
        if not isinstance(init_state_distn, isd.InitialStateDistribution):
            raise TypeError("'init_state_distn' must be a subclass of"
                            " ssm.init_state_distns.InitialStateDistribution")

        # Make the transition model
        transition_classes = dict(
            standard=trans.StationaryTransitions,
            stationary=trans.StationaryTransitions,
            sticky=trans.StickyTransitions,
            inputdriven=trans.InputDrivenTransitions,
            recurrent=trans.RecurrentTransitions,
            recurrent_only=trans.RecurrentOnlyTransitions,
            rbf_recurrent=trans.RBFRecurrentTransitions,
            nn_recurrent=trans.NeuralNetworkRecurrentTransitions
            )

        if isinstance(transitions, str):
            if transitions not in transition_classes:
                raise Exception("Invalid transition model: {}. Must be one of {}".
                    format(transitions, list(transition_classes.keys())))

            transition_kwargs = transition_kwargs or {}
            transitions = \
                hier.HierarchicalTransitions(transition_classes[transitions], K, D, M=M,
                                        tags=hierarchical_transition_tags,
                                        **transition_kwargs) \
                if hierarchical_transition_tags is not None \
                else transition_classes[transitions](K, D, M=M, **transition_kwargs)
        if not isinstance(transitions, trans.Transitions):
            raise TypeError("'transitions' must be a subclass of"
                            " ssm.transitions.Transitions")

        # This is the master list of observation classes.
        # When you create a new observation class, add it here.
        observation_classes = dict(
            gaussian=obs.GaussianObservations,
            diagonal_gaussian=obs.DiagonalGaussianObservations,
            studentst=obs.MultivariateStudentsTObservations,
            t=obs.MultivariateStudentsTObservations,
            diagonal_t=obs.StudentsTObservations,
            diagonal_studentst=obs.StudentsTObservations,
            bernoulli=obs.BernoulliObservations,
            categorical=obs.CategoricalObservations,
            poisson=obs.PoissonObservations,
            vonmises=obs.VonMisesObservations,
            ar=obs.AutoRegressiveObservations,
            autoregressive=obs.AutoRegressiveObservations,
            no_input_ar=obs.AutoRegressiveObservationsNoInput,
            diagonal_ar=obs.AutoRegressiveDiagonalNoiseObservations,
            diagonal_autoregressive=obs.AutoRegressiveDiagonalNoiseObservations,
            independent_ar=obs.IndependentAutoRegressiveObservations,
            robust_ar=obs.RobustAutoRegressiveObservations,
            no_input_robust_ar=obs.RobustAutoRegressiveObservationsNoInput,
            robust_autoregressive=obs.RobustAutoRegressiveObservations,
            diagonal_robust_ar=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            diagonal_robust_autoregressive=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            )

        if isinstance(observations, str):
            observations = observations.lower()
            if observations not in observation_classes:
                raise Exception("Invalid observation model: {}. Must be one of {}".
                    format(observations, list(observation_classes.keys())))

            observation_kwargs = observation_kwargs or {}
            observations = \
                hier.HierarchicalObservations(observation_classes[observations], K, D, M=M,
                                        tags=hierarchical_observation_tags,
                                        **observation_kwargs) \
                if hierarchical_observation_tags is not None \
                else observation_classes[observations](K, D, M=M, **observation_kwargs)
        if not isinstance(observations, obs.Observations):
            raise TypeError("'observations' must be a subclass of"
                            " ssm.observations.Observations")

        self.K, self.D, self.M = K, D, M
        self._init_state_distn = init_state_distn
        self._transition_distn = transitions
        self._observation_distn = observations

    @property
    def initial_state_distn(self):
        return self._init_state_distn

    @property
    def transition_distn(self):
        return self._transition_distn

    @property
    def observation_distn(self):
        return self._observation_distn

    # TODO: Expose properties for common model attributes of the model

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        assert np.all(np.sort(perm) == np.arange(self.K))
        self.initial_state_distn.permute(perm)
        self.transition_distn.permute(perm)
        self.observation_distn.permute(perm)

    def sample(self, num_timesteps,
               inputs=None,
               preceding_states=None,
               preceding_data=None,
               preceding_inputs=None,
               tag=None,
               with_noise=True):
        """
        Sample synthetic data from the model. Optionally, condition on a given
        prefix (preceding discrete states and data).

        Parameters
        ----------
        num_timesteps : int
            number of time steps to sample

        inputs : iterable of length num_timesteps

        preceding_states : interable
            Optional set of preceding latente states.  Must match the type of
            the output states.

        preceding_data : interable of same length as preceding_states
            Optional set of preceding observed data.  Must match the type of
            the output states.

        preceding_inputs : interable of same length as preceding_states
            Optional set of preceding observed data.  Must match the type of
            the output states.

        Returns
        -------
        states : array_like of dtype int
            Sequence of sampled discrete states

        data : array_like
            Array of sampled data
        """
        states, data = \
            super(HMM, self).sample(num_timesteps,
                                    inputs=inputs,
                                    preceding_states=preceding_states,
                                    preceding_data=preceding_data,
                                    preceding_inputs=preceding_inputs,
                                    tag=tag,
                                    with_noise=with_noise)

        states = np.array(states, dtype=int)
        data = np.array(data)
        return states, data



class HSMM(HMM):
    """
    Hidden semi-Markov model with non-geometric duration distributions.
    The trick is to expand the state space with "super states" and "sub states"
    that effectively count duration. We rely on the transition model to
    specify a "state map," which maps the super states (1, .., K) to
    super+sub states ((1,1), ..., (1,r_1), ..., (K,1), ..., (K,r_K)).
    Here, r_k denotes the number of sub-states of state k.
    """
    _posterior_classes = dict(exact=post.HSMMExactPosterior)

    def __init__(self, K, D, M=0, init_state_distn=None,
                 transitions="nb", transition_kwargs=None,
                 observations="gaussian", observation_kwargs=None,
                 **kwargs):

        if init_state_distn is None:
            init_state_distn = isd.InitialStateDistribution(K, D, M=M)
        if not isinstance(init_state_distn, isd.InitialStateDistribution):
            raise TypeError("'init_state_distn' must be a subclass of"
                            " ssm.init_state_distns.InitialStateDistribution")

        # Make the transition model
        transition_classes = dict(
            nb=trans.NegativeBinomialSemiMarkovTransitions,
            )
        if isinstance(transitions, str):
            if transitions not in transition_classes:
                raise Exception("Invalid transition model: {}. Must be one of {}".
                    format(transitions, list(transition_classes.keys())))

            transition_kwargs = transition_kwargs or {}
            transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)
        if not isinstance(transitions, trans.Transitions):
            raise TypeError("'transitions' must be a subclass of"
                            " ssm.transitions.Transitions")

        # This is the master list of observation classes.
        # When you create a new observation class, add it here.
        observation_classes = dict(
            gaussian=obs.GaussianObservations,
            diagonal_gaussian=obs.DiagonalGaussianObservations,
            studentst=obs.MultivariateStudentsTObservations,
            t=obs.MultivariateStudentsTObservations,
            diagonal_t=obs.StudentsTObservations,
            diagonal_studentst=obs.StudentsTObservations,
            bernoulli=obs.BernoulliObservations,
            categorical=obs.CategoricalObservations,
            poisson=obs.PoissonObservations,
            vonmises=obs.VonMisesObservations,
            ar=obs.AutoRegressiveObservations,
            autoregressive=obs.AutoRegressiveObservations,
            diagonal_ar=obs.AutoRegressiveDiagonalNoiseObservations,
            diagonal_autoregressive=obs.AutoRegressiveDiagonalNoiseObservations,
            independent_ar=obs.IndependentAutoRegressiveObservations,
            robust_ar=obs.RobustAutoRegressiveObservations,
            robust_autoregressive=obs.RobustAutoRegressiveObservations,
            diagonal_robust_ar=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            diagonal_robust_autoregressive=obs.RobustAutoRegressiveDiagonalNoiseObservations,
            )

        if isinstance(observations, str):
            observations = observations.lower()
            if observations not in observation_classes:
                raise Exception("Invalid observation model: {}. Must be one of {}".
                    format(observations, list(observation_classes.keys())))

            observation_kwargs = observation_kwargs or {}
            observations = observation_classes[observations](K, D, M=M, **observation_kwargs)
        if not isinstance(observations, obs.Observations):
            raise TypeError("'observations' must be a subclass of"
                            " ssm.observations.Observations")

        self.K, self.D, self.M = K, D, M
        self._init_state_distn = init_state_distn
        self._transition_distn = transitions
        self._observation_distn = observations

    @property
    def initial_state_distn(self):
        return self._init_state_distn

    @property
    def transition_distn(self):
        return self._transition_distn

    @property
    def observation_distn(self):
        return self._observation_distn

    @property
    def state_map(self):
        return self.transition_distn.state_map

    def sample(self, num_timesteps,
               inputs=None,
               preceding_states=None,
               preceding_data=None,
               preceding_inputs=None,
               tag=None,
               with_noise=True):
        """
        Sample synthetic data from the model. Optionally, condition on a given
        prefix (preceding discrete states and data).

        Parameters
        ----------
        num_timesteps : int
            number of time steps to sample

        inputs : iterable of length num_timesteps

        preceding_states : interable
            Optional set of preceding latente states.  Must match the type of
            the output states.

        preceding_data : interable of same length as preceding_states
            Optional set of preceding observed data.  Must match the type of
            the output states.

        preceding_inputs : interable of same length as preceding_states
            Optional set of preceding observed data.  Must match the type of
            the output states.

        Returns
        -------
        states : array_like of dtype int
            Sequence of sampled discrete states

        data : array_like
            Array of sampled data
        """
        states, data = \
            super(HSMM, self).sample(num_timesteps,
                                    inputs=inputs,
                                    preceding_states=preceding_states,
                                    preceding_data=preceding_data,
                                    preceding_inputs=preceding_inputs,
                                    tag=tag,
                                    with_noise=with_noise)

        states = np.array(states, dtype=int)
        data = np.array(data)

        # Collapse the states
        states = self.state_map[states]

        return states, data


