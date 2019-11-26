import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.scipy.special import gammaln, digamma, logsumexp
from autograd.scipy.special import logsumexp

from ssm.util import random_rotation, ensure_args_are_lists, \
    logistic, logit, one_hot
from ssm.regression import fit_linear_regression, generalized_newton_studentst_dof
from ssm.preprocessing import interpolate_data
from ssm.cstats import robust_ar_statistics
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs
import ssm.stats as stats
import ssm.regression as regression
import ssm.stats.distributions as dist


class NewHMMObservations(object):

    def __init__(self, K, D, M=0, observations="gaussian"):
        self.K, self.D, self.M = K, D, M

        # Instantiate an observation distribution for each of the K classes
        obs_classes = dict(
            gaussian=dist.MultivariateNormal,
            independent_gaussian=dist.DiagonalGaussian,
            studentst=dist.MultivariateStudentsT,
            independent_studentst=dist.DiagonalStudentsT,
            bernoulli=dist.Bernoulli,
            poisson=dist.Poisson,
            categorical=dist.Categorical,
            vonmises=dist.VonMises
            )

        self.obs_distns = [obs_classes[observations](dim=D) for _ in range(K)]

    @property
    def params(self):
        return [o.params for o in self.obs_distns]

    @params.setter
    def params(self, value):
        for o, v in zip(self.obs_distns, value):
            o.params = v

    def permute(self, perm):
        obs_distns = [self.obs_distns[i] for i in perm]
        self.obs_distns = obs_distns

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def log_prior(self):
        return 0

    def log_likelihoods(self, data, input, mask, tag):
        raise NotImplementedError

    def sample(self, states, data, input=None, tag=None, with_noise=True):
        raise NotImplementedError

    def m_step(self, expectations, datas, inputs, masks, tags,
               optimizer="bfgs", **kwargs):
        """
        If M-step cannot be done in closed form for the observations, default to SGD.
        """
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, e \
                in zip(datas, inputs, masks, tags, expectations):
                lls = self.log_likelihoods(data, input, mask, tag)
                elbo += np.sum(e["Ez"] * lls)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = optimizer(_objective, self.params, **kwargs)

    def smooth(self, expectations, data, input, tag):
        raise NotImplementedError

    def hessian_expected_log_dynamics_prob(self, Ez, data, input, mask, tag=None):
        # warnings.warn("Analytical Hessian is not implemented for this dynamics class. \
        #                Optimization via Laplace-EM may be slow. Consider using an \
        #                alternative posterior and inference method. ")
        raise NotImplementedError


class HMMObservations(object):

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

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def log_prior(self):
        return 0

    def log_likelihoods(self, data, input, mask, tag):
        raise NotImplementedError

    def sample(self, states, data, input=None, tag=None, with_noise=True):
        raise NotImplementedError

    def m_step(self, expectations, datas, inputs, masks, tags,
               optimizer="bfgs", **kwargs):
        """
        If M-step cannot be done in closed form for the observations, default to SGD.
        """
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, e \
                in zip(datas, inputs, masks, tags, expectations):
                lls = self.log_likelihoods(data, input, mask, tag)
                elbo += np.sum(e["Ez"] * lls)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = optimizer(_objective, self.params, **kwargs)

    def smooth(self, expectations, data, input, tag):
        raise NotImplementedError

    def hessian_expected_log_dynamics_prob(self, Ez, data, input, mask, tag=None):
        # warnings.warn("Analytical Hessian is not implemented for this dynamics class. \
        #                Optimization via Laplace-EM may be slow. Consider using an \
        #                alternative posterior and inference method. ")
        raise NotImplementedError


class GaussianObservations(HMMObservations):
    def __init__(self, K, D, M=0):
        super(GaussianObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self._sqrt_Sigmas = npr.randn(K, D, D)

    @property
    def params(self):
        return self.mus, self._sqrt_Sigmas

    @params.setter
    def params(self, value):
        self.mus, self._sqrt_Sigmas = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._sqrt_Sigmas = self._sqrt_Sigmas[perm]

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.mus = km.cluster_centers_
        Sigmas = np.array([np.atleast_2d(np.cov(data[km.labels_ == k].T))
                           for k in range(self.K)])
        assert np.all(np.isfinite(Sigmas))
        self._sqrt_Sigmas = np.linalg.cholesky(Sigmas + 1e-8 * np.eye(self.D))

    def log_likelihoods(self, data, input, mask, tag):
        mus, Sigmas = self.mus, self.Sigmas
        if mask is not None and np.any(~mask) and not isinstance(mus, np.ndarray):
            raise Exception("Current implementation of multivariate_normal_logpdf for masked data"
                            "does not work with autograd because it writes to an array. "
                            "Use DiagonalGaussian instead if you need to support missing data.")

        # stats.multivariate_normal_logpdf supports broadcasting, but we get
        # significant performance benefit if we call it with (TxD), (D,), and (D,D)
        # arrays as inputs
        return np.column_stack([stats.multivariate_normal_logpdf(data, mu, Sigma)
                               for mu, Sigma in zip(mus, Sigmas)])

    def sample(self, states, data, input=None, tag=None, with_noise=True):
        z = states[-1]
        D, mus = self.D, self.mus
        sqrt_Sigmas = self._sqrt_Sigmas if with_noise else np.zeros((self.K, self.D, self.D))
        return mus[z] + np.dot(sqrt_Sigmas[z], npr.randn(D))

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        K, D = self.K, self.D
        J = np.zeros((K, D))
        h = np.zeros((K, D))
        for e, y in zip(expectations, datas):
            Ez = e["Ez"]
            J += np.sum(Ez[:, :, None], axis=0)
            h += np.sum(Ez[:, :, None] * y[:, None, :], axis=0)
        self.mus = h / J

        # Update the variance
        sqerr = np.zeros((K, D, D))
        weight = np.zeros((K,))
        for e, y in zip(expectations, datas):
            Ez = e["Ez"]
            resid = y[:, None, :] - self.mus
            sqerr += np.sum(Ez[:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)
            weight += np.sum(Ez, axis=0)
        self._sqrt_Sigmas = np.linalg.cholesky(sqerr / weight[:, None, None] + 1e-8 * np.eye(self.D))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)


class DiagonalGaussianObservations(HMMObservations):
    def __init__(self, K, D, M=0):
        super(DiagonalGaussianObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self._log_sigmasq = -2 + npr.randn(K, D)

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @sigmasq.setter
    def sigmasq(self, value):
        assert np.all(value > 0) and value.shape == (self.K, self.D)
        self._log_sigmasq = np.log(value)

    @property
    def params(self):
        return self.mus, self._log_sigmasq

    @params.setter
    def params(self, value):
        self.mus, self._log_sigmasq = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._log_sigmasq = self._log_sigmasq[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.mus = km.cluster_centers_
        sigmas = np.array([np.var(data[km.labels_ == k], axis=0)
                           for k in range(self.K)])
        self._log_sigmasq = np.log(sigmas + 1e-16)

    def log_likelihoods(self, data, input, mask, tag):
        mus, sigmas = self.mus, np.exp(self._log_sigmasq) + 1e-16
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.diagonal_gaussian_logpdf(data[:, None, :], mus, sigmas, mask=mask[:, None, :])

    def sample(self, states, data, input=None, tag=None, with_noise=True):
        z = states[-1]
        D, mus = self.D, self.mus
        sigmas = np.exp(self._log_sigmasq) if with_noise else np.zeros((self.K, self.D))
        return mus[z] + np.sqrt(sigmas[z]) * npr.randn(D)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([e["Ez"] for e in expectations])
        for k in range(self.K):
            self.mus[k] = np.average(x, axis=0, weights=weights[:, k])
            sqerr = (x - self.mus[k])**2
            self._log_sigmasq[k] = np.log(np.average(sqerr, weights=weights[:, k], axis=0))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)


class StudentsTObservations(HMMObservations):
    def __init__(self, K, D, M=0):
        super(StudentsTObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self._log_sigmasq = -2 + npr.randn(K, D)
        # Student's t distribution also has a degrees of freedom parameter
        self._log_nus = np.log(4) * np.ones((K, D))

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @property
    def params(self):
        return self.mus, self._log_sigmasq, self._log_nus

    @params.setter
    def params(self, value):
        self.mus, self._log_sigmasq, self._log_nus = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._log_sigmasq = self._log_sigmasq[perm]
        self._log_nus = self._log_nus[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.mus = km.cluster_centers_
        sigmas = np.array([np.var(data[km.labels_ == k], axis=0) for k in range(self.K)])
        self._log_sigmasq = np.log(sigmas + 1e-16)
        self._log_nus = np.log(4) * np.ones((self.K, self.D))

    def log_likelihoods(self, data, input, mask, tag):
        D, mus, sigmas, nus = self.D, self.mus, self.sigmasq, self.nus
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.independent_studentst_logpdf(data[:, None, :], mus, sigmas, nus, mask=mask[:, None, :])

    def sample(self, states, data, input=None, tag=None, with_noise=True):
        z = states[-1]
        D, mus, sigmas, nus = self.D, self.mus, self.sigmasq, self.nus
        tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        sigma = sigmas[z] / tau if with_noise else 0
        return mus[z] + np.sqrt(sigma) * npr.randn(D)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t for
        complete details.
        """
        self._m_step_mu_sigma(expectations, datas, inputs, masks, tags)
        self._m_step_nu(expectations, datas, inputs, masks, tags)

    def _m_step_mu_sigma(self, expectations, datas, inputs, masks, tags):
        K, D = self.K, self.D

        # Estimate the precisions w for each data point
        E_taus = []
        for y in datas:
            # nu: (K,D)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> tau: (T, K, D)
            alpha = self.nus/2 + 1/2
            beta = self.nus/2 + 1/2 * (y[:, None, :] - self.mus)**2 / self.sigmasq
            E_taus.append(alpha / beta)

        # Update the mean (notation from natural params of Gaussian)
        J = np.zeros((K, D))
        h = np.zeros((K, D))
        for E_tau, e, y in zip(E_taus, expectations, datas):
            Ez = e["Ez"]
            J += np.sum(Ez[:, :, None] * E_tau, axis=0)
            h += np.sum(Ez[:, :, None] * E_tau * y[:, None, :], axis=0)
        self.mus = h / J

        # Update the variance
        sqerr = np.zeros((K, D))
        weight = np.zeros((K, D))
        for E_tau, e, y in zip(E_taus, expectations, datas):
            Ez = e["Ez"]
            sqerr += np.sum(Ez[:, :, None] * E_tau * (y[:, None, :] - self.mus)**2, axis=0)
            weight += np.sum(Ez[:, :, None], axis=0)
        self._log_sigmasq = np.log(sqerr / weight + 1e-16)

    def _m_step_nu(self, expectations, datas, inputs, masks, tags):
        """
        The shape parameter nu determines a gamma prior.  We have

            tau_n ~ Gamma(nu/2, nu/2)
            y_n ~ N(mu, sigma^2 / tau_n)

        To update nu, we do EM and optimize the expected log likelihood using
        a generalized Newton's method.  See the notebook in doc/students_t for
        complete details.
        """
        K, D = self.K, self.D

        # Compute the precisions w for each data point
        E_taus = np.zeros((K, D))
        E_logtaus = np.zeros((K, D))
        weights = np.zeros(K)
        for y, e in zip(datas, expectations):
            # nu: (K,D)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> alpha/beta: (T, K, D)
            alpha = self.nus/2 + 1/2
            beta = self.nus/2 + 1/2 * (y[:, None, :] - self.mus)**2 / self.sigmasq

            Ez = e["Ez"]
            E_taus += np.sum(Ez[:, :, None] * (alpha / beta), axis=0)
            E_logtaus += np.sum(Ez[:, :, None] * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights[:, None]
        E_logtaus /= weights[:, None]

        for k in range(K):
            for d in range(D):
                self._log_nus[k, d] = np.log(generalized_newton_studentst_dof(E_taus[k, d], E_logtaus[k, d]))


class MultivariateStudentsTObservations(HMMObservations):
    def __init__(self, K, D, M=0):
        super(MultivariateStudentsTObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self._sqrt_Sigmas = npr.randn(K, D, D)
        self._log_nus = np.log(4) * np.ones((K,))

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @property
    def params(self):
        return self.mus, self._sqrt_Sigmas, self._log_nus

    @params.setter
    def params(self, value):
        self.mus, self._sqrt_Sigmas, self._log_nus = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._sqrt_Sigmas = self._sqrt_Sigmas[perm]
        self._log_nus = self._log_nus[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.mus = km.cluster_centers_
        Sigmas = np.array([np.atleast_2d(np.cov(data[km.labels_ == k].T))
                           for k in range(self.K)])
        assert np.all(np.isfinite(Sigmas))
        self._sqrt_Sigmas = np.linalg.cholesky(Sigmas + 1e-8 * np.eye(self.D))
        self._log_nus = np.log(4) * np.ones((self.K,))

    def log_likelihoods(self, data, input, mask, tag):
        assert np.all(mask), "MultivariateStudentsTObservations does not support missing data"
        D, mus, Sigmas, nus = self.D, self.mus, self.Sigmas, self.nus

        # stats.multivariate_studentst_logpdf supports broadcasting, but we get
        # significant performance benefit if we call it with (TxD), (D,), (D,D), and (,)
        # arrays as inputs
        return np.column_stack([stats.multivariate_studentst_logpdf(data, mu, Sigma, nu)
                               for mu, Sigma, nu in zip(mus, Sigmas, nus)])

        # return stats.multivariate_studentst_logpdf(data[:, None, :], mus, Sigmas, nus)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t for
        complete details.
        """
        self._m_step_mu_sigma(expectations, datas, inputs, masks, tags)
        self._m_step_nu(expectations, datas, inputs, masks, tags)

    def _m_step_mu_sigma(self, expectations, datas, inputs, masks, tags):
        K, D = self.K, self.D

        # Estimate the precisions w for each data point
        E_taus = []
        for y in datas:
            # nu: (K,)  mus: (K, D)  Sigmas: (K, D, D)  y: (T, D)  -> tau: (T, K)
            alpha = self.nus/2 + D/2
            beta = self.nus/2 + 1/2 * stats.batch_mahalanobis(self._sqrt_Sigmas, y[:, None, :] - self.mus)
            E_taus.append(alpha / beta)

        # Update the mean (notation from natural params of Gaussian)
        J = np.zeros((K,))
        h = np.zeros((K, D))
        for E_tau, e, y in zip(E_taus, expectations, datas):
            Ez = e["Ez"]
            J += np.sum(Ez * E_tau, axis=0)
            h += np.sum(Ez[:, :, None] * E_tau[:, :, None] * y[:, None, :], axis=0)
        self.mus = h / J[:, None]

        # Update the variance
        sqerr = np.zeros((K, D, D))
        weight = np.zeros((K,))
        for E_tau, e, y in zip(E_taus, expectations, datas):
            # sqerr += np.sum(Ez[:, :, None] * E_tau * (y[:, None, :] - self.mus)**2, axis=0)
            Ez = e["Ez"]
            resid = y[:, None, :] - self.mus
            sqerr += np.einsum('tk,tk,tki,tkj->kij', Ez, E_tau, resid, resid)
            weight += np.sum(Ez, axis=0)

        self._sqrt_Sigmas = np.linalg.cholesky(sqerr / weight[:, None, None] + 1e-8 * np.eye(D))

    def _m_step_nu(self, expectations, datas, inputs, masks, tags):
        """
        The shape parameter nu determines a gamma prior.  We have

            tau_n ~ Gamma(nu/2, nu/2)
            y_n ~ N(mu, Sigma / tau_n)

        To update nu, we do EM and optimize the expected log likelihood using
        a generalized Newton's method.  See the notebook in doc/students_t for
        complete details.
        """
        K, D = self.K, self.D

        # Compute the precisions w for each data point
        E_taus = np.zeros(K)
        E_logtaus = np.zeros(K)
        weights = np.zeros(K)
        for y, e in zip(datas, expectations):
            # nu: (K,)  mus: (K, D)  Sigmas: (K, D, D)  y: (T, D)  -> alpha/beta: (T, K)
            alpha = self.nus/2 + D/2
            beta = self.nus/2 + 1/2 * stats.batch_mahalanobis(self._sqrt_Sigmas, y[:, None, :] - self.mus)

            Ez = e["Ez"]
            E_taus += np.sum(Ez * (alpha / beta), axis=0)
            E_logtaus += np.sum(Ez * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights
        E_logtaus /= weights

        for k in range(K):
            self._log_nus[k] = np.log(generalized_newton_studentst_dof(E_taus[k], E_logtaus[k]))

    def sample(self, states, data, input=None, tag=None, with_noise=True):
        z = states[-1]
        D, mus, Sigmas, nus = self.D, self.mus, self.Sigmas, self.nus
        tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        sqrt_Sigma = np.linalg.cholesky(Sigmas[z] / tau) if with_noise else 0
        return mus[z] + np.dot(sqrt_Sigma, npr.randn(D))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)


class BernoulliObservations(HMMObservations):

    def __init__(self, K, D, M=0):
        super(BernoulliObservations, self).__init__(K, D, M)
        self.logit_ps = npr.randn(K, D)

    @property
    def params(self):
        return self.logit_ps

    @params.setter
    def params(self, value):
        self.logit_ps = value

    def permute(self, perm):
        self.logit_ps = self.logit_ps[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):

        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        ps = np.clip(km.cluster_centers_, 1e-3, 1-1e-3)
        self.logit_ps = logit(ps)

    def log_likelihoods(self, data, input, mask, tag):
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.bernoulli_logpdf(data[:, None, :], self.logit_ps, mask=mask[:, None, :])

    def sample(self, states, data, input=None, tag=None, with_noise=True):
        z = states[-1]
        ps = 1 / (1 + np.exp(self.logit_ps))
        return npr.rand(self.D) < ps[z]

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([e["Ez"] for e in expectations])
        for k in range(self.K):
            ps = np.clip(np.average(x, axis=0, weights=weights[:,k]), 1e-3, 1-1e-3)
            self.logit_ps[k] = logit(ps)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        ps = 1 / (1 + np.exp(self.logit_ps))
        return expectations.dot(ps)


class PoissonObservations(HMMObservations):

    def __init__(self, K, D, M=0):
        super(PoissonObservations, self).__init__(K, D, M)
        self.log_lambdas = npr.randn(K, D)

    @property
    def params(self):
        return self.log_lambdas

    @params.setter
    def params(self, value):
        self.log_lambdas = value

    def permute(self, perm):
        self.log_lambdas = self.log_lambdas[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):

        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.log_lambdas = np.log(km.cluster_centers_ + 1e-3)

    def log_likelihoods(self, data, input, mask, tag):
        lambdas = np.exp(self.log_lambdas)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.poisson_logpdf(data[:, None, :], lambdas, mask=mask[:, None, :])

    def sample(self, states, data, input=None, tag=None, with_noise=True):
        z = states[-1]
        lambdas = np.exp(self.log_lambdas)
        return npr.poisson(lambdas[z])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([e["Ez"] for e in expectations])
        for k in range(self.K):
            self.log_lambdas[k] = np.log(np.average(x, axis=0, weights=weights[:,k]) + 1e-16)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(np.exp(self.log_lambdas))


class PoissonGLMObservations(HMMObservations):
    """
    Poisson observations with a GLM mapping from inputs to rates.

    lambda_{n,t} = f(w_{z_t, n}^T x_t + b_{n, z_t}))

    where

    n:    neuron index
    t:    time index
    z_t:  discrete state at time t
    x_t:  input at time t
    """

    def __init__(self, K, D, M=0, mean_function="softplus",
                 weight_mean=0, weight_variance=1,
                 init_m_steps=0, init_softmax_temp=1.0):
        """
        K:  number of discrete states
        D:  number of neurons
        M:  number of input dimensions
        """
        super(PoissonGLMObservations, self).__init__(K, D, M)
        self.mean_function = mean_function
        self.init_m_steps = init_m_steps
        self.init_softmax_temp = init_softmax_temp

        # Set the prior
        self.weight_mean = weight_mean * np.ones(M + 1)
        self.weight_var = weight_variance * np.eye(M + 1)

        # Initialize weights
        self.W = npr.randn(K, D, M) / (np.sqrt(M) * 3)
        self.b = np.zeros((K, D))

    @property
    def params(self):
        return (self.W, self.b)

    @params.setter
    def params(self, value):
        self.W, self.b = value

    def permute(self, perm):
        self.W = self.W[perm]
        self.b = self.b[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        """Initialize model with mixture of GLMs."""
        if self.init_m_steps == 0:
            return

        # Initialize expectations.
        expectations = []
        K = self.W.shape[0]
        for inp in inputs:
            T = inp.shape[0]
            ep = np.zeros((T, K))
            # ep[np.arange(T), npr.randint(K, size=T)] = 1.0
            ep = npr.exponential(size=(T, K))
            ep /= np.sum(ep, axis=1, keepdims=True)
            expectations.append((ep, None, None))

        # Perform one m-step.
        print('performing an initial m-step...')
        self.m_step(expectations, datas, inputs, masks, tags)

        # Perform more m-steps.
        for _ in range(self.init_m_steps - 1):
            expectations = []
            for d, inp, m, tg in zip(datas, inputs, masks, tags):
                lls = self.init_softmax_temp * self.log_likelihoods(d, inp, m, tg)
                ep = np.exp(lls - logsumexp(lls, axis=-1, keepdims=True))
                expectations.append((ep, None, None))

            print('performing an initial m-step...')
            self.m_step(expectations, datas, inputs, masks, tags)

    def log_likelihoods(self, data, input, mask, tag):
        # inputs:   (T, M)
        # data:     (T, D)
        # weights:  (K, D, M)
        # biases:   (K, D)
        # lambdas:  (T, K, D)
        # lls:      (T, K)
        f = regression.mean_functions[self.mean_function]
        lambdas = f(np.einsum('tm,kdm->tkd', input, self.W) + self.b[None, :, :])
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.poisson_logpdf(data[:, None, :], lambdas, mask=mask[:, None, :])

    def sample(self, states, data, input=None, tag=None, with_noise=True):
        z = states[-1]
        f = regression.mean_functions[self.mean_function]
        lambdas = f(np.einsum('m,kdm->kd', input[-1], self.W) + self.b)
        return npr.poisson(lambdas[z])

    def m_step(self, *args, **kwargs):
        return self._fast_m_step(*args, **kwargs)

    def _slow_m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        weights = np.concatenate([e["Ez"] for e in expectations])
        Xs = np.row_stack(inputs)
        ys = np.row_stack(datas)

        for k in range(self.K):
            for d in range(self.D):
                self.W[k, d], self.b[k, d] = \
                    regression.fit_scalar_glm(Xs, ys[:, d], weights=weights[:, k],
                        model="poisson", mean_function=self.mean_function, fit_intercept=True,
                        prior=(self.weight_mean, self.weight_var),
                        initial_theta=np.concatenate((self.W[k, d], [self.b[k, d]])))

                assert np.all(np.isfinite(self.W))
                assert np.all(np.isfinite(self.b))

    def _fast_m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        weights = np.concatenate([e["Ez"] for e in expectations])
        X = np.row_stack(inputs)  # (timesteps x n_inputs)
        Y = np.row_stack(datas)   # (timesteps x n_units)

        # Add bias term.
        nt = X.shape[0]
        X = np.column_stack([X, np.ones(nt)])

        for k in range(self.K):
            Wk = np.column_stack([self.W[k], self.b[k]]) # (n_units x n_inputs)
            z = weights[:, k]

            Xw = X @ Wk.T           # timesteps x units
            exp_Xw = np.exp(Xw)     # timesteps x units

            losses = np.squeeze(z[None, :] @ (exp_Xw - Y * Xw))  # units

            zX = z[:, None] * X
            grads = (exp_Xw - Y).T @ zX  # units x inputs

            # units x inputs x inputs
            hess = np.einsum("ij,ihk->kjh", X, exp_Xw[:, None, :] * zX[:, :, None])
            dWk = np.linalg.solve(hess, grads)

            for n in range(self.D):
                ss = 1.0
                converged = False

                while (not converged) and (ss > 1e-7):
                    Wnew = Wk[n] - (ss * dWk[n])
                    _Xw = X @ Wnew
                    _eXw = np.exp(_Xw)
                    newloss = z @ (_eXw - Y[:, n] * _Xw)

                    if newloss < losses[n]:
                        self.W[k, n] = Wnew[:-1]
                        self.b[k, n] = Wnew[-1]
                        converged = True
                    else:
                        ss *= .5

                if not converged:
                    # print("warn: failed to converge.")
                    break

                assert np.all(np.isfinite(self.W))
                assert np.all(np.isfinite(self.b))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        f = regression.mean_functions[self.mean_function]
        lambdas = f(np.einsum('tm,kdm->tkd', input, self.W) + self.b[None, :, :])
        return np.sum(expectations[:, :, None] * lambdas, axis=1)


class CategoricalObservations(HMMObservations):

    def __init__(self, K, D, M=0, C=2):
        """
        @param C:  number of classes in the categorical observations
        """
        super(CategoricalObservations, self).__init__(K, D, M)
        self.C = C
        self.logits = npr.randn(K, D, C)

    @property
    def params(self):
        return self.logits

    @params.setter
    def params(self, value):
        self.logits = value

    def permute(self, perm):
        self.logits = self.logits[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def log_likelihoods(self, data, input, mask, tag):
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.categorical_logpdf(data[:, None, :], self.logits, mask=mask[:, None, :])

    def sample(self, states, data, input=None, tag=None, with_noise=True):
        z = states[-1]
        ps = np.exp(self.logits - logsumexp(self.logits, axis=2, keepdims=True))
        return np.array([npr.choice(self.C, p=ps[z, d]) for d in range(self.D)])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([e["Ez"] for e in expectations])
        for k in range(self.K):
            # compute weighted histogram of the class assignments
            xoh = one_hot(x, self.C)                                          # T x D x C
            ps = np.average(xoh, axis=0, weights=weights[:, k]) + 1e-3        # D x C
            ps /= np.sum(ps, axis=-1, keepdims=True)
            self.logits[k] = np.log(ps)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError


class VonMisesObservations(HMMObservations):
    def __init__(self, K, D, M=0):
        super(VonMisesObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self.log_kappas = np.log(-1*npr.uniform(low=-1, high=0, size=(K, D)))

    @property
    def params(self):
        return self.mus, self.log_kappas

    @params.setter
    def params(self, value):
        self.mus, self.log_kappas = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self.log_kappas = self.log_kappas[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # TODO: add spherical k-means for initialization
        pass

    def log_likelihoods(self, data, input, mask, tag):
        mus, kappas = self.mus, np.exp(self.log_kappas)

        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.vonmises_logpdf(data[:, None, :], mus, kappas, mask=mask[:, None, :])

    def sample(self, states, data, input=None, tag=None, with_noise=True):
        z = states[-1]
        D, mus, kappas = self.D, self.mus, np.exp(self.log_kappas)
        return npr.vonmises(self.mus[z], kappas[z], D)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        x = np.concatenate(datas)
        weights = np.concatenate([e["Ez"] for e in expectations])  # T x D
        assert x.shape[0] == weights.shape[0]

        # convert angles to 2D representation and employ closed form solutions
        x_k = np.stack((np.sin(x), np.cos(x)), axis=1)  # T x 2 x D

        r_k = np.tensordot(weights.T, x_k, axes=1)  # K x 2 x D
        r_norm = np.sqrt(np.sum(np.power(r_k, 2), axis=1))  # K x D

        mus_k = np.divide(r_k, r_norm[:, None])  # K x 2 x D
        r_bar = np.divide(r_norm, np.sum(weights, 0)[:, None])  # K x D

        mask = (r_norm.sum(1) == 0)
        mus_k[mask] = 0
        r_bar[mask] = 0

        # Approximation
        kappa0 = r_bar * (self.D + 1 - np.power(r_bar, 2)) / (1 - np.power(r_bar, 2))  # K,D

        kappa0[kappa0 == 0] += 1e-6

        for k in range(self.K):
            self.mus[k] = np.arctan2(*mus_k[k])  #
            self.log_kappas[k] = np.log(kappa0[k])  # K, D

    def smooth(self, expectations, data, input, tag):
        mus = self.mus
        return expectations.dot(mus)
