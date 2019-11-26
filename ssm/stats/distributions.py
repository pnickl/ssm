"""
ssm.stats provides functions for evaluating log probabilities and
classes for wrapping distributions.

We should really be using JAX for fitting these things!

"""
import autograd.numpy as np
from autograd.scipy.special import gammaln, logsumexp

from ssm.util.util import one_hot, flatten_to_dim, batch_mahalanobis
from ssm.util.optimizers import adam, bfgs, lbfgs, rmsprop, sgd


class Distribution(object):
    """
    Base class for distribution objects.  These are analogous to
    scipy.stats distribution objects or TensorFlow Probability /
    Pytorch distribution objects, but replicated here to work with
    autograd.  In the future, it would be really nice if autograd/JAX
    came with these types of objects already!
    """
    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def log_probability(self, data):
        """
        Return the log probability of the data under the given parameters
        """
        raise NotImplementedError

    def sample(self, num_samples=1):
        raise NotImplementedError

    def m_step(self, datas, weights=None, expected_stats=None, optimizer="lbfgs"):
        if weights is None:
            weights = [np.ones(data.shape[0]) for data in datas]

        def _objective(prms):
            self.params = prms
            lp = 0
            scale = 0
            for data, w in zip(datas, weights):
                lp += np.dot(w, self.log_probability(data))
                scale += np.sum(w)
            return lp / scale

        # Call the optimizer. Persist state (e.g. SGD momentum) across calls to m_step.
        optimizer = dict(sgd=sgd, adam=adam, rmsprop=rmsprop, bfgs=bfgs, lbfgs=lbfgs)[optimizer]
        optimizer_state = self.optimizer_state if hasattr(self, "optimizer_state") else None
        self.params, self.optimizer_state = \
            optimizer(_objective, self.params, num_iters=num_iters,
                      state=optimizer_state, full_output=True, **kwargs)


class MultivariateNormal(Distribution):
    def __init__(self, mu=None, Sigma=None, dim=None):
        assert (mu and Sigma) or dim, "Constructor requires either a mean and covariance or a data dimension."

        if mu and Sigma:
            self.dim = mu.shape[-1]
            self.mu = mu
            self.Sigma = Sigma
        else:
            assert isinstance(dim, int)
            self.dim = dim
            self.mu = np.zeros(dim)
            self.Sigma = np.eye(dim)

    @property
    def params(self):
        return self.mu, self._Sigma_chol

    @params.setter
    def params(self, value):
        self.mu, self._Sigma_chol = value

    @property
    def Sigma(self):
        return np.matmul(self._Sigma_chol, np.swapaxes(self._Sigma_chol, -1, -2))

    @Sigma.setter
    def Sigma(self, value):
        self._Sigma_chol = np.linalg.cholesky(value)

    def log_probability(self, data, mask=None):
        return multivariate_normal_logpdf(data, self.mu, self.Sigma, mask=mask)

    def expected_log_probability(self, expectations):
        Ex, ExxT = expectations
        mumuT = np.matmul(self.mu, np.swapaxes(self.mu, -1, -2))
        return expected_multivariate_normal_logpdf(Ex, ExxT, self.mu, mumuT, self.Sigma)

    def m_step(self, datas, weights=None, expected_stats=None):
        if weights is None:
            weights = [np.ones(data.shape[0]) for data in datas]

        if expected_stats is None:
            dim = self.dim
            Ex = np.zeros(dim)
            ExxT = np.zeros((dim, dim))

            w_tot = 1e-16
            for w, x in zip(weights, datas):
                w_tot += np.sum(w)
                Ex += np.sum(w[:, None] * x, axis=0)
                ExxT += einsum('n,ni,nj->ij', w, x, x)

            Ex /= w_tot
            ExxT /= w_tot

        else:
            Ex, ExxT = expected_stats

        self.mu = Ex
        self.Sigma = ExxT - np.outer(Ex, Ex)


def _multivariate_normal_logpdf(data, mus, Sigmas, Ls=None):
    """
    Compute the log probability density of a multivariate Gaussian distribution.
    This will broadcast as long as data, mus, Sigmas have the same (or at
    least be broadcast compatible along the) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The mean(s) of the Gaussian distribution(s)

    Sigmas : array_like (..., D, D)
        The covariances(s) of the Gaussian distribution(s)

    Ls : array_like (..., D, D)
        Optionally pass in the Cholesky decomposition of Sigmas

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the multivariate Gaussian distribution(s).
    """
    # Check inputs
    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert Sigmas.shape[-2] == Sigmas.shape[-1] == D
    if Ls is not None:
        assert Ls.shape[-2] == Ls.shape[-1] == D
    else:
        Ls = np.linalg.cholesky(Sigmas)                              # (..., D, D)

    # Quadratic term
    lp = -0.5 * batch_mahalanobis(Ls, data - mus)                    # (...,)
    # Normalizer
    L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]     # (..., D)
    half_log_det = np.sum(np.log(abs(L_diag)), axis=-1)              # (...,)
    lp = lp - 0.5 * D * np.log(2 * np.pi) - half_log_det             # (...,)

    return lp


def multivariate_normal_logpdf(data, mus, Sigmas, mask=None):
    """
    Compute the log probability density of a multivariate Gaussian distribution.
    This will broadcast as long as data, mus, Sigmas have the same (or at
    least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The mean(s) of the Gaussian distribution(s)

    Sigmas : array_like (..., D, D)
        The covariances(s) of the Gaussian distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the multivariate Gaussian distribution(s).
    """
    # Check inputs
    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert Sigmas.shape[-2] == Sigmas.shape[-1] == D

    # If there's no mask, we can just use the standard log pdf code
    if mask is None:
        return _multivariate_normal_logpdf(data, mus, Sigmas)

    # Otherwise we need to separate the data into sets with the same mask,
    # since each one will entail a different covariance matrix.
    #
    # First, determine the output shape. Allow mus and Sigmas to
    # have different shapes; e.g. many Gaussians with the same
    # covariance but different means.
    shp1 = np.broadcast(data, mus).shape[:-1]
    shp2 = np.broadcast(data[..., None], Sigmas).shape[:-2]
    assert len(shp1) == len(shp2)
    shp = tuple(max(s1, s2) for s1, s2 in zip(shp1, shp2))

    # Broadcast the data into the full shape
    full_data = np.broadcast_to(data, shp + (D,))

    # Get the full mask
    assert mask.dtype == bool
    assert mask.shape == data.shape
    full_mask = np.broadcast_to(mask, shp + (D,))

    # Flatten the mask and get the unique values
    flat_data = flatten_to_dim(full_data, 1)
    flat_mask = flatten_to_dim(full_mask, 1)
    unique_masks, mask_index = np.unique(flat_mask, return_inverse=True, axis=0)

    # Initialize the output
    lls = np.nan * np.ones(flat_data.shape[0])

    # Compute the log probability for each mask
    for i, this_mask in enumerate(unique_masks):
        this_inds = np.where(mask_index == i)[0]
        this_D = np.sum(this_mask)
        if this_D == 0:
            lls[this_inds] = 0
            continue

        this_data = flat_data[np.ix_(this_inds, this_mask)]
        this_mus = mus[..., this_mask]
        this_Sigmas = Sigmas[np.ix_(*[np.ones(sz, dtype=bool) for sz in Sigmas.shape[:-2]], this_mask, this_mask)]

        # Precompute the Cholesky decomposition
        this_Ls = np.linalg.cholesky(this_Sigmas)

        # Broadcast mus and Sigmas to full shape and extract the necessary indices
        this_mus = flatten_to_dim(np.broadcast_to(this_mus, shp + (this_D,)), 1)[this_inds]
        this_Ls = flatten_to_dim(np.broadcast_to(this_Ls, shp + (this_D, this_D)), 2)[this_inds]

        # Evaluate the log likelihood
        lls[this_inds] = _multivariate_normal_logpdf(this_data, this_mus, this_Sigmas, Ls=this_Ls)

    # Reshape the output
    assert np.all(np.isfinite(lls))
    return np.reshape(lls, shp)


def expected_multivariate_normal_logpdf(E_xs, E_xxTs, E_mus, E_mumuTs, Sigmas, Ls=None):
    """
    Compute the expected log probability density of a multivariate Gaussian distribution.
    This will broadcast as long as data, mus, Sigmas have the same (or at
    least be broadcast compatible along the) leading dimensions.
    Parameters
    ----------
    E_xs : array_like (..., D)
        The expected value of the points at which to evaluate the log density
    E_xxTs : array_like (..., D, D)
        The second moment of the points at which to evaluate the log density
    E_mus : array_like (..., D)
        The expected mean(s) of the Gaussian distribution(s)
    E_mumuTs : array_like (..., D, D)
        The second moment of the mean
    Sigmas : array_like (..., D, D)
        The covariances(s) of the Gaussian distribution(s)
    Ls : array_like (..., D, D)
        Optionally pass in the Cholesky decomposition of Sigmas
    Returns
    -------
    lps : array_like (...,)
        Expected log probabilities under the multivariate Gaussian distribution(s).
    TODO
    ----
    - Allow for uncertainty in the covariance as well.
    """
    # Check inputs
    D = E_xs.shape[-1]
    assert E_xxTs.shape[-2] == E_xxTs.shape[-1] == D
    assert E_mus.shape[-1] == D
    assert E_mumuTs.shape[-2] == E_mumuTs.shape[-1] == D
    assert Sigmas.shape[-2] == Sigmas.shape[-1] == D
    if Ls is not None:
        assert Ls.shape[-2] == Ls.shape[-1] == D
    else:
        Ls = np.linalg.cholesky(Sigmas)                              # (..., D, D)

    # TODO: Figure out how to perform this computation without explicit inverse
    Sigma_invs = np.linalg.inv(Sigmas)

    # Compute  E[(x-mu)^T Sigma^{-1}(x-mu)]
    #        = Tr(Sigma^{-1} E[(x-mu)(x-mu)^T])
    #        = Tr(Sigma^{-1} E[xx^T - x mu^T - mu x^T + mu mu^T])
    #        = Tr(Sigma^{-1} (E[xx^T - E[x]E[mu]^T - E[mu]E[x]^T + E[mu mu^T]]))
    #        = Tr(Sigma^{-1} A)
    #        = Tr((LL^T)^{-1} A)
    #        = Tr(L^{-1} A L^{-T} )
    #        = sum_{ij} [Sigma^{-1}]_{ij} * A_{ij}
    # where
    #
    # A = E[xx^T - E[x]E[mu]^T - E[mu]E[x]^T + E[mu mu^T]]
    #
    # However, since Sigma^{-1} is symmetric, we get the same
    # answer with
    #
    # A = E[xx^T - 2 * E[x]E[mu]^T + E[mu mu^T]]
    #
    E_xmuT = E_xs[..., :, None] * E_mus[..., None, :]
    # E_muxT = np.swapaxes(E_xmuT, -1, -2)
    # As = E_xxTs - E_xmuT - E_muxT + E_mumuTs
    As = E_xxTs - 2 * E_xmuT + E_mumuTs
    lp = -0.5 * np.sum(Sigma_invs * As, axis=(-2, -1))

    # Normalizer
    L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]     # (..., D)
    half_log_det = np.sum(np.log(abs(L_diag)), axis=-1)              # (...,)
    lp = lp - 0.5 * D * np.log(2 * np.pi) - half_log_det             # (...,)

    return lp


class DiagonalGaussian(Distribution):
    def __init__(self, mu=None, sigmasq=None, dim=None):
        assert (mu and sigmasq) or dim, "Constructor requires mean and variance or data dimension."
        if mu and sigmasq:
            self.dim = mu.shape[-1]
            self.mu = mu
            self.sigmasq = sigmasq
        else:
            assert isinstance(dim, int)
            self.dim = dim
            self.mu = np.zeros(dim)
            self.sigmasq = np.ones(dim)

    @property
    def params(self):
        return self.mu, self._sqrt_sigmasq

    @params.setter
    def params(self, value):
        self.mu, self._sqrt_sigmasq = value

    @property
    def sigmasq(self):
        return self._sqrt_sigmasq**2

    @sigmasq.setter
    def sigmasq(self, value):
        self._sqrt_sigmasq = np.sqrt(value)

    def log_probability(self, data, mask=None):
        return diagonal_gaussian_logpdf(data, self.mu, self.sigmasq, mask=mask)

    def m_step(self, datas, weights=None, expected_stats=None):
        if weights is None:
            weights = [np.ones(data.shape[0]) for data in datas]

        if expected_stats is None:
            dim = self.mu.shape[-1]
            Ex = np.zeros(dim)
            Exsq = np.zeros(dim)

            w_tot = 1e-16
            for w, x in zip(weights, datas):
                w_tot += np.sum(w)
                Ex += np.sum(w[:, None] * x, axis=0)
                Exsq += np.sum(w[:, None], x**2, axis=0)

            Ex /= w_tot
            Exsq /= w_tot

        else:
            Ex, ExxT = expected_stats

        self.mu = Ex
        self.sigmasq = Exsq - Ex**2


def diagonal_gaussian_logpdf(data, mus, sigmasqs, mask=None):
    """
    Compute the log probability density of a Gaussian distribution with
    a diagonal covariance.  This will broadcast as long as data, mus,
    sigmas have the same (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The mean(s) of the Gaussian distribution(s)

    sigmasqs : array_like (..., D)
        The diagonal variances(s) of the Gaussian distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the diagonal Gaussian distribution(s).
    """
    # Check inputs
    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert sigmasqs.shape[-1] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    normalizer = -0.5 * np.log(2 * np.pi * sigmasqs)
    return np.sum((normalizer - 0.5 * (data - mus)**2 / sigmasqs) * mask, axis=-1)


class MultivariateStudentsT(Distribution):
    def __init__(self, mu=None, Sigma=None, nu=None, dim=None):
        assert (mu and Sigma and nu) or dim, "Constructor requires either parameters or data dimension."
        if mu and Sigma and nu:
            self.dim = mu.shape[-1]
            self.mu = mu
            self.Sigma = Sigma
            self.nu = nu
        else:
            assert isinstance(dim, int)
            self.dim = dim
            self.mu = np.zeros(dim)
            self.Sigma = np.eye(dim)
            self.nu = 1.0

    @property
    def params(self):
        return self.mu, self._sqrt_sigmasq, self._sqrt_nu

    @params.setter
    def params(self, value):
        self.mu, self._sqrt_sigmasq, self._sqrt_nu = value

    @property
    def Sigma(self):
        return np.matmul(self._Sigma_chol, np.swapaxes(self._Sigma_chol, -1, -2))

    @Sigma.setter
    def Sigma(self, value):
        self._Sigma_chol = np.linalg.cholesky(value)

    @property
    def nu(self):
        return self._sqrt_nu**2

    @nu.setter
    def nu(self, value):
        self._sqrt_nu = np.sqrt(value)

    def log_probability(self, data, mask=None):
        return multivariate_studentst_logpdf(data, self.mu, self.Sigma, self.nu, Ls=self._Sigma_chol)

    def m_step(self, datas, weights=None):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t for
        complete details.
        """
        if weights is None:
            weights = [np.ones(data.shape[0]) for data in datas]

        self._m_step_mu_sigma(datas, weights)
        self._m_step_nu(datas, weights)

    def _m_step_mu_sigma(self, datas, weights):
        dim = self.dim

        # Estimate the precisions w for each data point
        taus = []
        for y in datas:
            alpha = 0.5 * (self.nu  + dim)
            beta = 0.5 * (self.nu + batch_mahalanobis(self._Sigma_chol, y - self.mu))
            taus.append(alpha / beta)

        # Compute expected sufficient statistics given these precisions
        # The precisions act as weights on each data point
        Ex = np.zeros(dim)
        ExxT = np.zeros((dim, dim))
        w_tot = 1e-16
        for tau, w, x in zip(tau, weights, datas):
            w_eff = tau * w
            w_tot += np.sum(w_eff)
            Ex += np.sum(w_eff[:, None] * x, axis=0)
            ExxT += np.einsum('n,ni,nj', w_eff, x, x)

        Ex /= w_tot
        ExxT /= w_tot

        # Update the mean and covariance
        self.mu = Ex
        self.Sigma = ExxT - np.outer(Ex, Ex)

    def _m_step_nu(self, datas, weights):
        """
        The shape parameter nu determines a gamma prior.  We have

            tau_n ~ Gamma(nu/2, nu/2)
            y_n ~ N(mu, Sigma / tau_n)

        To update nu, we do EM and optimize the expected log likelihood using
        a generalized Newton's method.  See the notebook in doc/students_t for
        complete details.
        """

        # Compute the precisions w for each data point
        E_tau = 0
        E_logtau = 0
        w_tot = 0
        for w, y in zip(weights, datas):
            alpha = 0.5 * (self.nu  + dim)
            beta = 0.5 * (self.nu + batch_mahalanobis(self._Sigma_chol, y - self.mu))

            E_tau += np.sum(w * (alpha / beta), axis=0)
            E_logtau += np.sum(w * (digamma(alpha) - np.log(beta)), axis=0)
            w_tot += np.sum(w, axis=0)

        E_taus /= w_tot
        E_logtaus /= w_tot

        # Solve for the most likely value of nu
        self.nu = generalized_newton_studentst_dof(E_tau, E_logtau)


def multivariate_studentst_logpdf(data, mus, Sigmas, nus, Ls=None):
    """
    Compute the log probability density of a multivariate Student's t distribution.
    This will broadcast as long as data, mus, Sigmas, nus have the same (or at
    least be broadcast compatible along the) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The mean(s) of the t distribution(s)

    Sigmas : array_like (..., D, D)
        The covariances(s) of the t distribution(s)

    nus : array_like (...,)
        The degrees of freedom of the t distribution(s)

    Ls : array_like (..., D, D)
        Optionally pass in the Cholesky decomposition of Sigmas

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the multivariate Gaussian distribution(s).
    """
    # Check inputs
    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert Sigmas.shape[-2] == Sigmas.shape[-1] == D
    if Ls is not None:
        assert Ls.shape[-2] == Ls.shape[-1] == D
    else:
        Ls = np.linalg.cholesky(Sigmas)                              # (..., D, D)

    # Quadratic term
    q = batch_mahalanobis(Ls, data - mus) / nus                      # (...,)
    lp = - 0.5 * (nus + D) * np.log1p(q)                             # (...,)

    # Normalizer
    lp = lp + gammaln(0.5 * (nus + D)) - gammaln(0.5 * nus)          # (...,)
    lp = lp - 0.5 * D * np.log(np.pi) - 0.5 * D * np.log(nus)        # (...,)
    L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]     # (..., D)
    half_log_det = np.sum(np.log(abs(L_diag)), axis=-1)              # (...,)
    lp = lp - half_log_det

    return lp


def expected_multivariate_studentst_logpdf(E_xs, E_xxTs, E_mus, E_mumuTs, Sigmas, nus, Ls=None):
    """
    Compute the expected log probability density of a multivariate Gaussian distribution.
    This will broadcast as long as data, mus, Sigmas have the same (or at
    least be broadcast compatible along the) leading dimensions.
    Parameters
    ----------
    E_xs : array_like (..., D)
        The expected value of the points at which to evaluate the log density
    E_xxTs : array_like (..., D, D)
        The second moment of the points at which to evaluate the log density
    E_mus : array_like (..., D)
        The expected mean(s) of the Gaussian distribution(s)
    E_mumuTs : array_like (..., D, D)
        The second moment of the mean
    Sigmas : array_like (..., D, D)
        The covariances(s) of the Gaussian distribution(s)
    Ls : array_like (..., D, D)
        Optionally pass in the Cholesky decomposition of Sigmas
    Returns
    -------
    lps : array_like (...,)
        Expected log probabilities under the multivariate Gaussian distribution(s).
    TODO
    ----
    - Allow for uncertainty in the covariance Sigmas and dof nus as well.
    """
    # Check inputs
    D = E_xs.shape[-1]
    assert E_xxTs.shape[-2] == E_xxTs.shape[-1] == D
    assert E_mus.shape[-1] == D
    assert E_mumuTs.shape[-2] == E_mumuTs.shape[-1] == D
    assert Sigmas.shape[-2] == Sigmas.shape[-1] == D
    if Ls is not None:
        assert Ls.shape[-2] == Ls.shape[-1] == D
    else:
        Ls = np.linalg.cholesky(Sigmas)                              # (..., D, D)

    # TODO: Figure out how to perform this computation without explicit inverse
    Sigma_invs = np.linalg.inv(Sigmas)

    # Compute  E[(x-mu)^T Sigma^{-1}(x-mu)]
    #        = Tr(Sigma^{-1} E[(x-mu)(x-mu)^T])
    #        = Tr(Sigma^{-1} E[xx^T - 2 x mu^T + mu mu^T])
    #        = Tr(Sigma^{-1} (E[xx^T - 2 E[x]E[mu]^T + E[mu mu^T]]))
    #        = Tr(Sigma^{-1} A)
    #        = Tr((LL^T)^{-1} A)
    #        = Tr(L^{-1} A L^{-T} )
    #        = sum_{ij} [Sigma^{-1}]_{ij} * A_{ij}
    # where
    #
    # A = E[xx^T - 2 E[x]E[mu]^T + E[mu mu^T]]
    #
    As = E_xxTs - 2 * E_xs[..., :, None] * E_mus[..., None, :] + E_mumuTs   # (..., D, D)
    q = np.sum(Sigma_invs * As, axis=(-2, -1)) / nus                        # (...,)
    lp = - 0.5 * (nus + D) * np.log1p(q)                                    # (...,)

    # Normalizer
    L_diag = np.reshape(Ls, Ls.shape[:-2] + (-1,))[..., ::D + 1]            # (..., D)
    half_log_det = np.sum(np.log(abs(L_diag)), axis=-1)                     # (...,)
    lp = lp - 0.5 * D * np.log(2 * np.pi) - half_log_det                    # (...,)

    return lp


def generalized_newton_studentst_dof(E_tau, E_logtau, nu0=1, max_iter=100, nu_min=1e-3, nu_max=20, tol=1e-8, verbose=False):
    """
    Generalized Newton's method for the degrees of freedom parameter, nu,
    of a Student's t distribution.  See the notebook in the doc/students_t
    folder for a complete derivation.
    """
    delbo = lambda nu: 1/2 * (1 + np.log(nu/2)) - 1/2 * digamma(nu/2) + 1/2 * E_logtau - 1/2 * E_tau
    ddelbo = lambda nu: 1/(2 * nu) - 1/4 * polygamma(1, nu/2)

    dnu = np.inf
    nu = nu0
    for itr in range(max_iter):
        if abs(dnu) < tol:
            break

        if nu < nu_min or nu > nu_max:
            warn("generalized_newton_studentst_dof fixed point grew beyond "
                 "bounds [{},{}].".format(nu_min, nu_max))
            nu = np.clip(nu, nu_min, nu_max)
            break

        # Perform the generalized Newton update
        a = -nu**2 * ddelbo(nu)
        b = delbo(nu) - a / nu
        assert a > 0 and b < 0, "generalized_newton_studentst_dof encountered invalid values of a,b"
        dnu = -a / b - nu
        nu = nu + dnu

    if itr == max_iter - 1:
        warn("generalized_newton_studentst_dof failed to converge"
             "at tolerance {} in {} iterations.".format(tol, itr))

    return nu


class DiagonalStudentsT(Distribution):
    def __init__(self, mu=None, sigmasq=None, nu=None, dim=None):
        assert (mu and sigmasq and nu) or dim, "Constructor requires either parameters or data dimension."
        if mu and sigmasq and nu:
            self.dim = mu.shape[-1]
            self.mu = mu
            self.sigmasq = sigmasq
            self.nu = nu
        else:
            assert isinstance(dim, int)
            self.dim = dim
            self.mu = np.zeros(dim)
            self.sigmasq = np.ones(dim)
            self.nu = np.ones(dim)

    @property
    def params(self):
        return self.mu, self._sqrt_sigmasq, self._sqrt_nu

    @params.setter
    def params(self, value):
        self.mu, self._sqrt_sigmasq, self._sqrt_nu = value

    @property
    def sigmasq(self):
        return self._sqrt_sigmasq**2

    @sigmasq.setter
    def sigmasq(self, value):
        self._sqrt_sigmasq = np.sqrt(value)

    @property
    def nu(self):
        return self._sqrt_nu**2

    @nu.setter
    def nu(self, value):
        self._sqrt_nu = np.sqrt(value)

    def log_probability(self, data, mask=None):
        return independent_studentst_logpdf(data, self.mu, self.sigmasq, self.nu, mask=mask)


    def m_step(self, datas, weights=None):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t for
        complete details.
        """
        if weights is None:
            weights = [np.ones(data.shape[0]) for data in datas]

        self._m_step_mu_sigma(datas, weights)
        self._m_step_nu(datas, weights)

    def _m_step_mu_sigma(self, datas, weights):
        dim = self.dim

        # Estimate the precisions w for each data point
        taus = []
        for y in datas:
            alpha = 0.5 * (self.nu  + dim)
            beta = 0.5 * (self.nu + (y - self.mu)**2 / self.sigmasq)
            taus.append(alpha / beta)

        # Compute expected sufficient statistics given these precisions
        # The precisions act as weights on each data point
        Ex = np.zeros(dim)
        Exsq = np.zeros(dim)
        w_tot = 1e-16
        for tau, w, x in zip(tau, weights, datas):
            w_eff = tau * w
            Ex += np.sum(w_eff[:, None] * x, axis=0)
            Exsq += np.sum(w_eff[:, None] * x**2, axis=0)
            w_tot += np.sum(w_eff)

        Ex /= w_tot
        Exsq /= w_tot

        # Update the mean and covariance
        self.mu = Ex
        self.sigmasq = Exsq - Ex**2

    def _m_step_nu(self, datas, weights):
        """
        The shape parameter nu determines a gamma prior.  We have

            tau_n ~ Gamma(nu/2, nu/2)
            y_n ~ N(mu, Sigma / tau_n)

        To update nu, we do EM and optimize the expected log likelihood using
        a generalized Newton's method.  See the notebook in doc/students_t for
        complete details.
        """
        dim = self.dim

        # Compute the precisions w for each data point
        E_tau = 0
        E_logtau = 0
        w_tot = 0
        for w, y in zip(weights, datas):
            alpha = 0.5 * (self.nu + 1)
            beta = 0.5 * (self.nu + (y - self.mu)**2 / self.sigmasq)

            E_tau += np.sum(w * (alpha / beta), axis=0)
            E_logtau += np.sum(w * (digamma(alpha) - np.log(beta)), axis=0)
            w_tot += np.sum(w, axis=0)

        E_taus /= w_tot
        E_logtaus /= w_tot

        # Solve for the most likely value of nu
        nu = np.zeros(dim)
        for d in range(D):
            nu[d] = generalized_newton_studentst_dof(E_tau[d], E_logtau[d])
        self.nu = nu


def independent_studentst_logpdf(data, mus, sigmasqs, nus, mask=None):
    """
    Compute the log probability density of a Gaussian distribution with
    a diagonal covariance.  This will broadcast as long as data, mus,
    sigmas have the same (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The mean(s) of the Student's t distribution(s)

    sigmasqs : array_like (..., D)
        The diagonal variances(s) of the Student's t distribution(s)

    nus : array_like (..., D)
        The degrees of freedom of the Student's t distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the Student's t distribution(s).
    """
    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert sigmasqs.shape[-1] == D
    assert nus.shape[-1] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    normalizer = gammaln(0.5 * (nus + 1)) - gammaln(0.5 * nus)
    normalizer = normalizer - 0.5 * (np.log(np.pi) + np.log(nus) + np.log(sigmasqs))
    ll = normalizer - 0.5 * (nus + 1) * np.log(1.0 + (data - mus)**2 / (sigmasqs * nus))
    return np.sum(ll * mask, axis=-1)


class Bernoulli(Distribution):
    def __init__(self, p=None, dim=None):
        assert p or dim, "Constructor requires either parameters or data dimension."
        if p:
            self.dim = np.atleast_1d(p).shape[-1]
            self.p = np.atleast_1d(p)
        else:
            assert isinstance(dim, int)
            self.dim = dim
            self.p = 0.5 * np.ones(dim)

    @property
    def params(self):
        return self._logit_p

    @params.setter
    def params(self, value):
        self._logit_p = value

    @property
    def p(self):
        return logistic(self._logit_p)

    @o.setter
    def p(self, value):
        self._logit_p = logit(value)

    def log_probability(self, data, mask=None):
        return bernoulli_logpdf(data, self._logit_p, mask=mask)

    def m_step(self, datas, weights=None, expected_stats=None):
        if weights is None:
            weights = [np.ones(data.shape[0]) for data in datas]

        if expected_stats is None:
            dim = self.dim
            Ex = np.zeros(dim)

            w_tot = 1e-16
            for w, x in zip(weights, datas):
                w_tot += np.sum(w)
                Ex += np.sum(w[:, None] * x, axis=0)

            Ex /= w_tot

        else:
            Ex = expected_stats

        self.p = Ex


def bernoulli_logpdf(data, logit_ps, mask=None):
    """
    Compute the log probability density of a Bernoulli distribution.
    This will broadcast as long as data and logit_ps have the same
    (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    logit_ps : array_like (..., D)
        The logit(s) log p / (1 - p) of the Bernoulli distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the Bernoulli distribution(s).
    """
    D = data.shape[-1]
    assert (data.dtype == int or data.dtype == bool)
    assert data.min() >= 0 and data.max() <= 1
    assert logit_ps.shape[-1] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    # Evaluate log probability
    # log Pr(x | p) = x * log(p) + (1-x) * log(1-p)
    #               = x * log(p / (1-p)) + log(1-p)
    #               = x * log(p / (1-p)) - log(1/(1-p))
    #               = x * log(p / (1-p)) - log(1 + p/(1-p)).
    #
    # Let u = log (p / (1-p)) = logit(p), then
    #
    # log Pr(x | p) = x * u - log(1 + e^u)
    #               = x * u - log(e^0 + e^u)
    #               = x * u - log(e^m * (e^-m + e^(u-m))
    #               = x * u - m - log(exp(-m) + exp(u-m)).
    #
    # This holds for any m. we choose m = max(0, u) to avoid overflow.
    m = np.maximum(0, logit_ps)
    lls = data * logit_ps - m - np.log(np.exp(-m) + np.exp(logit_ps - m))
    return np.sum(lls * mask, axis=-1)


class Poisson(Distribution):
    def __init__(self, lmbda=None, dim=None):
        assert lmbda or dim, "Constructor requires either parameters or data dimension."
        if lmbda:
            self.dim = np.atleast_1d(lmbda).shape[-1]
            self.lmbda = np.atleast_1d(lmbda)
        else:
            assert isinstance(dim, int)
            self.dim = dim
            self.lmbda = np.ones(dim)

    @property
    def params(self):
        return self._log_lambda

    @params.setter
    def params(self, value):
        self._log_lambda = value

    @property
    def lmbda(self):
        return np.exp(self._log_lambda)

    @lmbda.setter
    def lmbda(self, value):
        self._log_lambda = np.log(value + 1e-16)

    def log_probability(self, data, mask=None):
        return poisson_logpdf(data, self.lmbda, mask=mask)

    def m_step(self, datas, weights=None, expected_stats=None):
        if weights is None:
            weights = [np.ones(data.shape[0]) for data in datas]

        if expected_stats is None:
            dim = self.dim
            Ex = np.zeros(dim)

            w_tot = 1e-16
            for w, x in zip(weights, datas):
                w_tot += np.sum(w)
                Ex += np.sum(w[:, None] * x, axis=0)

            Ex /= w_tot

        else:
            Ex = expected_stats

        self.lmbda = Ex


def poisson_logpdf(data, lambdas, mask=None):
    """
    Compute the log probability density of a Poisson distribution.
    This will broadcast as long as data and lambdas have the same
    (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    lambdas : array_like (..., D)
        The rates of the Poisson distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the Poisson distribution(s).
    """
    D = data.shape[-1]
    assert data.dtype in (int, np.int8, np.int16, np.int32, np.int64)
    assert lambdas.shape[-1] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    # Compute log pdf
    lls = -gammaln(data + 1) - lambdas + data * np.log(lambdas)
    return np.sum(lls * mask, axis=-1)


class Categorical(Distribution):
    def __init__(self, p=None, num_classes=None, dim=1):
        assert p or (num_classes and dim), "Constructor requires either parameters or number of classes and data dimension."
        if p:
            self.p = np.atleast_2d(p)
            self.dim, self.num_classes = self.p.shape[:-1], p.shape[-1]
            assert np.allclose(self.p.sum(axis=-1), 1.0), "p must be a normalized distribution over classes."
        else:
            assert isinstance(num_classes, int)
            self.num_classes = num_classes
            assert isinstance(dim, int)
            self.dim = dim
            self.p = np.ones((dim, num_classes)) / num_classes

    @property
    def params(self):
        return self._logit_p

    @params.setter
    def params(self, value):
        self._logit_p = value

    @property
    def p(self):
        return np.exp(self._logit_p - logsumexp(self._logit_p, axis=-1))

    @p.setter
    def p(self, value):
        self._logit_p = np.log(value + 1e-16)

    def log_probability(self, data, mask=None):
        return categorical_logpdf(data, self._logit_p, mask=mask)

    def m_step(self, datas, weights=None, expected_stats=None):
        if weights is None:
            weights = [np.ones(data.shape[0]) for data in datas]

        if expected_stats is None:
            dim = self.dim
            Ex = np.zeros(dim)

            w_tot = 1e-16
            for w, x in zip(weights, datas):
                w_tot += np.sum(w)
                Ex += np.einsum('n,n...c->...c', w, one_hot(x, self.num_classes))
            Ex /= w_tot

        else:
            Ex = expected_stats

        self.p = Ex / Ex.sum(axis=-1, keepdims=True)


def categorical_logpdf(data, logits, mask=None):
    """
    Compute the log probability density of a categorical distribution.
    This will broadcast as long as data and logits have the same
    (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (...,) int (0 <= data < C)
        The points at which to evaluate the log density

    logits : array_like (..., C)
        The logits of the categorical distribution(s) with C classes

    mask : array_like (...,) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the categorical distribution(s).
    """
    C = logits.shape[-1]
    assert data.dtype in (int, np.int8, np.int16, np.int32, np.int64)
    assert np.all((data >= 0) & (data < C))

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    logits = logits - logsumexp(logits, axis=-1, keepdims=True) # (...,C)
    return logits[data] * mask                                  # (...,)


class VonMises(Distribution):
    def __init__(self, mu=None, kappa=None, dim=None):
        assert (mu and kappa) or dim
        if mu and kappa:
            self.dim = mu.shape
            self.mu = mu
            self.kappa = kappa
        else:
            self.dim = dim
            self.mu = np.ones(dim)
            self.kappa = np.ones(dim)

    @property
    def params(self):
        return self._mu_unnorm, self._log_kappa

    @params.setter
    def params(self, value):
        self._mu_unnorm, self._log_kappa = value

    @property
    def mu(self):
        R = np.linalg.norm(self._mu_unnorm, axis=-1, keepdims=True)
        assert np.all(R > 0)
        return self._mu_unnorm / R

    @mu.setter
    def mu(self, value):
        self._mu_unnorm = value

    @property
    def kappa(self):
        return np.exp(self._log_kappa)

    @kappa.setter
    def kappa(self, value):
        return np.log(value + 1e-16)

    def log_probability(self, data, mask=None):
        return vonmises_logpdf(data, self.mu, self.kappa, mask=mask)

    def m_step(self, datas, weights=None, expected_stats=None):
        if weights is None:
            weights = [np.ones(data.shape[0]) for data in datas]

        dim = self.dim
        if expected_stats is None:
            Ex = np.zeros(dim)
            w_tot = 1e-16
            for w, x in zip(weights, datas):
                w_tot += np.sum(w)
                Ex += np.einsum('n,n...->...', w, x)
            Ex /= w_tot

        else:
            Ex = expected_stats

        # The MLE of the mean is available in closed form
        R = np.linalg.norm(Ex, axis=-1)
        self.mu = Ex / R

        # We use Wikipedia's simple approximation for the concentration
        self.kappa = R * (dim[-1] - R**2) / (1 - R**2)
        assert self.kappa > 0


def vonmises_logpdf(data, mus, kappas, mask=None):
    """
    Compute the log probability density of a von Mises distribution.
    This will broadcast as long as data, mus, and kappas have the same
    (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The means of the von Mises distribution(s)

    kappas : array_like (..., D)
        The concentration of the von Mises distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the von Mises distribution(s).
    """
    try:
        from autograd.scipy.special import i0
    except:
        raise Exception("von Mises relies on the function autograd.scipy.special.i0. "
                        "This is present in the latest Github code, but not on pypi. "
                        "Please use the Github version of autograd instead.")

    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert kappas.shape[-1] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    ll = kappas * np.cos(data - mus) - np.log(2 * np.pi) - np.log(i0(kappas))
    return np.sum(ll * mask, axis=-1)
