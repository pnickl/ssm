from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
from autograd import hessian

from ssm.util import ensure_args_are_lists, \
    logistic, logit, softplus, inv_softplus
from ssm.preprocessing import interpolate_data, pca_with_imputation
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs


# Observation models for linear dynamical systems
class LDSObservations(object):
    def __init__(self, data_dim, state_dim, input_dim=0):
        self.data_dim = data_dim
        self.state_dim = state_dim
        self.input_dim = input_dim

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def permute(self, perm):
        pass

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        pass

    def initialize_from_arhmm(self, arhmm, pca):
        pass

    def log_prior(self):
        return 0

    def log_likelihoods(self, states, data, input, mask, tag):
        raise NotImplementedError

    def forward(self, x, input=None, tag=None):
        raise NotImplemented

    def invert(self, data, input=None, mask=None, tag=None):
        raise NotImplemented

    def sample(self, states, input=None, tag=None):
        raise NotImplementedError

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError

    def hessian_log_emissions_prob(self, states, data, input, mask, tag):
        warn("Analytical Hessian is not implemented for this class. \
              Optimization via Laplace-EM may be slow. Consider using an \
              alternative posterior and inference method.")
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        obj = lambda xt, datat, inputt, maskt: \
            self.log_likelihoods(datat[None,:], inputt[None,:], maskt[None,:], tag, xt[None,:])[0, 0]
        hess = hessian(obj)
        terms = np.array([np.squeeze(hess(xt, datat, inputt, maskt))
                          for xt, datat, inputt, maskt in zip(states, data, input, mask)])
        return terms

    def m_step(self, posteriors, datas, inputs, masks, tags,
               weights=None, optimizer="bfgs", maxiter=100, **kwargs):
        """
        If M-step in Laplace-EM cannot be done in closed form for the emissions, default to SGD.
        """
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

        # expected log likelihood
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = 0
            obj += self.log_prior()
            for posterior, data, input, mask, tag, weight, in \
                zip(posteriors, datas, inputs, masks, tags, weights):
                states = posterior.sample()
                obj += np.sum(self.log_likelihoods(states, data, input, mask, tag))
            return -obj / T

        # Optimize emissions log-likelihood
        self.params = optimizer(_objective, self.params,
                                num_iters=maxiter,
                                suppress_warnings=True,
                                **kwargs)


# Many emissions models start with a linear layer
class _LinearObservations(LDSObservations):
    """
    A simple linear mapping from continuous states x to data y.

        E[y | x] = Cx + d + Fu

    where C is an emission matrix, d is a bias, F an input matrix,
    and u is an input.
    """
    def __init__(self, data_dim, state_dim, input_dim=0):
        super(_LinearEmissions, self).__init__(data_dim, state_dim, input_dim=input_dim)

        # Initialize linear layer.  Set _Cs to be private so that it can be
        # changed in subclasses.
        self.C = npr.randn(data_dim, state_dim)
        self.D = npr.randn(data_dim, input_dim)
        self.d = npr.randn(data_dim)

    @property
    def params(self):
        return self.C, self.D, self.d

    @params.setter
    def params(self, value):
        self.C, self.D, self.d = value

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        C, D, d = self.C, self.D, self.d
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        bias = input.dot(D.T) + d

        if not np.all(mask):
            data = interpolate_data(data, mask)
            # We would like to find the PCA coordinates in the face of missing data
            # To do so, alternate between running PCA and imputing the missing entries
            for itr in range(25):
                mu = (data - bias).dot(C_pseudoinv)
                data[:, ~mask[0]] = (mu.dot(C.T) + bias)[:, ~mask[0]]

        # Project data to get the mean
        return (data - bias).dot(C_pseudoinv)

    def forward(self, states, input, tag):
        return np.dot(states, self.C.T) + np.dot(input, self.D.T) + self.d

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        Keff = 1 if self.single_subspace else self.K

        # First solve a linear regression for data given input
        if self.input_dim > 0:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(fit_intercept=False)
            lr.fit(np.vstack(inputs), np.vstack(datas))
            self.D = lr.coef_

        # Compute residual after accounting for input
        resids = [data - np.dot(input, self.D.T) for data, input in zip(datas, inputs)]

        # Run PCA to get a linear embedding of the data
        pca, xs, ll = pca_with_imputation(self.state_dim, resids, masks, num_iters=num_iters)

        self.Cs = pca.components_.T
        self.ds = pca.mean_

        return pca


class _OrthogonalLinearObservations(_LinearEmissions):
    """
    A linear emissions matrix constrained such that the emissions matrix
    is orthogonal. Use the rational Cayley transform to parameterize
    the set of orthogonal emission matrices. See
    https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.5b02015
    for a derivation of the rational Cayley transform.
    """
    def __init__(self, data_dim, state_dim, input_dim=0):
        super(_OrthogonalLinearEmissions, self).__init__(data_dim, state_dim, input_dim=input_dim)

        # Initialize linear layer
        assert data_dim > state_dim
        self._M = npr.randn(state_dim, state_dim)
        self._A = npr.randn(data_dim - state_dim, state_Dim)
        self.D = npr.randn(data_dim, input_dim)
        self.d = npr.randn(data_dim)

        # Set the emission matrix to be a random orthogonal matrix
        C0 = npr.randn(data_dim, state_dim)
        C0 = np.linalg.svd(C0, full_matrices=False)[0]
        self.C = C0

    @property
    def C(self):
        state_dim = self.state_dim
        T = lambda X: np.swapaxes(X, -1, -2)

        B = 0.5 * (self._M - T(self._M))    # Bs is skew symmetric
        F = np.matmul(T(self._A), self._A) - B
        trm1 = np.concatenate((np.eye(state_dim) - F, 2 * self._A), axis=1)
        trm2 = np.eye(D) + F
        Cs = T(np.linalg.solve(T(trm2), T(trm1)))
        assert np.allclose(C.T @ C, np.eye(state_dim))
        return Cs

    @Cs.setter
    def Cs(self, value):
        N, D = self.data_dim, self.state_dim
        T = lambda X: np.swapaxes(X, -1, -2)

        # Make sure value is the right shape and orthogonal
        assert value.shape == (N, D)
        assert np.allclose(np.matmul(T(value), value), np.eye(D))

        Q1, Q2 = value[:D, :], value[D:, :]
        F = T(np.linalg.solve(T(np.eye(D) + Q1), T(np.eye(D) - Q1)))
        # Bs = 0.5 * (T(Fs) - Fs) = 0.5 * (self._Ms - T(self._Ms)) -> _Ms = T(Fs)
        self._M = T(F)
        self._A = 0.5 * np.matmul(Q2, np.eye(D) + F)
        assert np.allclose(self.C, value)

    @property
    def params(self):
        return self._A, self._M, self.F, self.d

    @params.setter
    def params(self, value):
        self._As, self._Ms, self.Fs, self.ds = value


# Sometimes we just want a bit of additive noise on the observations
class _IdentityEmissions(Emissions):
    def __init__(self, data_dim, state_dim, input_dim=0):
        super(_IdentityEmissions, self).__init__(data_dim, state_dim, input_dim=input_dim)
        assert data_dim == state_dim

    def forward(self, x, input):
        return x

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Inverse is just the data
        """
        return np.copy(data)


# Allow general nonlinear emission models with neural networks
class _NeuralNetworkEmissions(Emissions):
    def __init__(self, data_dim, state_dim, input_dim=0, hidden_layer_sizes=(50,)):
        super(_NeuralNetworkEmissions, self).__init__(data_dim, state_dim, input_dim=input_dim)

        # Initialize the neural network weights
        assert data_dim > state_dim
        layer_sizes = (state_dim + input_dim,) + hidden_layer_sizes + (data_dim,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

    @property
    def params(self):
        return self.weights, self.biases

    @params.setter
    def params(self, value):
        self.weights, self.biases = value

    def permute(self, perm):
        pass

    def forward(self, x, input, tag):
        inputs = np.column_stack((x, input))
        for W, b in zip(self.weights, self.biases):
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs[:, None, :]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Inverse is... who knows!
        """
        return npr.randn(data.shape[0], self.state_dim)


# Observation models for SLDS
class _GaussianObservationsMixin(object):
    def __init__(self, data_dim, state_dim, input_dim=0, **kwargs):
        super(_GaussianEmissionsMixin, self).__init__(data_dim, state_dim, input_dim=input_dim, **kwargs)
        self.log_sigmasqs = -1 + npr.randn(data_dim)

    @property
    def params(self):
        return super(_GaussianEmissionsMixin, self).params + (self.log_sigmasqs,)

    @params.setter
    def params(self, value):
        self.log_sigmasqs = value[-1]
        super(_GaussianEmissionsMixin, self.__class__).params.fset(self, value[:-1])

    def log_likelihoods(self, states, data, input, mask, tag):
        mus = self.forward(states, input, tag)
        sigmasqs = np.exp(self.log_sigmasqs)
        return diagonal_gaussian_logpdf(data, mus, sigmasqs, mask=mask)

    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, states, input=None, tag=None):
        T = states.shape[0]
        mus = self.forward(states, input, tag)
        sigmasqs = np.exp(self.log_sigmasqs)
        return mus + np.sqrt(sigmasqs) * npr.randn(T, self.N)

    def smooth(self, states, data, input=None, mask=None, tag=None):
        return self.forward(vstates, input, tag)


class GaussianObservations(_GaussianObservationsMixin, _LinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        # pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        # self.inv_etas[:,...] = np.log(pca.noise_variance_)
        pass

    def hessian_log_emissions_prob(self, data, input, mask, tag, x):
        assert self.single_subspace, "Only implemented for a single emission model"
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        hess = -1.0 * self.Cs[0].T@np.diag( 1.0 / np.exp(self.inv_etas[0]) )@self.Cs[0]
        return np.tile(hess[None,:,:], (T, 1, 1))

    def m_step(self, posteriors, datas, inputs, masks, tags, **kwargs):
        # TODO: Fit the linear regression using posterior expectations

        # Return exact m-step updates for C, F, d, and inv_etas
        # stack across all datas
        x = np.vstack(continuous_expectations)
        u = np.vstack(inputs)
        y = np.vstack(datas)
        T, state_dim = np.shape(x)
        xb = np.hstack((np.ones((T,1)), x, u)) # design matrix
        params = np.linalg.lstsq(xb.T@xb, xb.T@y, rcond=None)[0].T
        self.d = params[:,0]
        self.C = params[:,1:state_dim+1]
        self.D = params[:,state_dim+1:]

        # TODO: Not quite right -- need the covariance of Sigma
        mu = np.dot(xb, params.T)
        Sigma = (y-mu).T@(y-mu) / T
        self.inv_sigmasqs = np.log(np.diag(Sigma))


class GaussianOrthogonalObservations(_GaussianObservationsMixin, _OrthogonalLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)

    def hessian_log_emissions_prob(self, data, input, mask, tag, x):
        # Return (T, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        hess = -1.0 * self.C.T @ np.diag( 1.0 / np.exp(self.inv_sigmasqs) ) @ self.C
        return np.tile(hess[None,:,:], (T, 1, 1))


class GaussianIdentityObservations(_GaussianObservationsMixin, _IdentityEmissions):
    pass


class GaussianNeuralNetworkObservations(_GaussianObservationsMixin, _NeuralNetworkEmissions):
    pass


class _StudentsTObservationsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_StudentsTEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_nus = np.log(4) * np.ones((1, N)) if single_subspace else np.log(4) * np.ones(K, N)

    @property
    def params(self):
        return super(_StudentsTEmissionsMixin, self).params + (self.inv_etas, self.inv_nus)

    @params.setter
    def params(self, value):
        super(_StudentsTEmissionsMixin, self.__class__).params.fset(self, value[:-2])

    def log_likelihoods(self, data, input, mask, tag, x):
        N, etas, nus = self.N, np.exp(self.inv_etas), np.exp(self.inv_nus)
        mus = self.forward(x, input, tag)

        resid = data[:, None, :] - mus
        z = resid / etas
        return -0.5 * (nus + N) * np.log(1.0 + (resid * z).sum(axis=2) / nus) + \
            gammaln((nus + N) / 2.0) - gammaln(nus / 2.0) - N / 2.0 * np.log(nus) \
            -N / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(etas), axis=1)

    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)
        taus = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        return mus[np.arange(T), z, :] + np.sqrt(etas[z] / taus) * npr.randn(T, self.N)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)


class StudentsTObservations(_StudentsTObservationsMixin, _LinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class StudentsTOrthogonalObservations(_StudentsTObservationsMixin, _OrthogonalLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class StudentsTIdentityObservations(_StudentsTEmissionsMixin, _IdentityEmissions):
    pass


class StudentsTNeuralNetworkObservations(_StudentsTObservationsMixin, _NeuralNetworkEmissions):
    pass


class _BernoulliObservationsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="logit", **kwargs):
        super(_BernoulliEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)

        self.link_name = link
        mean_functions = dict(
            logit=logistic
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            logit=logit
            )
        self.link = link_functions[link]

    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == bool or (data.dtype == int and data.min() >= 0 and data.max() <= 1)
        ps = self.mean(self.forward(x, input, tag))
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = data[:, None, :] * np.log(ps) + (1 - data[:, None, :]) * np.log(1 - ps)
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(np.clip(data, .1, .9))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        ps = self.mean(self.forward(x, input, tag))
        return (npr.rand(T, self.N) < ps[np.arange(T), z,:]).astype(int)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        ps = self.mean(self.forward(variational_mean, input, tag))
        return ps[:,0,:] if self.single_subspace else np.sum(ps * expected_states[:,:,None], axis=1)


class BernoulliLDSObservations(_BernoulliObservationsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def hessian_log_emissions_prob(self, data, input, mask, tag, x):
        """
        d/dx  (y - p) * C
            = -dpsi/dx (dp/d\psi)  C
            = -C p (1-p) C
        """
        assert self.single_subspace
        assert self.link_name == "logit"
        psi =  self.forward(x, input, tag)[:, 0, :]
        p = self.mean(psi)
        dp_dpsi = p * (1 - p)
        return np.einsum('tn, ni, nj ->tij', -dp_dpsi, self.Cs[0], self.Cs[0])


class BernoulliOrthogonalLDSObservations(_BernoulliObservationsMixin, _OrthogonalLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def hessian_log_emissions_prob(self, data, input, mask, tag, x):
        """
        d/dx  (y - p) * C
            = -dpsi/dx (dp/d\psi)  C
            = -C p (1-p) C
        """
        assert self.single_subspace
        assert self.link_name == "logit"
        psi =  self.forward(x, input, tag)[:, 0, :]
        p = self.mean(psi)
        dp_dpsi = p * (1 - p)
        return np.einsum('tn, ni, nj ->tij', -dp_dpsi, self.Cs[0], self.Cs[0])


class BernoulliIdentityLDSObservations(_BernoulliObservationsMixin, _IdentityEmissions):
    pass


class BernoulliNeuralNetworkEmissions(_BernoulliObservationsMixin, _NeuralNetworkEmissions):
    pass


class _PoissonObservationsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="log", bin_size=1.0, **kwargs):
        super(_PoissonEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)

        self.link_name = link
        self.bin_size = bin_size
        mean_functions = dict(
            log=lambda x: np.exp(x) * self.bin_size,
            softplus= lambda x: softplus(x) * self.bin_size
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            log=lambda rate: np.log(rate) - np.log(self.bin_size),
            softplus=lambda rate: inv_softplus(rate / self.bin_size)
            )
        self.link = link_functions[link]

        # Set the bias to be small if using log link
        if link == "log":
            self.ds = -3 + .5 * npr.randn(1, N) if single_subspace else npr.randn(K, N)

    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == int
        lambdas = self.mean(self.forward(x, input, tag))
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = -gammaln(data[:,None,:] + 1) -lambdas + data[:,None,:] * np.log(lambdas)
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(np.clip(data, .1, np.inf))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        lambdas = self.mean(self.forward(x, input, tag))
        y = npr.poisson(lambdas[np.arange(T), z, :])
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        lambdas = self.mean(self.forward(variational_mean, input, tag))
        return lambdas[:,0,:] if self.single_subspace else np.sum(lambdas * expected_states[:,:,None], axis=1)


class PoissonLDSObservations(_PoissonObservationsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def hessian_log_emissions_prob(self, data, input, mask, tag, x):
        """
        d/dx log p(y | x) = d/dx [y * (Cx + Fu + d) - exp(Cx + Fu + d)
                          = y * C - lmbda * C
                          = (y - lmbda) * C

        d/dx  (y - lmbda)^T C = d/dx -exp(Cx + Fu + d)^T C
            = -C^T exp(Cx + Fu + d)^T C
        """
        if self.link_name == "log":
            assert self.single_subspace
            lambdas = self.mean(self.forward(x, input, tag))
            return np.einsum('tn, ni, nj ->tij', -lambdas[:, 0, :], self.Cs[0], self.Cs[0])

        elif self.link_name == "softplus":
            assert self.single_subspace
            lambdas = self.mean(self.forward(x, input, tag))[:, 0, :] / self.bin_size
            expterms = np.exp(-np.dot(x,self.Cs[0].T)-np.dot(input,self.Fs[0].T)-self.ds[0])
            diags = (data / lambdas * (expterms - 1.0 / lambdas) - expterms * self.bin_size) / (1.0+expterms)**2
            return np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])


class PoissonOrthogonalObservations(_PoissonObservationsMixin, _OrthogonalLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

    def hessian_log_emissions_prob(self, data, input, mask, tag, x):
        """
        d/dx log p(y | x) = d/dx [y * (Cx + Fu + d) - exp(Cx + Fu + d)
                          = y * C - lmbda * C
                          = (y - lmbda) * C

        d/dx  (y - lmbda)^T C = d/dx -exp(Cx + Fu + d)^T C
            = -C^T exp(Cx + Fu + d)^T C
        """
        if self.link_name == "log":
            assert self.single_subspace
            lambdas = self.mean(self.forward(x, input, tag))
            return np.einsum('tn, ni, nj ->tij', -lambdas[:, 0, :], self.Cs[0], self.Cs[0])

        elif self.link_name == "softplus":
            assert self.single_subspace
            lambdas = self.mean(self.forward(x, input, tag))[:, 0, :] / self.bin_size
            expterms = np.exp(-np.dot(x,self.Cs[0].T)-np.dot(input,self.Fs[0].T)-self.ds[0])
            diags = (data / lambdas * (expterms - 1.0 / lambdas) - expterms * self.bin_size) / (1.0+expterms)**2
            return np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])

class PoissonIdentityObservations(_PoissonObservationsMixin, _IdentityEmissions):
    pass


class PoissonNeuralNetworkObservations(_PoissonObservationsMixin, _NeuralNetworkEmissions):
    pass
