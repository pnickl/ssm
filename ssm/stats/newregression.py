import autograd.numpy as np
from autograd.scipy.special import logsumexp

import ssm.stats.dist as dist
from ssm.util.optimizers import adam, bfgs, lbfgs, rmsprop, sgd


class Regression(object):
    """
    Base class for regression distribution objects.
    """
    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def mean(self, covariates):
        raise NotImplementedError

    def log_probability(self, covariates, data):
        """
        Return the log probability of the data under the given parameters
        """
        raise NotImplementedError

    def sample(self, covariates, num_samples=1):
        raise NotImplementedError

    def m_step(self, covariates, datas, weights=None, expected_stats=None,
               optimizer="lbfgs", num_iters=100, **kwargs):
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


class LinearRegression(Regression):
    """
    Standard linear regression with multivariate normal errors.

        y ~ N(Ax + b, Sigma)

    """
    def __init__(self, A, b, Sigma):
        self.A = A
        self.b = b
        self.Sigma = Sigma

    @property
    def params(self):
        return self.A, self.b, self._Sigma_chol

    @params.setter
    def params(self, value):
        self.A, self.b, self._Sigma_chol = value

    @property
    def Sigma(self):
        return np.matmul(self._Sigma_chol, np.swapaxes(self._Sigma_chol, -1, -2))

    @Sigma.setter
    def Sigma(self, value):
        self._Sigma_chol = np.linalg.cholesky(value)

    def mean(self, covariates):
        return np.dot(covariates, self.A.T) + self.b

    def log_probability(self, covariates, data):
        return dist.MultivariateNormal(self.mean(covariates), self.Sigma).log_probability(data)

    def m_step(self, covariates, datas, weights=None, expected_stats=None):
        if weights is None:
            weights = [np.ones(data.shape[0]) for data in datas]

        outdim, indim = self.A.shape
        if expected_stats is None:
            Ex = np.zeros(indim)
            Ey = np.zeros(outdim)
            ExxT = np.zeros((indim, indim))
            ExyT = np.zeros((indim, outdim))
            EyyT = np.zeros((outdim, outdim))
            w_tot = 1e-16

            for w, x, y in zip(weights, covariates, datas):
                # Compute sufficient statistics using augmented covariates
                Ex += np.einsum('n,ni->i', w, x)
                Ey += np.einsum('n,ni->i', w, y)
                ExxT += np.einsum('n,ni,nj->ij', w, x, x)
                ExyT += np.einsum('n,ni,nj->ij', w, x, y)
                EyyT += np.einsum('n,ni,nj->ij', w, y, y)
                w_tot += np.sum(w)

            Ex /= w_tot
            Ey /= w_tot
            ExxT /= w_tot
            ExyT /= w_tot
            EyyT /= w_tot

        else:
            Ex, Ey, ExxT, ExyT, EyyT = expected_stats

        # Package sufficient statistics into full matrices for covariates (x, 1)
        full_ExxT = np.zeros((indim + 1, indim + 1))
        full_ExxT[:indim, :indim] = ExxT
        full_ExxT[:indim, -1] = Ex
        full_ExxT[-1, :indim] = Ex
        full_ExxT[-1, -1] = 1

        full_ExyT = np.zeros((indim + 1, outdim))
        full_ExyT[:indim] = ExyT
        full_ExyT[-1] = Ey

        # Solve the linear regression
        full_A = np.linalg.solve(full_ExxT, full_ExyT).T
        self.A = full_A[:, :-1]
        self.b = full_A[:, -1]
        self.Sigma = EyyT - 2 * np.dot(full_ExyT.T, full_A) + np.dot(full_A, np.dot(full_ExxT, full_A.T))


# TODO:
# Linear regression with Diagonal noise
# Linear regression with students T noise
# Autoregression features

class _GeneralizedLinearModelBase(Regression):
    """
    Base class for generalized linear models.
    """
    def __init__(self, A, b):
        self.A = A
        self.b = b

    @property
    def params(self):
        return self.A, self.b

    @params.setter
    def params(self, value):
        self.A, self.b = value

    @property
    def mean_function(self):
        return self._mean_function

    def mean(self, covariates):
        return self.mean_function(np.dot(covariates, self.A.T) + self.b)



class BernoulliGLM(_GeneralizedLinearModelBase):
    """
    y ~ Po(sigma(Ax + b))

    where sigma is the logistic sigmoid function.
    """
    def __init__(self, A, b, mean_function="logistic"):
        super(PoissonGLM, self).__init__(A, b)

        # Set the mean function
        mean_functions = dict(
            logistic=lambda v: 1 / (1 + np.exp(-v))
            )
        self._mean_function = mean_functions[mean_function]

    def log_probability(self, covariates, data):
        """
        Return the log probability of the data under the given parameters
        """
        return dist.Bernoulli(self.mean(covariates)).log_probability(data)


class PoissonGLM(_GeneralizedLinearModelBase):
    """
    y ~ Po(f(Ax + b))

    where f is a mean function mapping R -> R_+
    """
    def __init__(self, A, b, mean_function="softplus"):
        super(PoissonGLM, self).__init__(A, b)

        # Set the mean function
        mean_functions = dict(
            exp=lambda v: np.exp(v),
            softplus=lambda v: np.log(1 + np.exp(v)),
            relu=lambda v: np.maximum(0, v)
            )
        self._mean_function = mean_functions[mean_function]

    def log_probability(self, covariates, data):
        """
        Return the log probability of the data under the given parameters
        """
        return dist.Poisson(self.mean(covariates)).log_probability(data)


class CategoricalGLM(_GeneralizedLinearModelBase):
    """
    y ~ Cat(f(Ax + b))

    where f is a multiclass logistic function
    """
    def __init__(self, A, b, mean_function="logistic"):
        super(CategoricalGLM, self).__init__(A, b)

        # Set the mean function
        mean_functions = dict(
            logistic=lambda v: np.exp(v - logsumexp(v, axis=-1))
            )
        self._mean_function = mean_functions[mean_function]

    def log_probability(self, covariates, data):
        """
        Return the log probability of the data under the given parameters
        """
        return dist.Categorical(self.mean(covariates)).log_probability(data)


# Alias logistic regression
LogisticRegression = BernoulliGLM
MulticlassLogisticRegression = CategoricalGLM
