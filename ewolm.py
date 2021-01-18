import numpy as np
from time import time
from autograd import grad
import autograd.numpy as anp
from scipy.optimize import fsolve


class BinaryCrossentropy:
    def __init__(self, clip=1e-15):
        self.clip = clip

    def __call__(self, y_true, y_pred):
        y_true = anp.array(y_true)
        y_pred = anp.array(y_pred)
        y_pred = anp.clip(y_pred, self.clip, 1 - self.clip)  # This is for numerical stability.
        loss = anp.where(y_true == 1, -anp.log(y_pred[:, 1]), -anp.log(1 - y_pred[:, 1]))
        return loss.mean()


class MLBinaryCrossentropy:
    def __init__(self, clip=1e-15):
        self.clip = clip

    def __call__(self, y_true, y_pred):
        y_true = anp.array(y_true).ravel()
        y_pred = anp.array(y_pred).ravel()
        y_pred = anp.clip(y_pred, self.clip, 1 - self.clip)
        loss = anp.where(y_true == 1, -anp.log(y_pred[:, 1]), -anp.log(1 - y_pred[:, 1]))
        return loss.mean()


class RMSE:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        y_true = anp.array(y_true).ravel()
        y_pred = anp.array(y_pred).ravel()
        return anp.sqrt(anp.mean((y_true - y_pred)**2.))


class ewolm:
    def __init__(self, metric='mlbce'):  # working fine
        if metric == 'mlbce':
            self.metric = MLBinaryCrossentropy()
        elif metric == 'bce':
            self.metric = BinaryCrossentropy()
        elif metric == 'rmse':
            self.metric = RMSE()
        else:
            metric = metric
        self.weights = 0.

    def individual_cvs(self):  # working fine
        for i, p in enumerate(self.y_pred):
            print('M%d metric: %.7f' % (i, self.metric(self.y_true, p)))

    def blend_oofs(self, y_pred, weights):
        return anp.tensordot(y_pred, weights, axes=(0, 0))

    def Lagrange_func(self, params):
        """
        Calculate the Lagrangian (L) with constraints
        L = Metric(y_true, sum(y_pi.wi)) - lambda (sum(wi) - 1)
        """
        ws = anp.array(params[:-1])
        _lambda = params[-1]
        oof_blend = self.blend_oofs(self.y_pred, ws)
        return self.metric(self.y_true, oof_blend) - _lambda * (1. - np.sum(np.abs(ws)))

    def partial_derivatives(self, params):
        """
        Calculate the partial derivatives of the Lagrangian
        dLdws and dLdlambda
        """
        grad_L = grad(self.Lagrange_func)
        pars = grad_L(params)  # Lagrange_func requires the full parameter set
        dLdws = pars[:-1]
        res = anp.append(dLdws, sum(params[:-1]) - 1.)  # The last element of params is the lagrange multiplier.
        return res

    def __call__(self, y_true, y_pred):
        t1 = time()
        self.y_true = y_true
        self.y_pred = y_pred

        self.individual_cvs()

        self.weights = anp.random.dirichlet([2] * len(y_pred), size=1)[0].tolist() + [1]
        print('Initial Weights:', self.weights[:-1])  # Format self.weights

        pars = fsolve(self.partial_derivatives, self.weights)

        self.weights = np.float32(pars[:-1])
        self._lambda = np.float32(pars[-1])

        if np.any([self.weights < 0]):
            print('Caution: There are negative weights in the solution!')

        print('Optimum Weights:', self.weights.tolist())  # Format self.weights
        oof_b = self.blend_oofs(self.y_pred, self.weights)
        self.optimized_cv = self.metric(y_true, oof_b)
        self.blended_pred = np.tensordot(y_pred, self.weights, axes=(0, 0))
        print('Blend metric:', self.optimized_cv)
        t2 = time()
        print('Finished in %.2f seconds' % (t2 - t1))
