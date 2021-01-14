from autograd import grad
import autograd.numpy as anp  # autograd likes to work with its own numpy wrapper.
from scipy.optimize import fsolve


sample_size = 1000

# Generate Labels
npseeded = anp.random.RandomState(42)
labels = npseeded.randint(2, size=sample_size)  # shape: [sample_size]

# Generate Predictions
npseeded = anp.random.RandomState(42)
p1 = npseeded.dirichlet([5., 2.], size=[sample_size])  # shape: [sample_size, 2]
npseeded = anp.random.RandomState(84)
p2 = npseeded.dirichlet([6., 2.], size=[sample_size])
npseeded = anp.random.RandomState(86)
p3 = npseeded.dirichlet([3., 2.], size=[sample_size])

# Stack Predictions
ps = anp.stack([p1, p2, p3])  # shape: [3, sample_size, 2]


# Define the metric
def binarycrossentropy(y_true, y_pred, clip=1e-15):
    y_true = anp.array(y_true)
    y_pred = anp.array(y_pred)
    y_pred = anp.clip(y_pred, clip, 1 - clip)
    loss = anp.where(y_true == 1, -anp.log(y_pred[:, 1]), -anp.log(1 - y_pred[:, 1]))
    return loss.mean()


def individual_cvs(y_true, y_pred):
    for i, y_p in enumerate(y_pred):
        print('M%d metric: %.7f' % (i, binarycrossentropy(y_true, y_p)))


# Print the individual scores
individual_cvs(labels, ps)


def blend_oofs(y_pred, weights):
    return anp.tensordot(y_pred, weights, axes=(0, 0))


def Lagrange_func(params):
    ws = anp.array(params[:-1])
    _lambda = params[-1]
    oof_blend = blend_oofs(ps, ws)
    return binarycrossentropy(labels, oof_blend) - _lambda * (ws.sum() - 1.)


def partial_derivatives(params):
    grad_L = grad(Lagrange_func)
    pars = grad_L(params)
    dLdws = pars[:-1]
    res = anp.append(dLdws, anp.sum(params[:-1]) - 1.)
    return res


npseeded = anp.random.RandomState(22)
weights = npseeded.dirichlet([2] * len(ps), size=1)[0].tolist() + [1.]
print('Initial parameters (w1, w2, w3, lambda):', weights)
pars = fsolve(partial_derivatives, weights)

oof_b = blend_oofs(ps, weights[:-1])
print('Blend metric:', binarycrossentropy(labels, oof_b))
print('Optimized Weights (w1, w2, w3, lambda):', pars)

oof_b = blend_oofs(ps, [1 / 3., 1 / 3., 1 / 3.])
print('Blend metric:', binarycrossentropy(labels, oof_b))
