# EWOLM

EWOLM is a python package to optimize the weights of an ensemble of machine learning models with the Lagrange multiplier method.



In machine learning, blending predictions from a number of models yields a better predictive performance than could be obtained from any of the constituent models alone.

EWOLM finds the optimum weights for a linear blending of models, optimizing the following Lagrangian:


```math
L(y_t, y_p, w) = f\left(y_t, \sum_{i=1}^{N} y_{pi}w_i \right) - \lambda \left(1-\sum_{i=0}^N w_i\right)
```



where $f$ is a differentiable metric, $y_t$ is the true label set, $y_{pi}$ is the predicted probability set for the $i^{th}$ model, and $w_i$ is the weight for the $i^{th}$ model. The term multiplied by the Lagrange multiplier ($\lambda$) dictates that the sum of weights have to be 1.



#### Install

---

```python
pip install ewolm
```



#### Install Dependencies

---

```python
conda install numpy autograd scipy
```



#### Example

---

```python
from ewolm import ewolm

y_true = pd.read_csv('targets.csv')  # e.g. A pandas dataframe of shape [rows, columns]
y_pred = np.read_csv('oofs.npy') # A numpy array of shape [N_models, rows, columns]

ewolm(metric='bce')  # bce for Binary Crossentropy
ewolm(y_true, y_pred)
```



#### Metric

----

The `metric` keyword takes any differentiable numpy function. The following metrics are readily available in the package:

`bce`: Binary crossentropy

`mlbce`: Multilabel binary crossentropy

`rmse`: Root mean square error





