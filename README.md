# EWOLM

EWOLM is a python package to optimize the weights of an ensemble of machine learning models with the Lagrange multiplier method.


#### Install


```python
pip install ewolm
```



#### Install Dependencies


```python
conda install numpy autograd scipy
```



#### Example


```python
from ewolm import ewolm

y_true = pd.read_csv('targets.csv')  # e.g. A pandas dataframe of shape [rows, columns]
y_pred = np.read_csv('oofs.npy') # A numpy array of shape [N_models, rows, columns]

ewolm(metric='bce')  # bce for Binary Crossentropy
ewolm(y_true, y_pred)
```



#### Metric


The `metric` keyword takes any differentiable numpy function. The following metrics are readily available in the package:

`bce`: Binary crossentropy

`mlbce`: Multilabel binary crossentropy

`rmse`: Root mean square error





