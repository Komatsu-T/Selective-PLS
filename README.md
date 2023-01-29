# selectivePLS

## Overview
selectivePLS is a Python package for partial least squares (PLS) regression models. This package provides functions as follows:
* Build general PLS regression models like sklearn
* Calculate Variable Importance in Projection (VIP) values
* Visualize projection vectors used to transform inputted features
* **Build PLS regression models that transform only user-selected features to latent features**

![overview](https://user-images.githubusercontent.com/79096203/215299395-4725242b-1e22-4385-a43a-d60a5ae9783c.png)

## Installation
```python
pip install git+https://github.com/Komatsu-T/selective-PLS.git
```

## Dependency
```python
numpy
pandas
matplotlib
scikit-learn
```
## Usage
* See [Sample01.ipynb](https://github.com/Komatsu-T/selective-PLS/blob/main/Sample01.ipynb) for general PLS regression models
* See [Sample02.ipynb](https://github.com/Komatsu-T/selective-PLS/blob/main/Sample02.ipynb) for selective PLS regression models

## Algorithm
### PLS Model
$\boldsymbol{X} = \boldsymbol{TP} + \boldsymbol{E}$  
$\boldsymbol{T} = \boldsymbol{XW}$  
$\boldsymbol{y} = \boldsymbol{Tq} + \boldsymbol{e}$  

X: design matrix  
y: target vector  
T: matrix of latent features  
W: projection matrix used to transform X  
P: loadings of X  
q: loadings of y  
E: residuals of X  
e: residuals of y  
W: projection matrix used to transform X







