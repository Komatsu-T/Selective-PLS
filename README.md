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

$\boldsymbol{X}$: design matrix  
$\boldsymbol{y}$: target vector  
$\boldsymbol{T}$: matrix of latent features  
$\boldsymbol{W}$: projection matrix used to transform X  
$\boldsymbol{P}$: loadings of X  
$\boldsymbol{q}$: loadings of y  
$\boldsymbol{E}$: residuals of X  
$\boldsymbol{e}$: residuals of y  

### General PLS Algorithm
1. $\boldsymbol{X_0} = \boldsymbol{X}$: $(n \times d)$    
2. $\boldsymbol{y_0} = \boldsymbol{y}$: $(n \times 1)$    
3. $\boldsymbol{t_0} = \boldsymbol{0}$: $(n \times 1)$  
4. $\boldsymbol{p_0} = \boldsymbol{0}$: $(d \times 1)$  
5. $q_0 = 0$: scalar  
6. **for** $i=1$ to $r$ **do**  
7. $\hspace{20pt}\boldsymbol{X_i}\leftarrow \boldsymbol{X_{i-1}} - (\boldsymbol{t_{i-1}}\boldsymbol{p_{i-1}}^{T})$  
8. $\hspace{20pt}\boldsymbol{y_i}\leftarrow \boldsymbol{y_{i-1}} - (q_{i-1}\boldsymbol{t_{i-1}})$  
9. $\hspace{20pt}\boldsymbol{w_i}\leftarrow \mathop{\rm argmax}\limits_{\boldsymbol{w_i}} \boldsymbol{y_i}^{T}\boldsymbol{X_i}\boldsymbol{w_i}-k(||\boldsymbol{w_i}||^2-1)$  
10. $\hspace{20pt}\boldsymbol{t_i}\leftarrow \boldsymbol{X_i}\boldsymbol{w_i}$
11. $\hspace{20pt}\boldsymbol{p_i}\leftarrow \mathop{\rm argmin}\limits_{\boldsymbol{p_i}} ||\boldsymbol{X_i}-\boldsymbol{t_i}\boldsymbol{p_i}^{T}||^2$
12. $\hspace{20pt}q_i\leftarrow \mathop{\rm argmin}\limits_{q_i} ||\boldsymbol{y_i}-\boldsymbol{t_i}q_i||^2$
13. $\boldsymbol{W} = (\boldsymbol{w_1},\ldots,\boldsymbol{w_r})$
14. $\boldsymbol{T} = (\boldsymbol{t_1},\ldots,\boldsymbol{t_r})$
15. $\boldsymbol{P} = (\boldsymbol{p_1},\ldots,\boldsymbol{p_r})$
16. $\boldsymbol{q} = (q_1,\ldots,q_r)$






