import numpy as np

def f(a, b, x):
    return a * np.sin(b * x) + a * x * np.sin(b * x**2)

def log_likelihood(params, args):
    a, b = params
    y_obs, x = args
    y_pred = f(a, b, x)

    # Assuming Gaussian likelihood with a known standard deviation
    sigma = 1
    n = len(y_obs)
    ll = -0.5 * n * np.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2)) * np.sum((y_obs - y_pred)**2)
    return ll