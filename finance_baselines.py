import numpy as np
from utils import finance_utils
from functools import partial
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt


def polynomial(args, X, Y, gY, X_prime, poly=3):
    """
    :param args:
    :param X: Nx*D
    :param Y: Nx*Ny
    :param gY: Nx*Ny
    :param X_prime: N_test*D
    :param poly: int
    :return:
    """
    eps = 0.1
    X_standardized, X_mean, X_std = finance_utils.standardize(X)
    X_prime_standardized = (X_prime - X_mean) / X_std

    powers = jnp.arange(0, poly + 1)
    X_poly = X_standardized ** powers
    theta = jnp.linalg.inv(X_poly.T @ X_poly + eps * jnp.eye(poly + 1)) @ X_poly.T @ gY.mean(1)
    X_prime_poly = X_prime_standardized ** powers
    phi = X_prime_poly @ theta
    std = jnp.zeros_like(phi)

    # Debug
    # true_EgY_X = jnp.load(f"{args.save_path}/finance_EgY_X.npy")
    # plt.figure()
    # plt.ylim(-2, 15)
    # plt.plot(X_prime.squeeze(), true_EgY_X, color='red', label='true')
    # plt.plot(X_prime.squeeze(), phi.squeeze(), color='blue', label='LSMC')
    # plt.scatter(X.squeeze(), gY.mean(1).squeeze(), color='black', label='samples')
    # plt.legend()
    # plt.show()
    # pause = True
    return phi, std


def importance_sampling_single(tree, py_x_fn, x_prime):
    x, Yi, gYi = tree
    Yi_standardized, Yi_mean, Yi_scale = finance_utils.standardize(Yi)
    py_x_standardized_fn = partial(py_x_fn, sigma=0.3, T=2, t=1, y_scale=Yi_scale, y_mean=Yi_mean)
    py_x_prime = py_x_standardized_fn(Yi_standardized, x_prime)
    py_x_i = py_x_standardized_fn(Yi_standardized, x)
    weight = py_x_prime / py_x_i
    return (weight * gYi).mean() / weight.mean()


def importance_sampling_(py_x_fn, X, Y, gY, x_prime):
    x_prime = x_prime[:, None]
    importance_sampling_single_fn = partial(importance_sampling_single, py_x_fn=py_x_fn, x_prime=x_prime)
    importance_sampling_single_vmap = jax.vmap(importance_sampling_single_fn, in_axes=((0, 0, 0),))
    tree = (X, Y, gY)
    dummy = importance_sampling_single_vmap(tree)
    return dummy.mean()


def importance_sampling(py_x_fn, X_prime, X, Y, gY):
    """
    Vectorized importance sampling
    :param py_x_fn:
    :param X_prime: N_test*D
    :param X: Nx*D
    :param Y: Nx*Ny*D
    :param gY: Nx*Ny
    :return:
    """
    importance_sampling_fn = partial(importance_sampling_, py_x_fn, X, Y, gY)
    importance_sampling_vmap = jax.vmap(importance_sampling_fn)
    IS_mean = importance_sampling_vmap(X_prime)
    return IS_mean, 0 * IS_mean


def importance_sampling_old(py_x_fn, X_prime, X, Y, gY):
    Nx, Ny = Y.shape
    IS_list = []
    for j in range(len(X_prime)):
        x_prime = X_prime[j][:, None]
        dummy_list = []
        for i in range(Nx):
            x = X[i]
            Yi = Y[i, :][:, None]
            Yi_standardized, Yi_mean, Yi_scale = finance_utils.standardize(Yi)
            gYi = gY[i, :][:, None]

            py_x_standardized_fn = partial(py_x_fn, sigma=0.3, T=2, t=1, y_scale=Yi_scale, y_mean=Yi_mean)
            py_x_prime = py_x_standardized_fn(Yi_standardized, x_prime)
            py_x_i = py_x_standardized_fn(Yi_standardized, x)
            weight = py_x_prime / py_x_i
            dummy_list.append((weight * gYi).mean() / weight.mean())
        IS_list.append(np.array(dummy_list).mean())
        pause = True
    return jnp.array(IS_list), jnp.array(IS_list) * 0
