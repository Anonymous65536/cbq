import jax.numpy as jnp
from functools import partial
import jax


def polynomial(X, Y, gY, X_prime, poly=3):
    """
    Polynomial Regression
    :param poly: int
    :param X_prime: N_test*D
    :param X: Nx*D
    :param Y: Nx*Ny*D
    :param gY: Nx*Ny
    :return:
    """
    powers = jnp.arange(0, poly + 1)
    X_poly = X[:, :, None] ** powers
    X_poly = X_poly.reshape([X.shape[0], -1])

    eps = 1.0
    D = (1 + poly) * X.shape[1]
    theta = jnp.linalg.inv(X_poly.T @ X_poly + eps * jnp.eye(D)) @ X_poly.T @ gY.mean(1)

    X_prime_poly = X_prime[:, :, None] ** powers
    X_prime_poly = X_prime_poly.reshape([X_prime.shape[0], -1])
    phi = X_prime_poly @ theta
    std = 0
    pause = True
    return phi, std


# def importance_sampling_(log_py_x_fn, X, Y, gY, x_prime):
#     # xi = X[i, :]
#     # Yi = Y[i, :, :]
#     # gYi = gY[i, :]
#     tree = (X, Y, gY)
#     importance_sampling_single_fn = jax.jit(partial(importance_sampling_single, log_py_x_fn=log_py_x_fn, x_prime=x_prime))
#     importance_sampling_single_vmap = jax.jit(jax.vmap(importance_sampling_single_fn, in_axes=((0, 0, 0), )))
#     temp_array = importance_sampling_single_vmap(tree)
#     return temp_array.mean()
#
#
# @partial(jax.jit, static_argnums=(1,))
# def importance_sampling_single(tree, log_py_x_fn, x_prime):
#     """
#     :param log_py_x_fn:
#     :param tree: consists of xi: (D, ), Yi: (Ny, D), gYi: (Ny, )
#     :param x_prime:
#     :return:
#     """
#     xi, Yi, gYi = tree
#     log_py_x_prime = log_py_x_fn(theta=Yi, alpha=x_prime)
#     log_py_x_i = log_py_x_fn(theta=Yi, alpha=xi)
#     weight = jnp.exp(log_py_x_prime - log_py_x_i)
#     mu = (weight * gYi).mean() / (weight.mean() + 0.01)
#     return mu
#
#
# def importance_sampling(log_py_x_fn, X, Y, gY, X_prime):
#     """
#     Vectorized importance sampling
#     :param log_py_x_fn:
#     :param X_prime: N_test*D
#     :param X: Nx*D
#     :param Y: Nx*Ny*D
#     :param gY: Nx*Ny
#     :return:
#     """
#     importance_sampling_fn = jax.jit(partial(importance_sampling_, log_py_x_fn, X, Y, gY))
#     importance_sampling_vmap = jax.jit(jax.vmap(importance_sampling_fn))
#     IS_mean = importance_sampling_vmap(X_prime)
#     return IS_mean, 0 * IS_mean


# @partial(jax.jit, static_argnums=(0,))
def importance_sampling(log_py_x_fn, X, Y, gY, X_prime):
    """
    :param log_py_x_fn:
    :param X_prime: N_test*D
    :param X: Nx*D
    :param Y: Nx*Ny*D
    :param gY: Nx*Ny
    :return:
    """
    # log_py_x_prime is (Nx, Ny, N_test)
    log_py_x_prime = log_py_x_fn(theta=Y, alpha=X_prime)
    # log_py_x_i is (Nx, Ny, Nx)
    log_py_x_i = log_py_x_fn(theta=Y, alpha=X)

    # log_py_x_prime is (Nx, Ny, N_test)
    log_py_x_prime = log_py_x_prime.transpose(2, 0, 1)
    # log_py_x_i is (Nx, Nx, Ny)
    log_py_x_i = log_py_x_i.transpose(2, 0, 1)

    # weight is (N_test, Nx, Ny)
    weight = jnp.exp(log_py_x_prime - jnp.diagonal(log_py_x_i, axis1=0, axis2=1).transpose(1, 0))
    # mu is (N_test, Nx)
    if X.shape[1] == 2:
        mu = (weight * gY).mean(2)
    else:
        mu = (weight * gY).mean(2) / (weight.mean(2) + 0.00)
    IS_mean = mu.mean(1)
    return IS_mean, 0
