import numpy as np
import matplotlib.pyplot as plt
import time
import jax
import jax.numpy as jnp
import optax
from functools import partial
from tqdm import tqdm
import decision_baselines
from kernels import *
from utils import decision_utils
import os
import pwd
import shutil
import argparse
import pickle
from jax.config import config

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)


plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()


def f1(x, y):
    """
    :param x: (T, 1)
    :param y: (T, N, 9)
    :return: (T, N)
    """
    lamba_ = 1e4
    # lamba_ = 1.0
    # lambda * (X_5 * X_6 * X_7 + X_8 * X_9 * X_10) - (X_1 + X_2 * X_3 * X_4)
    # X_5 is x
    # X_1 is y[0], X_2 is y[1], X_3 is y[2], X_4 is y[3], X_6 is y[4], X_7 is y[5], X_8 is y[6],
    # X_9 is y[7], X_10 is y[8],
    return lamba_ * (x * y[:, :, 4] * y[:, :, 5] + y[:, :, 6] * y[:, :, 7] * y[:, :, 8]) - \
           (y[:, :, 0] + y[:, :, 1] * y[:, :, 2] * y[:, :, 3])


def f2(x, y):
    """
    :param x: (T, 1)
    :param y: (T, N, 9)
    :return: (N, T)
    """
    lamba_ = 1e4
    # lambda * (X_5 * X_6 * X_7 + X_8 * X_9 * X_10) - (X_1 + X_2 * X_3 * X_4)
    # X_14 is x
    # X_4 is y[0], X_11 is y[1], X_12 is y[2], X_13 is y[3], X_15 is y[4], X_16 is y[5],
    # X_17 is y[6], X_18 is y[7], X_19 is y[8]
    return lamba_ * (x * y[:, :, 4] * y[:, :, 5] + y[:, :, 6] * y[:, :, 7] * y[:, :, 8]) - \
           (y[:, :, 1] + y[:, :, 2] * y[:, :, 3] * y[:, :, 0])


def conditional_distribution(joint_mean, joint_covariance, x, dimensions_y, dimensions_x):
    """
    :param joint_mean: (19,)
    :param joint_covariance: (19, 19)
    :param x: (N, len(dimensions_x))
    :param dimensions_y: list
    :param dimensions_x: list
    :return: (N, len(dimensions_y))
    """

    dimensions_y = jnp.array(dimensions_y)
    dimensions_x = jnp.array(dimensions_x)

    mean_x = jnp.take(joint_mean, dimensions_x)[:, None]
    mean_y = jnp.take(joint_mean, dimensions_y)[:, None]

    # Create a grid of indices from A and B using meshgrid
    cov_XX = joint_covariance[jnp.ix_(dimensions_x, dimensions_x)]
    cov_YY = joint_covariance[jnp.ix_(dimensions_y, dimensions_y)]
    cov_YX = joint_covariance[jnp.ix_(dimensions_y, dimensions_x)]
    cov_XY = joint_covariance[jnp.ix_(dimensions_x, dimensions_y)]

    x_t = x.T
    mean_y_given_x = mean_y + cov_YX @ jnp.linalg.inv(cov_XX) @ (x_t - mean_x)
    cov_y_given_x = cov_YY - cov_YX @ jnp.linalg.inv(cov_XX) @ cov_XY
    return mean_y_given_x.T, cov_y_given_x


def Bayesian_Monte_Carlo_Matern(rng_key, u, y, gy, mu_y_x, sigma_y_x):
    """
    :param u: (N, D)
    :param sigma_y_x: (D, D)
    :param mu_y_x: (D, )
    :param rng_key:
    :param y: (N, D)
    :param gy: (N, )
    :return:
    """
    N, D = y.shape[0], y.shape[1]
    eps = 1e-6

    K_no_scale = jnp.zeros([N, N])
    phi_no_scale = jnp.zeros([N, 1])

    l_array = jnp.array([1.0] * 3 + [1.0] + [1.0] * 5)
    for i in range(D):
        l = l_array[i]
        u_i = u[:, i][:, None]
        K_no_scale += my_Matern(u_i, u_i, l)
        phi_no_scale += kme_Matern_Gaussian(l, u_i)

    A = gy.T @ K_no_scale @ gy / N
    K = A * K_no_scale
    phi = A * phi_no_scale

    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    varphi = phi.mean()

    BMC_mean = phi.T @ K_inv @ gy
    BMC_std = jnp.sqrt(jnp.abs(varphi - phi.T @ K_inv @ phi))

    BMC_mean = BMC_mean.squeeze()
    BMC_std = BMC_std.squeeze()
    pause = True
    return BMC_mean, BMC_std


def GP(psi_y_x_mean, psi_y_x_std, X, X_prime, eps):
    """
    :param psi_y_x_mean: (n_alpha, )
    :param psi_y_x_std: (n_alpha, )
    :param X: (n_alpha, D)
    :param X_prime: (N_test, D)
    :return:
    """
    Nx, D = X.shape[0], X.shape[1]
    l_array = jnp.array([0.3, 1.0, 2.0, 3.0])

    if psi_y_x_std is None:
        sigma_array = jnp.array([1.0, 0.1, 0.01, 0.001])
        A_array = jnp.array([10.0, 100.0, 300.0, 1000.0])
        nll_array = jnp.zeros([len(l_array), len(A_array), len(sigma_array)])
    else:
        sigma_array = jnp.array([0.0])
        A_array = 0 * l_array
        nll_array = jnp.zeros([len(l_array), 1])

    if psi_y_x_std is None:
        for i, l in enumerate(l_array):
            for j, A in enumerate(A_array):
                for k, sigma in enumerate(sigma_array):
                    K = A * my_Matern(X, X, l) + jnp.eye(Nx) * sigma
                    K_inv = jnp.linalg.inv(K)
                    nll = -(-0.5 * psi_y_x_mean.T @ K_inv @ psi_y_x_mean - 0.5 * jnp.log(
                        jnp.linalg.det(K) + 1e-6)) / Nx
                    nll_array = nll_array.at[i, j].set(nll.squeeze())
        min_index_flat = jnp.argmin(nll_array)
        i1, i2, i3 = jnp.unravel_index(min_index_flat, nll_array.shape)
        l = l_array[i1]
        A = A_array[i2]
        sigma = sigma_array[i3]
    else:
        for i, l in enumerate(l_array):
            K_no_scale = my_Matern(X, X, l)
            A = psi_y_x_mean.T @ K_no_scale @ psi_y_x_mean / Nx
            A_array = A_array.at[i].set(A.squeeze())
            K = A * my_Matern(X, X, l) + eps * jnp.eye(Nx) + jnp.diag(psi_y_x_std ** 2)
            K_inv = jnp.linalg.inv(K)
            nll = -(-0.5 * psi_y_x_mean.T @ K_inv @ psi_y_x_mean - 0.5 * jnp.log(
                jnp.linalg.det(K) + 1e-6)) / Nx
            nll_array = nll_array.at[i].set(nll.squeeze())

        l = l_array[jnp.argmin(nll_array)]
        A = A_array[jnp.argmin(nll_array)]

    if psi_y_x_std is None:
        K_train_train = A * my_Matern(X, X, l) + jnp.eye(Nx) * sigma
        K_train_train_inv = jnp.linalg.inv(K_train_train)
        K_test_train = A * my_Matern(X_prime, X, l)
        K_test_test = A * my_Matern(X_prime, X_prime, l) + jnp.eye(X_prime.shape[0]) * sigma
    else:
        # print(A)
        # A = 10.0
        K_train_train = A * my_Matern(X, X, l) + eps * jnp.eye(Nx) + jnp.diag(psi_y_x_std ** 2)
        K_train_train_inv = jnp.linalg.inv(K_train_train)
        K_test_train = A * my_Matern(X_prime, X, l)
        K_test_test = A * my_Matern(X_prime, X_prime, l) + eps * jnp.eye(X_prime.shape[0])
    mu_y_x = K_test_train @ K_train_train_inv @ psi_y_x_mean
    var_y_x = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    var_y_x = jnp.abs(var_y_x)
    std_y_x = jnp.sqrt(var_y_x)
    pause = True
    return mu_y_x, std_y_x


def main(args):
    seed = args.seed
    rng_key = jax.random.PRNGKey(seed)
    XY_mean = jnp.array([1000., 0.1, 5.2, 400., 0.7,
                         0.3, 3.0, 0.25, -0.1, 0.5,
                         1500, 0.08, 6.1, 0.8, 0.3,
                         3.0, 0.2, -0.1, 0.5])
    XY_sigma = jnp.array([1.0, 0.02, 1.0, 200, 0.1,
                          0.1, 0.5, 0.1, 0.02, 0.2,
                          1.0, 0.02, 1.0, 0.1, 0.05,
                          1.0, 0.05, 0.02, 0.2])
    XY_sigma = jnp.diag(XY_sigma ** 2)
    XY_sigma = XY_sigma.at[4, 6].set(0.6 * jnp.sqrt(XY_sigma[4, 4]) * jnp.sqrt(XY_sigma[6, 6]))
    XY_sigma = XY_sigma.at[6, 4].set(0.6 * jnp.sqrt(XY_sigma[4, 4]) * jnp.sqrt(XY_sigma[6, 6]))
    XY_sigma = XY_sigma.at[4, 13].set(0.6 * jnp.sqrt(XY_sigma[4, 4]) * jnp.sqrt(XY_sigma[13, 13]))
    XY_sigma = XY_sigma.at[13, 4].set(0.6 * jnp.sqrt(XY_sigma[4, 4]) * jnp.sqrt(XY_sigma[13, 13]))
    XY_sigma = XY_sigma.at[4, 15].set(0.6 * jnp.sqrt(XY_sigma[4, 4]) * jnp.sqrt(XY_sigma[15, 15]))
    XY_sigma = XY_sigma.at[15, 4].set(0.6 * jnp.sqrt(XY_sigma[4, 4]) * jnp.sqrt(XY_sigma[15, 15]))
    XY_sigma = XY_sigma.at[6, 13].set(0.6 * jnp.sqrt(XY_sigma[6, 6]) * jnp.sqrt(XY_sigma[13, 13]))
    XY_sigma = XY_sigma.at[13, 6].set(0.6 * jnp.sqrt(XY_sigma[6, 6]) * jnp.sqrt(XY_sigma[13, 13]))
    XY_sigma = XY_sigma.at[6, 15].set(0.6 * jnp.sqrt(XY_sigma[6, 6]) * jnp.sqrt(XY_sigma[15, 15]))
    XY_sigma = XY_sigma.at[15, 6].set(0.6 * jnp.sqrt(XY_sigma[6, 6]) * jnp.sqrt(XY_sigma[15, 15]))
    XY_sigma = XY_sigma.at[13, 15].set(0.6 * jnp.sqrt(XY_sigma[13, 13]) * jnp.sqrt(XY_sigma[15, 15]))
    XY_sigma = XY_sigma.at[15, 13].set(0.6 * jnp.sqrt(XY_sigma[13, 13]) * jnp.sqrt(XY_sigma[15, 15]))

    # X is index (4, 13), Y is index (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18)
    X_mean = jnp.array([0.7, 0.8])
    X_sigma = jnp.array([[0.01, 0.01 * 0.6], [0.01 * 0.6, 0.01]])
    Y_mean = jnp.array([1000., 0.1, 5.2, 400.,
                        0.3, 3.0, 0.25, -0.1, 0.5,
                        1500, 0.08, 6.1, 0.3,
                        3.0, 0.2, -0.1, 0.5])
    Y_sigma = jnp.array([1.0, 0.02, 1.0, 200,
                         0.1, 0.5, 0.1, 0.02, 0.2,
                         1.0, 0.02, 1.0, 0.05,
                         1.0, 0.05, 0.02, 0.2])
    f1_cond_dist_fn = partial(conditional_distribution, joint_mean=XY_mean, joint_covariance=XY_sigma,
                              dimensions_x=[4], dimensions_y=[0, 1, 2, 3, 5, 6, 7, 8, 9])
    f2_cond_dist_fn = partial(conditional_distribution, joint_mean=XY_mean, joint_covariance=XY_sigma,
                              dimensions_x=[13], dimensions_y=[3, 10, 11, 12, 14, 15, 16, 17, 18])
    # ============= Debug code =============
    # dummy_y_x_mean, dummy_y_x_sigma = cond_dist_fn(x=X_mean[None, :])
    # pause = True
    # ============= Debug code =============

    # ============= Code to generate test points Begins =============
    N_test = 100
    large_sample_size = 10000
    rng_key, _ = jax.random.split(rng_key)
    X_test = jax.random.multivariate_normal(rng_key, X_mean, X_sigma, shape=(N_test,))
    X1_test = X_test[:, 0][:, None]
    X2_test = X_test[:, 1][:, None]
    f1_p_Y_X_mean_test, f1_p_Y_X_sigma_test = f1_cond_dist_fn(x=X1_test)
    f2_p_Y_X_mean_test, f2_p_Y_X_sigma_test = f2_cond_dist_fn(x=X2_test)

    Y1_test = jnp.zeros([N_test, large_sample_size, 9]) + 0.0
    u1_test = jnp.zeros([N_test, large_sample_size, 9]) + 0.0
    Y2_test = jnp.zeros([N_test, large_sample_size, 9]) + 0.0
    u2_test = jnp.zeros([N_test, large_sample_size, 9]) + 0.0
    for i in tqdm(range(N_test)):
        rng_key, _ = jax.random.split(rng_key)
        u1_temp = jax.random.multivariate_normal(rng_key,
                                                 mean=jnp.zeros_like(f1_p_Y_X_mean_test[i, :]),
                                                 cov=jnp.eye(f1_p_Y_X_sigma_test.shape[0]),
                                                 shape=(large_sample_size,))
        L1 = jnp.linalg.cholesky(f1_p_Y_X_sigma_test)
        Y1_temp = f1_p_Y_X_mean_test[i, :] + jnp.matmul(L1, u1_temp.T).T
        # Y1_temp = jax.random.multivariate_normal(rng_key, f1_p_Y_X_mean[i, :], f1_p_Y_X_sigma, shape=(Ny,))
        Y1_test = Y1_test.at[i, :, :].set(Y1_temp)
        u1_test = u1_test.at[i, :, :].set(u1_temp)

        rng_key, _ = jax.random.split(rng_key)
        u2_temp = jax.random.multivariate_normal(rng_key,
                                                 mean=jnp.zeros_like(f2_p_Y_X_mean_test[i, :]),
                                                 cov=jnp.eye(f2_p_Y_X_sigma_test.shape[0]),
                                                 shape=(large_sample_size,))
        L2 = jnp.linalg.cholesky(f2_p_Y_X_sigma_test)
        Y2_temp = f2_p_Y_X_mean_test[i, :] + jnp.matmul(L2, u2_temp.T).T
        Y2_test = Y2_test.at[i, :, :].set(Y2_temp)
        u2_test = u2_test.at[i, :, :].set(u2_temp)
    f1_Y_test = f1(X1_test, Y1_test)
    f2_Y_test = f2(X2_test, Y2_test)
    ground_truth_1 = f1_Y_test.mean(1)
    ground_truth_2 = f2_Y_test.mean(1)
    # ============= Code to generate test points Ends =============

    Nx_array = jnp.array([10, 20, 30])
    # Nx_array = jnp.concatenate((jnp.array([3, 5]), jnp.arange(10, 150, 10)))
    #
    # Ny_array = jnp.array([10, 30])
    # Ny_array = jnp.array([10, 30, 50, 70, 100])
    Ny_array = jnp.arange(20, 200, 10)

    for Nx in Nx_array:
        rng_key, _ = jax.random.split(rng_key)
        # X shape is (Nx, 2)
        # X = jax.random.multivariate_normal(rng_key, X_mean, X_sigma, shape=(Nx,))
        # X1 = X[:, 0][:, None]
        # X2 = X[:, 1][:, None]

        X1 = jnp.linspace(X1_test.min(), X1_test.max(), Nx)[:, None]
        X2 = jnp.linspace(X2_test.min(), X2_test.max(), Nx)[:, None]

        for Ny in tqdm(Ny_array):
            f1_p_Y_X_mean, f1_p_Y_X_sigma = f1_cond_dist_fn(x=X1)
            f2_p_Y_X_mean, f2_p_Y_X_sigma = f2_cond_dist_fn(x=X2)
            # Y1, Y2 shape is (Nx, Ny, 9)
            Y1 = jnp.zeros([Nx, Ny, 9]) + 0.0
            u1 = jnp.zeros([Nx, Ny, 9]) + 0.0
            Y2 = jnp.zeros([Nx, Ny, 9]) + 0.0
            u2 = jnp.zeros([Nx, Ny, 9]) + 0.0
            for i in range(Nx):
                rng_key, _ = jax.random.split(rng_key)
                u1_temp = jax.random.multivariate_normal(rng_key,
                                                         mean=jnp.zeros_like(f1_p_Y_X_mean[i, :]),
                                                         cov=jnp.eye(f1_p_Y_X_sigma.shape[0]),
                                                         shape=(Ny,))
                L1 = jnp.linalg.cholesky(f1_p_Y_X_sigma)
                Y1_temp = f1_p_Y_X_mean[i, :] + jnp.matmul(L1, u1_temp.T).T
                # Y1_temp = jax.random.multivariate_normal(rng_key, f1_p_Y_X_mean[i, :], f1_p_Y_X_sigma, shape=(Ny,))
                Y1 = Y1.at[i, :, :].set(Y1_temp)
                u1 = u1.at[i, :, :].set(u1_temp)

                rng_key, _ = jax.random.split(rng_key)
                u2_temp = jax.random.multivariate_normal(rng_key,
                                                         mean=jnp.zeros_like(f2_p_Y_X_mean[i, :]),
                                                         cov=jnp.eye(f2_p_Y_X_sigma.shape[0]),
                                                         shape=(Ny,))
                L2 = jnp.linalg.cholesky(f2_p_Y_X_sigma)
                Y2_temp = f2_p_Y_X_mean[i, :] + jnp.matmul(L2, u2_temp.T).T
                Y2 = Y2.at[i, :, :].set(Y2_temp)
                u2 = u2.at[i, :, :].set(u2_temp)

            # f_Y shape is (Nx, Ny)
            f1_Y = f1(X1, Y1)
            f2_Y = f2(X2, Y2)

            f1_psi_mean_array = jnp.zeros([Nx]) + 0.0
            f1_psi_std_array = jnp.zeros([Nx]) + 0.0
            f1_mc_mean_array = jnp.zeros([Nx]) + 0.0
            f2_psi_mean_array = jnp.zeros([Nx]) + 0.0
            f2_psi_std_array = jnp.zeros([Nx]) + 0.0
            f2_mc_mean_array = jnp.zeros([Nx]) + 0.0

            # ============= Code for f1 Starts =============
            for i in range(Nx):
                rng_key, _ = jax.random.split(rng_key)
                Y1_i = Y1[i, :, :]
                f1_Y_i = f1_Y[i, :]
                u1_i = u1[i, :, :]
                # scale = f1_Y_i.mean()
                scale = 1000.0
                f1_Y_i_standardized = f1_Y_i / scale
                f1_psi_mean, f1_psi_std = Bayesian_Monte_Carlo_Matern(rng_key, u1_i, Y1_i, f1_Y_i_standardized,
                                                                      f1_p_Y_X_mean[i, :], f1_p_Y_X_sigma)
                f1_psi_mean = f1_psi_mean * scale

                f1_mc_mean = f1_Y_i.mean()

                f1_psi_mean_array = f1_psi_mean_array.at[i].set(f1_psi_mean)
                f1_psi_std_array = f1_psi_std_array.at[i].set(f1_psi_std)
                f1_mc_mean_array = f1_mc_mean_array.at[i].set(f1_mc_mean)

                # # ============= Debug code =============
                # rng_key, _ = jax.random.split(rng_key)
                # Y_temp_large = jax.random.multivariate_normal(rng_key, f1_p_Y_X_mean[i, :], f1_p_Y_X_sigma,
                #                                               shape=(large_sample_size,))
                # f1_Y_large = f1(X1[i, :][None, :], Y_temp_large[None, :])
                # f1_mc_mean_large = f1_Y_large.mean()
                # print("=============")
                # print('Large sample MC', f1_mc_mean_large)
                # print(f'MC with {Ny} number of Y', f1_mc_mean)
                # print(f'BMC with {Ny} number of Y', f1_psi_mean)
                # print(f'BMC uncertainty {f1_psi_std}')
                # print(f"=============")
                # pause = True
                # ============= Debug code =============

            LSMC_mean_1, LSMC_std_1 = decision_baselines.polynomial(X1, Y1, f1_Y, X1_test)
            BMC_mean_1, BMC_std_1 = GP(f1_psi_mean_array, f1_psi_std_array, X1, X1_test, eps=f1_psi_std_array.mean())
            KMS_mean_1, KMS_std_1 = GP(f1_mc_mean_array, None, X1, X1_test, eps=0.0)

            # ============= Code for f2 Starts =============
            for i in range(Nx):
                rng_key, _ = jax.random.split(rng_key)
                Y2_i = Y2[i, :, :]
                f2_Y_i = f2_Y[i, :]
                u2_i = u2[i, :, :]
                # scale = f2_Y_i.mean()
                scale = 1000.0
                f2_Y_i_standardized = f2_Y_i / scale
                f2_psi_mean, f2_psi_std = Bayesian_Monte_Carlo_Matern(rng_key, u2_i, Y2_i, f2_Y_i_standardized,
                                                                      f2_p_Y_X_mean[i, :], f2_p_Y_X_sigma)
                f2_psi_mean = f2_psi_mean * scale

                f2_mc_mean = f2_Y_i.mean()

                f2_psi_mean_array = f2_psi_mean_array.at[i].set(f2_psi_mean)
                f2_psi_std_array = f2_psi_std_array.at[i].set(f2_psi_std)
                f2_mc_mean_array = f2_mc_mean_array.at[i].set(f2_mc_mean)

                # # ============= Debug code =============
                # rng_key, _ = jax.random.split(rng_key)
                # Y_temp_large = jax.random.multivariate_normal(rng_key, f2_p_Y_X_mean[i, :], f2_p_Y_X_sigma,
                #                                               shape=(large_sample_size,))
                # f2_Y_large = f2(X2[i, :][None, :], Y_temp_large[None, :])
                # f2_mc_mean_large = f2_Y_large.mean()
                # print("=============")
                # print('Large sample MC', f2_mc_mean_large)
                # print(f'MC with {Ny} number of Y', f2_mc_mean)
                # print(f'BMC with {Ny} number of Y', f2_psi_mean)
                # print(f'BMC uncertainty {f2_psi_std}')
                # print(f"=============")
                # pause = True
                # ============= Debug code =============

            LSMC_mean_2, LSMC_std_2 = decision_baselines.polynomial(X2, Y2, f2_Y, X2_test)
            BMC_mean_2, BMC_std_2 = GP(f2_psi_mean_array, f2_psi_std_array, X2, X2_test, eps=f2_psi_std_array.mean())
            KMS_mean_2, KMS_std_2 = GP(f2_mc_mean_array, None, X2, X2_test, eps=0.0)
            
            # ============= Code for f2 Ends =============
            
            true_value = jnp.maximum(ground_truth_1, ground_truth_2)
            BMC_value = jnp.maximum(BMC_mean_1, BMC_mean_2)
            KMS_value = jnp.maximum(KMS_mean_1, KMS_mean_2)
            LSMC_value = jnp.maximum(LSMC_mean_1, LSMC_mean_2)

            rmse_BMC = jnp.sqrt(jnp.mean((BMC_value - true_value) ** 2))
            rmse_KMS = jnp.sqrt(jnp.mean((KMS_value - true_value) ** 2))
            rmse_LSMC = jnp.sqrt(jnp.mean((LSMC_value - true_value) ** 2))

            # true_value = jnp.maximum(ground_truth_1, ground_truth_2).mean()
            # BMC_value = jnp.maximum(BMC_mean_1, BMC_mean_2).mean()
            # KMS_value = jnp.maximum(KMS_mean_1, KMS_mean_2).mean()
            # LSMC_value = jnp.maximum(LSMC_mean_1, LSMC_mean_2).mean()
            #
            # rmse_BMC = jnp.sqrt(jnp.mean((BMC_value - true_value) ** 2))
            # rmse_KMS = jnp.sqrt(jnp.mean((KMS_value - true_value) ** 2))
            # rmse_LSMC = jnp.sqrt(jnp.mean((LSMC_value - true_value) ** 2))

            decision_utils.save(args, Nx, Ny, rmse_BMC, rmse_KMS, rmse_LSMC)

            # ============= Debug code =============
            # plt.figure()
            # X1_test_ = X1_test.squeeze()
            # ind = X1_test_.argsort()
            # plt.plot(X1_test_[ind], ground_truth_1[ind], label='Ground truth')
            # plt.plot(X1_test_[ind], BMC_mean_1[ind], label='BMC')
            # plt.plot(X1_test_[ind], KMS_mean_1[ind], label='KMS')
            # plt.scatter(X1.squeeze(), f1_psi_mean_array.squeeze())
            # plt.plot(X1_test_[ind], LSMC_mean_1[ind], label='LSMC')
            # plt.legend()
            # plt.show()
            #
            # plt.figure()
            # X2_test_ = X2_test.squeeze()
            # ind = X2_test_.argsort()
            # plt.plot(X2_test_[ind], ground_truth_2[ind], label='Ground truth')
            # plt.plot(X2_test_[ind], BMC_mean_2[ind], label='BMC')
            # plt.plot(X2_test_[ind], KMS_mean_2[ind], label='KMS')
            # plt.scatter(X2.squeeze(), f2_psi_mean_array.squeeze())
            # plt.plot(X2_test_[ind], LSMC_mean_2[ind], label='LSMC')
            # plt.legend()
            # plt.show()
            #
            # print(f"=============")
            # print(f"RMSE of BMC with {Nx} number of X and {Ny} number of Y", rmse_BMC)
            # print(f"RMSE of KMS with {Nx} number of X and {Ny} number of Y", rmse_KMS)
            # print(f"RMSE of LSMC with {Nx} number of X and {Ny} number of Y", rmse_LSMC)
            # print(f"=============")
            # pause = True
            # ============= Debug code =============



def get_config():
    parser = argparse.ArgumentParser(description='Conditional Bayesian Quadrature for Bayesian sensitivity analysis')
    # Args settings
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./data')
    args = parser.parse_args()
    return args


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/decision/'
    args.save_path += f"seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    return args


if __name__ == '__main__':
    args = get_config()
    create_dir(args)
    print(f'Device is {jax.devices()}')
    print(f'Seed is {args.seed}')
    main(args)
    save_path = args.save_path
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    if os.path.exists(f"{save_path}__complete"):
        print(f"Removing old results at {save_path}__complete")
        shutil.rmtree(f"{save_path}__complete")
    os.rename(save_path, f"{save_path}__complete")
    print("\n------------------- DONE -------------------\n")
