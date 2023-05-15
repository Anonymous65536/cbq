import numpy as np
import matplotlib.pyplot as plt
from jax.scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy import integrate
import time
import jax
import jax.numpy as jnp
import optax
from functools import partial
from tqdm import tqdm
import finance_baselines
from kernels import *
from utils import finance_utils
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


def get_config():
    parser = argparse.ArgumentParser(description='Conditional Bayesian Quadrature for finance data')

    # Data settings
    parser.add_argument('--kernel_x', type=str)
    parser.add_argument('--kernel_y', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./data')
    args = parser.parse_args()
    return args


@jax.jit
def grad_y_log_py_x(y, x, y_mean, y_scale, sigma, T, t):
    # dx log p(x) for log normal distribution with mu=-\sigma^2 / 2 * (T - t) and sigma = \sigma^2 (T - y)
    y = y * y_scale + y_mean
    part1 = (jnp.log(y) + sigma ** 2 * (T - t) / 2 - jnp.log(x)) / y / (sigma ** 2 * (T - t))
    return (-1. / y - part1) * y_scale


@jax.jit
def py_x_fn(y, x, y_scale, y_mean, sigma, T, t):
    """
    :param y: Ny * 1
    :param x: scalar
    :param y_scale: scalar
    :return: scalar
    """
    # p(x) for log normal distribution with mu=-\sigma^2 / 2 * (T - t) and sigma = \sigma^2 (T - t)
    y_tilde = y * y_scale + y_mean
    z = jnp.log(y_tilde / x)
    n = (z + sigma ** 2 * (T - t) / 2) / sigma / jnp.sqrt(T - t)
    p_n = jax.scipy.stats.norm.pdf(n)
    p_z = p_n / (sigma * jnp.sqrt(T - t))
    p_y_tilde = p_z / y_tilde
    p_y = p_y_tilde / y_scale
    return p_y


# @jax.jit
def log_py_x_fn(y, x, y_scale, sigma, T, t):
    # log p(x) for log normal distribution with mu=-\sigma^2 / 2 * (T - t) and sigma = \sigma^2 (T - t)
    y_tilde = y * y_scale
    z = jnp.log(y_tilde / x)
    n = (z + sigma ** 2 * (T - t) / 2) / sigma / jnp.sqrt(T - t)
    p_n = jax.scipy.stats.norm.pdf(n)
    p_z = p_n / (sigma * jnp.sqrt(T - t))
    p_y_tilde = p_z / y_tilde
    p_y = p_y_tilde / y_scale
    return jnp.log(p_y).sum()


@jax.jit
def stein_Matern(x, y, l, d_log_px, d_log_py):
    """
    :param x: N*D
    :param y: M*D
    :param l: scalar
    :param d_log_px: N*D
    :param d_log_py: M*D
    :return: N*M
    """
    K = my_Matern(x, y, l)
    dx_K = dx_Matern(x, y, l)
    dy_K = dy_Matern(x, y, l)
    dxdy_K = dxdy_Matern(x, y, l)
    part1 = d_log_px @ d_log_py.T * K
    part2 = (d_log_py[None, :] * dx_K).sum(-1)
    part3 = (d_log_px[:, None, :] * dy_K).sum(-1)
    part4 = dxdy_K
    return part1 + part2 + part3 + part4


@jax.jit
def log_normal_RBF(x, y, l, d_log_px, d_log_py):
    return my_RBF(jnp.log(x), jnp.log(y), l)


@jax.jit
def phi_log_normal_RBF(y, l, a, b):
    part1 = jnp.exp(-(a ** 2 + jnp.log(y) ** 2) / (2 * (b ** 2 + l ** 2)))
    part2 = jnp.power(y, a / (b ** 2 + l ** 2))
    part3 = b * jnp.sqrt(b ** (-2) + l ** (-2))
    return part1 * part2 / part3


@jax.jit
def varphi_log_normal_RBF(l, a, b):
    dummy = b ** 2 * jnp.sqrt(b ** (-2) + l ** (-2)) * jnp.sqrt(b ** (-2) + 1. / (b ** 2 + l ** 2))
    return 1. / dummy


@jax.jit
def stein_Laplace(x, y, l, d_log_px, d_log_py):
    """
    :param x: N*D
    :param y: M*D
    :param l: scalar
    :param d_log_px: N*D
    :param d_log_py: M*D
    :return: N*M
    """
    K = my_Laplace(x, y, l)
    dx_K = dx_Laplace(x, y, l)
    dy_K = dy_Laplace(x, y, l)
    dxdy_K = dxdy_Laplace(x, y, l)
    part1 = d_log_px @ d_log_py.T * K
    part2 = d_log_py.T * dx_K
    part3 = d_log_px * dy_K
    part4 = dxdy_K
    return part1 + part2 + part3 + part4


@partial(jax.jit, static_argnames=['Ky'])
def nllk_func(l, c, A, y, gy, d_log_py, Ky, eps):
    n = y.shape[0]
    K = A * Ky(y, y, l, d_log_py, d_log_py) + c + A * jnp.eye(n)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
    nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / n
    return nll[0][0]


@partial(jax.jit, static_argnames=['optimizer', 'Ky'])
def step(l, c, A, opt_state, optimizer, y, gy, d_log_py, Ky, eps):
    nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1, 2))(l, c, A, y, gy, d_log_py, Ky, eps)
    updates, opt_state = optimizer.update(grads, opt_state, (l, c, A))
    l, c, A = optax.apply_updates((l, c, A), updates)
    return l, c, A, opt_state, nllk_value


def train(x, y, y_scale, gy, d_log_py, dy_log_py_fn, rng_key, Ky):
    """
    :param y:
    :param gy:
    :param d_log_py:
    :param dy_log_py_fn:
    :param rng_key:
    :return:
    """
    rng_key, _ = jax.random.split(rng_key)
    n = y.shape[0]
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    eps = 1e-6

    c_init = c = 1.0
    l_init = l = 2.0
    A_init = A = 1.0 / jnp.sqrt(n)
    opt_state = optimizer.init((l_init, c_init, A_init))

    # @jax.jit
    # def nllk_func(l, c, A):
    #     # l = jnp.exp(log_l)
    #     n = y.shape[0]
    #     K = A * Ky(y, y, l, d_log_py, d_log_py) + c + A * jnp.eye(n)
    #     K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
    #     nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / n
    #     return nll[0][0]
    #
    # @jax.jit
    # def step(l, c, A, opt_state, rng_key):
    #     nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1, 2))(l, c, A)
    #     updates, opt_state = optimizer.update(grads, opt_state, (l, c, A))
    #     l, c, A = optax.apply_updates((l, c, A), updates)
    #     return l, c, A, opt_state, nllk_value

    # # Debug code
    # l_debug_list = []
    # c_debug_list = []
    # A_debug_list = []
    # nll_debug_list = []
    for _ in range(10):
        rng_key, _ = jax.random.split(rng_key)
        l, c, A, opt_state, nllk_value = step(l, c, A, opt_state, optimizer, y, gy, d_log_py, Ky, eps)
        # # Debug code
        # if jnp.isnan(nllk_value):
        #     # l = jnp.exp(log_l)
        #     K = A * Ky(y, y, l, d_log_py, d_log_py) + c + jnp.eye(n)
        #     K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
        #     pause = True
        # l_debug_list.append(l)
        # c_debug_list.append(c)
        # A_debug_list.append(A)
        # nll_debug_list.append(nllk_value)
    # Debug code
    # fig = plt.figure(figsize=(15, 6))
    # ax_1, ax_2, ax_3, ax_4 = fig.subplots(1, 4)
    # ax_1.plot(l_debug_list)
    # ax_2.plot(c_debug_list)
    # ax_3.plot(A_debug_list)
    # ax_4.plot(nll_debug_list)
    # plt.show()

    # l = jnp.exp(log_l)
    # A = jnp.exp(log_A)
    # y_debug = jnp.linspace(20, 160, 100)[:, None] / y_scale
    # d_log_py_debug = dy_log_py_fn(y_debug, x)
    # K_train_train = stein_Laplace(y, y, l, d_log_py, d_log_py) + c
    # K_train_train_inv = jnp.linalg.inv(K_train_train + eps * jnp.eye(n))
    # K_test_train = stein_Laplace(y_debug, y, l, d_log_py_debug, d_log_py) + c
    # gy_debug = K_test_train @ K_train_train_inv @ gy
    # plt.figure()
    # plt.scatter(y * y_scale, gy)
    # plt.plot(y_debug * y_scale, gy_debug)
    # plt.show()
    pause = True
    return l, c, A


class CBQ:
    def __init__(self, kernel_x, kernel_y):
        if kernel_y == 'stein_matern':
            self.Ky = stein_Matern
        elif kernel_y == 'stein_laplace':
            self.Ky = stein_Laplace
        elif kernel_y == 'stein_rbf':
            self.Ky = stein_Gaussian
        elif kernel_y == 'log_normal_RBF':
            self.Ky = log_normal_RBF
            self.ly = 0.1
        else:
            raise NotImplementedError

        if 'stein' not in kernel_y:
            self.cbq = self.cbq_no_stein
        elif 'stein' in kernel_y:
            self.cbq = self.cbq_stein
        else:
            pass

        if kernel_x == 'rbf':  # This is the best kernel for x
            self.Kx = my_RBF
            self.one_d_Kx = my_RBF
            self.lx = 1.5
        elif kernel_x == 'matern':
            self.Kx = my_Matern
            self.one_d_Kx = my_Matern
            self.lx = 0.7
        else:
            raise NotImplementedError
        return

    # @partial(jax.jit, static_argnums=(0,))
    def cbq_no_stein(self, X, Y, gY, rng_key):
        sigma = 0.3
        T = 2
        t = 1

        Nx = X.shape[0]
        Ny = Y.shape[1]
        eps = 1e-6
        Sigma = jnp.zeros(Nx)
        Mu = jnp.zeros(Nx)
        for i in range(Nx):
            x = X[i]
            Yi = Y[i, :][:, None]
            # Yi_standardized, Yi_scale = finance_utils.scale(Yi)
            Yi_standardized = Yi
            gYi = gY[i, :][:, None]
            # phi = \int ky(Y, y)p(y|x)dy, varphi = \int \int ky(y', y)p(y'|x)p(y|x)dydy'

            K = self.Ky(Yi_standardized, Yi_standardized, self.ly, None, None) + eps * jnp.eye(Ny)
            K_inv = jnp.linalg.inv(K)
            a = -sigma ** 2 * (T - t) / 2 + jnp.log(x)
            b = jnp.sqrt(sigma ** 2 * (T - t))
            phi = phi_log_normal_RBF(Yi_standardized, self.ly, a, b)
            varphi = varphi_log_normal_RBF(self.ly, a, b)
            mu_standardized = phi.T @ K_inv @ gYi
            std_standardized = jnp.sqrt(varphi - phi.T @ K_inv @ phi)

            Sigma = Sigma.at[i].set(std_standardized.squeeze())
            Mu = Mu.at[i].set(mu_standardized.squeeze())

            # ============= Debug code =============
            # print('True value', price(X[i], 10000, rng_key)[1].mean())
            # print(f'MC with {Ny} number of Y', gYi.mean())
            # print(f'BMC with {Ny} number of Y', mu_standardized.squeeze())
            # print(f"=================")
            # pause = True
            # ============= Debug code =============
        return Mu, Sigma

    # @partial(jax.jit, static_argnums=(0,))
    def cbq_stein(self, X, Y, gY, rng_key):
        """
        :param X: X is of size Nx
        :param Y: Y is of size Nx * Ny
        :param gY: gY is g(Y)
        :return: return the expectation E[g(Y)|X=x] of size Nx
        """

        Nx = X.shape[0]
        Ny = Y.shape[1]
        eps = 1e-6
        Sigma = jnp.zeros(Nx)
        Mu = jnp.zeros(Nx)

        for i in range(Nx):
            x = X[i]
            Yi = Y[i, :][:, None]
            Yi_standardized, Yi_mean, Yi_scale = finance_utils.standardize(Yi)
            gYi = gY[i, :][:, None]

            grad_y_log_py_x_fn = partial(grad_y_log_py_x, sigma=0.3, T=2, t=1, y_mean=Yi_mean, y_scale=Yi_scale)
            dy_log_py_x = grad_y_log_py_x_fn(Yi_standardized, x)
            if i == 0:
                ly, c, A = train(x, Yi_standardized, Yi_scale, gYi,
                                 dy_log_py_x, grad_y_log_py_x_fn, rng_key, self.Ky)
            # phi = \int ky(Y, y)p(y|x)dy, varphi = \int \int ky(y', y)p(y|x)p(y|x)dydy'

            K = A * self.Ky(Yi_standardized, Yi_standardized, ly, dy_log_py_x, dy_log_py_x) + c + A * jnp.eye(Ny)
            K_inv = jnp.linalg.inv(K + eps * jnp.eye(Ny))
            mu = c * (K_inv @ gYi).sum()
            std = jnp.sqrt(c - K_inv.sum() * c ** 2)

            Sigma = Sigma.at[i].set(std.squeeze())
            Mu = Mu.at[i].set(mu.squeeze())

            # # Large sample mu
            # print('True value', price(X[i], 10000, rng_key)[1].mean())
            # print(f'MC with {Ny} number of Y', gYi.mean())
            # print(f'BMC with {Ny} number of Y', mu)
            # print(f"=================")
            pause = True
        return Mu, Sigma

    # @partial(jax.jit, static_argnums=(0,))
    def GP(self, psi_y_x_mean, psi_y_x_std, X, X_prime):
        """
        :param psi_y_x_mean: Nx * 1
        :param psi_y_x_std: Nx * 1
        :param X: Nx * 1
        :param X_prime: N_prime * 1
        :return:
        """
        Nx = psi_y_x_mean.shape[0]
        X_standardized, X_mean, X_std = finance_utils.standardize(X)
        X_prime_standardized = (X_prime - X_mean) / X_std

        if psi_y_x_std is None:
            noise_array = jnp.array([0.01, 0.001, 0.0001])
            nll_array = 0. * noise_array
            for i, noise in enumerate(noise_array):
                K_train_train = self.Kx(X_standardized, X_standardized, self.lx) + noise * jnp.eye(Nx)
                nll = -(-0.5 * psi_y_x_mean.T @ K_train_train @ psi_y_x_mean -
                        0.5 * jnp.log(jnp.linalg.det(K_train_train) + 1e-6)) / Nx
                nll_array = nll_array.at[i].set(nll.squeeze())
            noise = noise_array[jnp.argmin(nll_array)]
            K_train_train = self.Kx(X_standardized, X_standardized, self.lx) + noise * jnp.eye(Nx)
            K_train_train_inv = jnp.linalg.inv(K_train_train)
            K_test_train = self.Kx(X_prime_standardized, X_standardized, self.lx)
            K_test_test = self.Kx(X_prime_standardized, X_prime_standardized, self.lx) + noise
            mu_y_x_prime = K_test_train @ K_train_train_inv @ psi_y_x_mean
            var_y_x_prime = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
            std_y_x_prime = jnp.sqrt(var_y_x_prime)
        else:
            noise = psi_y_x_mean.mean()
            K_train_train = self.Kx(X_standardized, X_standardized, self.lx) + jnp.diag(psi_y_x_std ** 2)
            K_train_train_inv = jnp.linalg.inv(K_train_train)
            K_test_train = self.Kx(X_prime_standardized, X_standardized, self.lx)
            K_test_test = self.Kx(X_prime_standardized, X_prime_standardized, self.lx) + noise
            mu_y_x_prime = K_test_train @ K_train_train_inv @ psi_y_x_mean
            var_y_x_prime = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
            std_y_x_prime = jnp.sqrt(var_y_x_prime)
        pause = True
        return mu_y_x_prime, std_y_x_prime

    def save(self, Nx, Ny, psi_x_mean, St, St_prime,
             BMC_mean, BMC_std, KMS_mean, IS_mean, LSMC_mean,
             time_cbq, time_IS, time_KMS, time_LSMC, calibration):
        true_EgY_X = jnp.load(f"{args.save_path}/finance_EgY_X.npy")

        # ========== Debug code ==========
        # plt.figure()
        # plt.plot(St_prime.squeeze(), true_EgY_X, color='red', label='true')
        # plt.scatter(St.squeeze(), psi_x_mean.squeeze())
        # plt.plot(St_prime.squeeze(), mu_y_x_prime_cbq.squeeze(), color='blue', label='BMC')
        # plt.plot(St_prime.squeeze(), mu_y_x_prime_IS.squeeze(), color='green', label='IS')
        # plt.plot(St_prime.squeeze(), mu_y_x_prime_LSMC.squeeze(), color='orange', label='LSMC')
        # plt.plot(St_prime.squeeze(), KMS_mean, color='purple', label='KMS')
        # plt.legend()
        # plt.title(f"GP_finance_X_{Nx}_y_{Ny}")
        # plt.savefig(f"{args.save_path}/figures/finance_X_{Nx}_y_{Ny}.pdf")
        # plt.show()
        # plt.close()
        # ========== Debug code ==========

        L_BMC = jnp.maximum(BMC_mean, 0).mean()
        L_IS = jnp.maximum(IS_mean, 0).mean()
        L_LSMC = jnp.maximum(LSMC_mean, 0).mean()
        L_KMS = jnp.maximum(KMS_mean, 0).mean()
        L_true = jnp.maximum(true_EgY_X, 0).mean()

        rmse_dict = {}
        rmse_dict['BMC'] = (L_true - L_BMC) ** 2
        rmse_dict['IS'] = (L_true - L_IS) ** 2
        rmse_dict['LSMC'] = (L_true - L_LSMC) ** 2
        rmse_dict['KMS'] = (L_true - L_KMS) ** 2
        with open(f"{args.save_path}/rmse_dict_X_{Nx}_y_{Ny}", 'wb') as f:
            pickle.dump(rmse_dict, f)

        time_dict = {'BMC': time_cbq, 'IS': time_IS, 'LSMC': time_LSMC, 'KMS': time_KMS}
        with open(f"{args.save_path}/time_dict_X_{Nx}_y_{Ny}", 'wb') as f:
            pickle.dump(time_dict, f)

        jnp.save(f"{args.save_path}/calibration_X_{Nx}_y_{Ny}", calibration)

        # ============= Debug code =============
        # print(f"=============")
        # print(f"RMSE of BMC with {Nx} number of X and {Ny} number of Y", rmse_dict['BMC'])
        # print(f"RMSE of IS with {Nx} number of X and {Ny} number of Y", rmse_dict['IS'])
        # print(f"RMSE of LSMC with {Nx} number of X and {Ny} number of Y", rmse_dict['LSMC'])
        # print(f"RMSE of KMS with {Nx} number of X and {Ny} number of Y", rmse_dict['KMS'])
        # print(f"Time of BMC with {Nx} number of X and {Ny} number of Y", time_cbq)
        # print(f"Time of IS with {Nx} number of X and {Ny} number of Y", time_IS)
        # print(f"Time of LSMC with {Nx} number of X and {Ny} number of Y", time_LSMC)
        # print(f"Time of KMS with {Nx} number of X and {Ny} number of Y", time_KMS)
        # print(f"=============")
        # ============= Debug code =============
        pause = True
        return

    def save_large(self, Nx, Ny, KMS_mean, mu_y_x_prime_LSMC, IS_mean, time_KMS, time_LSMC, time_IS):
        true_EgY_X = jnp.load(f"{args.save_path}/finance_EgY_X.npy")

        # Saving this would explode the memory on cluster
        # jnp.save(f"{args.save_path}/LSMC_mean_X_{Nx}_y_{Ny}.npy", mu_y_x_prime_LSMC.squeeze())
        # jnp.save(f"{args.save_path}/KMS_mean_X_{Nx}_y_{Ny}.npy", KMS_mean)

        L_LSMC = jnp.maximum(mu_y_x_prime_LSMC, 0).mean()
        L_KMS = jnp.maximum(KMS_mean, 0).mean()
        L_IS = jnp.maximum(IS_mean, 0).mean()
        L_true = jnp.maximum(true_EgY_X, 0).mean()

        rmse_dict = {}
        rmse_dict['BMC'] = None
        rmse_dict['IS'] = (L_true - L_IS) ** 2
        rmse_dict['LSMC'] = (L_true - L_LSMC) ** 2
        rmse_dict['KMS'] = (L_true - L_KMS) ** 2
        with open(f"{args.save_path}/rmse_dict_X_{Nx}_y_{Ny}", 'wb') as f:
            pickle.dump(rmse_dict, f)

        time_dict = {'BMC': None, 'IS': time_IS, 'LSMC': time_LSMC, 'KMS': time_KMS}
        with open(f"{args.save_path}/time_dict_X_{Nx}_y_{Ny}", 'wb') as f:
            pickle.dump(time_dict, f)
        pause = True
        return


@partial(jax.jit, static_argnums=(1,))
def price(St, N, rng_key):
    """
    :param St: the price St at time t
    :return: The function returns the price ST at time T sampled from the conditional
    distribution p(ST|St), and the loss \psi(ST) - \psi((1+s)ST) due to the shock. Their shape is Nx * Ny
    """
    K1 = 50
    K2 = 150
    s = -0.2
    sigma = 0.3
    T = 2
    t = 1

    output_shape = (St.shape[0], N)
    rng_key, _ = jax.random.split(rng_key)
    epsilon = jax.random.normal(rng_key, shape=output_shape)
    ST = St * jnp.exp(sigma * jnp.sqrt((T - t)) * epsilon - 0.5 * (sigma ** 2) * (T - t))
    psi_ST_1 = jnp.maximum(ST - K1, 0) + jnp.maximum(ST - K2, 0) - 2 * jnp.maximum(ST - (K1 + K2) / 2, 0)
    psi_ST_2 = jnp.maximum((1 + s) * ST - K1, 0) + jnp.maximum((1 + s) * ST - K2, 0) - 2 * jnp.maximum(
        (1 + s) * ST - (K1 + K2) / 2, 0)
    return ST, psi_ST_1 - psi_ST_2


def save_true_value(St, args):
    seed = args.seed
    rng_key = jax.random.PRNGKey(seed)
    rng_key, _ = jax.random.split(rng_key)

    # K1 = 50
    # K2 = 150
    # s = -0.2
    # t = 1
    # T = 2
    # sigma = 0.3
    # S0 = 50

    _, loss = price(St, 100000, rng_key)
    value = loss.mean(1)
    jnp.save(f"{args.save_path}/finance_X.npy", St)
    jnp.save(f"{args.save_path}/finance_EgY_X.npy", value)
    plt.figure()
    plt.plot(St, value)
    plt.xlabel(r"$X$")
    plt.ylabel(r"$\mathbb{E}[g(Y) \mid X]$")
    plt.title("True value for finance experiment")
    plt.savefig(f"{args.save_path}/true_distribution.pdf")
    # plt.show()
    plt.close()
    return


def cbq_option_pricing(args):
    seed = args.seed
    # seed = 0
    rng_key = jax.random.PRNGKey(seed)
    rng_key, _ = jax.random.split(rng_key)

    K1 = 50
    K2 = 150
    s = -0.2
    t = 1
    T = 2
    sigma = 0.3
    S0 = 50
    Nx_array = jnp.array([10, 50, 100])
    # Nx_array = jnp.array([2, 5, 10, 20, 30])
    Ny_array = jnp.array([10, 50, 100])
    # Ny_array = jnp.concatenate((jnp.array([5]), jnp.arange(5, 105, 5)))

    test_num = 200
    St_prime = jnp.linspace(20., 120., test_num)[:, None]
    save_true_value(St_prime, args)

    kernel_x = args.kernel_x
    kernel_y = args.kernel_y
    CBQ_class = CBQ(kernel_x=kernel_x, kernel_y=kernel_y)

    for Nx in Nx_array:
        for Ny in tqdm(Ny_array):
            rng_key, _ = jax.random.split(rng_key)
            # epsilon = jax.random.normal(rng_key, shape=(Nx, 1))
            # St = S0 * jnp.exp(sigma * jnp.sqrt(t) * epsilon - 0.5 * (sigma ** 2) * t)
            St = jnp.linspace(20, 120, Nx)[:, None]
            ST, loss = price(St, Ny.item(), rng_key)

            mc_mean = loss.mean(1)[:, None]
            _, _ = CBQ_class.GP(mc_mean, None, St, St_prime)
            t0 = time.time()
            KMS_mean, KMS_std = CBQ_class.GP(mc_mean, None, St, St_prime)
            KMS_mean = KMS_mean.squeeze()
            time_KMS = time.time() - t0

            _, _ = finance_baselines.importance_sampling(py_x_fn, St_prime, St, ST, loss)
            t0 = time.time()
            IS_mean, IS_std = finance_baselines.importance_sampling(py_x_fn, St_prime, St, ST, loss)
            time_IS = time.time() - t0

            _, _ = finance_baselines.polynomial(args, St, ST, loss, St_prime)
            t0 = time.time()
            LSMC_mean, LSMC_std = finance_baselines.polynomial(args, St, ST, loss, St_prime)
            time_LSMC = time.time() - t0

            _, _ = CBQ_class.cbq(St, ST, loss, rng_key)
            t0 = time.time()
            # St is X, ST is Y, loss is g(Y)
            psi_x_mean, psi_x_std = CBQ_class.cbq(St, ST, loss, rng_key)
            t1 = time.time()
            psi_x_std = np.nan_to_num(psi_x_std, nan=0.3)
            _, _ = CBQ_class.GP(psi_x_mean, psi_x_std, St, St_prime)
            t2 = time.time()
            BMC_mean, BMC_std = CBQ_class.GP(psi_x_mean, psi_x_std, St, St_prime)
            t3 = time.time()
            time_cbq = t3 - t2 + t1 - t0

            ground_truth = jnp.load(f"{args.save_path}/finance_EgY_X.npy")
            calibration = finance_utils.calibrate(ground_truth, BMC_mean, jnp.diag(BMC_std))

            CBQ_class.save(Nx, Ny, psi_x_mean, St, St_prime,
                           BMC_mean, BMC_std, KMS_mean, IS_mean, LSMC_mean,
                           time_cbq, time_IS, time_KMS, time_LSMC, calibration)

    # For very very large Nx and Ny.
    Nx = 1000
    Ny = 1000
    epsilon = jax.random.normal(rng_key, shape=(Nx, 1))
    St = S0 * jnp.exp(sigma * jnp.sqrt(t) * epsilon - 0.5 * (sigma ** 2) * t)
    # St = jnp.linspace(20, 120, Nx)[:, None]
    ST, loss = price(St, Ny, rng_key)

    t0 = time.time()
    mc_mean = loss.mean(1)[:, None]
    KMS_mean_large, _ = CBQ_class.GP(mc_mean, None, St, St_prime)
    time_KMS_large = time.time() - t0

    t0 = time.time()
    LSMC_mean_large, _ = finance_baselines.polynomial(args, St, ST, loss, St_prime)
    time_LSMC_large = time.time() - t0

    t0 = time.time()
    IS_mean_large, IS_std = finance_baselines.importance_sampling(py_x_fn, St_prime, St, ST, loss)
    time_IS_large = time.time() - t0

    CBQ_class.save_large(Nx, Ny, KMS_mean_large, LSMC_mean_large, IS_mean_large,
                         time_KMS_large, time_LSMC_large, time_IS_large)
    pause = True
    return


def main(args):
    seed = args.seed
    # seed = 0
    rng_key = jax.random.PRNGKey(seed)

    visualize_brownian = False
    debug_BSM = False
    if visualize_brownian:
        n = 100.
        T = 1.
        dt = T / n
        plt.figure()
        for i in range(10):
            St = finance_utils.Geometric_Brownian(n, dt, rng_key)
            plt.plot(St)
        # plt.show()
    elif debug_BSM:
        finance_utils.BSM_butterfly_analytic()
    else:
        pass
    cbq_option_pricing(args)
    return


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    if 'stein' in args.kernel_y:
        args.save_path += f'results/finance_stein/'
    else:
        args.save_path += f'results/finance/'
    args.save_path += f"seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(f"{args.save_path}/figures/", exist_ok=True)
    return args


if __name__ == '__main__':
    args = get_config()
    args = create_dir(args)
    print(f'Device is {jax.devices()}')
    print(args.seed)
    main(args)
    save_path = args.save_path
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    if os.path.exists(f"{save_path}__complete"):
        shutil.rmtree(f"{save_path}__complete")
    os.rename(save_path, f"{save_path}__complete")
    print("\n------------------- DONE -------------------\n")

