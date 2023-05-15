import matplotlib.pyplot as plt
from tqdm import tqdm
import jax.numpy as jnp
import jax
import numpy as np
import optax
import time
from kernels import *
from functools import partial
import os
import pwd
import pickle

plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["axes.titlesize"] = 28
plt.rcParams["font.size"] = 28
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["lines.markersize"] = 7
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 1.0
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["legend.facecolor"] = "white"
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8
plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['text.usetex'] = False
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath, amssymb}']


@jax.jit
def g(y):
    return jnp.exp(jnp.sin(10 * y) ** 2 - 4 * y ** 2)


### Important! Be caution about the gradient here, the derivative of the absolute value is taken to be 0 at 0.
# @jax.jit
def stein_Matern(x, y, l, mean, std):
    d_log_px = -1. / (std ** 2) * (x - mean)
    d_log_py = -1. / (std ** 2) * (y - mean)

    K = my_Matern(x, y, l)
    dx_K = dx_Matern(x, y, l)
    dy_K = dy_Matern(x, y, l)
    dxdy_K = dxdy_Matern(x, y, l)
    part1 = d_log_px @ d_log_py.T * K
    part2 = d_log_py.T * dx_K
    part3 = d_log_px * dy_K
    part4 = dxdy_K
    return part1 + part2 + part3 + part4


def CBQ(x, y, gy, mean, std, rng_key):
    """
    :param x: x is a scalar
    :param y: y is sampled from conditional distribution p(y|x)
    :param gy: g(y)
    :param mean: The mean of p(y|x)
    :param std: The std of p(y|x)
    :return: BMC mean and BMC std at x
    """
    n = y.shape[0]
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    eps = 1e-6

    c_init = c = 1.0
    log_l_init = log_l = jnp.log(0.3)
    A_init = A = 1.0 / jnp.sqrt(n)
    opt_state = optimizer.init((log_l_init, c_init, A_init))

    @jax.jit
    def nllk_func(log_l, c, A, mean, std):
        l = jnp.exp(log_l)
        n = y.shape[0]
        K = A * stein_Matern(y, y, l, mean, std) + c
        K_inv = jnp.linalg.inv(K)
        nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / n
        return nll[0][0]

    @jax.jit
    def step(log_l, c, A, opt_state, mean, std, rng_key):
        nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1, 2))(log_l, c, A, mean, std)
        updates, opt_state = optimizer.update(grads, opt_state, (log_l, c, A))
        log_l, c, A = optax.apply_updates((log_l, c, A), updates)
        return log_l, c, A, opt_state, nllk_value

    # # Debug code
    # log_l_debug_list = []
    # c_debug_list = []
    # A_debug_list = []
    # nll_debug_list = []
    for _ in range(2000):
        rng_key, _ = jax.random.split(rng_key)
        log_l, c, A, opt_state, nllk_value = step(log_l, c, A, opt_state, mean, std, rng_key)
    #     # Debug code
    #     log_l_debug_list.append(log_l)
    #     c_debug_list.append(c)
    #     A_debug_list.append(A)
    #     nll_debug_list.append(nllk_value)
    # # Debug code
    # fig = plt.figure(figsize=(15, 6))
    # ax_1, ax_2, ax_3, ax_4 = fig.subplots(1, 4)
    # ax_1.plot(log_l_debug_list)
    # ax_2.plot(c_debug_list)
    # ax_3.plot(A_debug_list)
    # ax_4.plot(nll_debug_list)
    # plt.show()

    l = jnp.exp(log_l)
    final_K = A * stein_Matern(y, y, l, mean, std) + c
    final_K_inv = jnp.linalg.inv(final_K)
    BMC_mean = c * (final_K_inv @ gy).sum()
    BMC_std = jnp.sqrt(c - final_K_inv.sum() * c * c)
    if jnp.isnan(BMC_std):
        BMC_std = 0.3
    return BMC_mean, BMC_std


def true_value():
    x_debug = np.linspace(-2, 2, 200)
    y_true_list = []
    cov = 0.8
    for x in tqdm(x_debug):
        mu_y_x = cov * x
        cov_y_x = 1 - cov * cov
        y = np.random.normal(size=(10000, 1)) * np.sqrt(cov_y_x) + mu_y_x
        gy = g(y)
        y_true_list.append(gy.mean())

    y_true = np.array(y_true_list)
    np.save('./data/toy_X.npy', x_debug)
    np.save('./data/toy_EgY_X.npy', y_true)
    plt.figure()
    plt.plot(x_debug, y_true)
    plt.xlabel(r"$X$")
    plt.ylabel(r"$\mathbb{E}[g(Y) \mid X]$")
    plt.title("True value for toy experiment")
    plt.savefig("./data/true_distribution_toy.pdf")
    # plt.show()
    plt.close()
    pause = True
    return


def GP(x_test, x, ny, BMC_x_mean_orig, BMC_x_std_orig, rng_key):
    """
    :param x_test: The value of x to evaluate
    :param x: The observed x
    :param BMC_x_mean_orig: The BMC estimate of \int g(y)p(y|x)dx at observed x
    :param BMC_x_mean_orig: The BMC uncertainty of \int g(y)p(y|x)dx at observed x
    :param rng_key:
    :return:
    """
    BMC_mean = BMC_x_mean_orig.mean()
    BMC_x = BMC_x_mean_orig - BMC_mean
    n = x.shape[0]
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    eps = 1e-6
    noise = 0.001  # This is used when we do not have BMC std.

    log_l_init = log_l = jnp.log(0.5)
    A_init = A = 1.0  # / jnp.sqrt(n)
    c_init = c = noise
    opt_state = optimizer.init((log_l_init, c_init, A_init))

    @jax.jit
    def nllk_func(log_l, c, A):
        l = jnp.exp(log_l)
        l = 0.5
        n = x.shape[0]
        K = 1 * my_RBF(x, x, l) + jnp.diag(BMC_x_std_orig ** 2) + noise * jnp.eye(n)
        K_inv = jnp.linalg.inv(K)
        nll = -(-0.5 * BMC_x.T @ K_inv @ BMC_x - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / n
        return nll[0][0]

    @jax.jit
    def step(log_l, c, A, opt_state, rng_key):
        nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1, 2))(log_l, c, A)
        updates, opt_state = optimizer.update(grads, opt_state, (log_l, c, A))
        log_l, c, A = optax.apply_updates((log_l, c, A), updates)
        return log_l, c, A, opt_state, nllk_value

    # Debug code
    log_l_debug_list = []
    A_debug_list = []
    nll_debug_list = []
    c_debug_list = []
    for _ in range(2000):
        rng_key, _ = jax.random.split(rng_key)
        log_l, c, A, opt_state, nllk_value = step(log_l, c, A, opt_state, rng_key)
        # Debug code
        log_l_debug_list.append(log_l)
        A_debug_list.append(A)
        c_debug_list.append(c)
        nll_debug_list.append(nllk_value)
    # Debug code
    # fig = plt.figure(figsize=(15, 6))
    # ax_1, ax_2, ax_3, ax_4 = fig.subplots(1, 4)
    # ax_1.plot(log_l_debug_list)
    # ax_2.plot(c_debug_list)
    # ax_3.plot(A_debug_list)
    # ax_4.plot(nll_debug_list)
    # plt.show()

    l = jnp.exp(log_l)
    K_train_train = A * my_RBF(x, x, l) + jnp.diag(BMC_x_std_orig ** 2) + c * jnp.eye(n)
    K_test_train = A * one_d_my_RBF(x_test, x, l)[None, :]
    K_test_test = A * one_d_my_RBF(x_test, x_test, l)[None][None] + c
    K_inv = jnp.linalg.inv(K_train_train)
    mean_true = (K_test_train @ K_inv @ BMC_x).squeeze()
    mean_true += BMC_mean
    std_true = jnp.sqrt(K_test_test + jnp.mean(BMC_x_std_orig ** 2) - K_test_train @ K_inv @ K_test_train.T).squeeze()

    # Debug code
    x_debug = jnp.linspace(-2, 2, 200)[:, None]
    K_train_train_debug = A * my_RBF(x, x, l) + jnp.diag(BMC_x_std_orig ** 2) + c * jnp.eye(n)
    K_test_train_debug = A * my_RBF(x_debug, x, l)
    K_test_test_debug = A * my_RBF(x_debug, x_debug, l) + c * jnp.eye(200)
    K_inv_debug = jnp.linalg.inv(K_train_train_debug)
    mean = (K_test_train_debug @ K_inv_debug @ BMC_x).squeeze()
    mean += BMC_mean
    std = jnp.diag(jnp.sqrt(K_test_test_debug + jnp.mean(BMC_x_std_orig ** 2) - K_test_train_debug @ K_inv_debug @ K_test_train_debug.T)).squeeze()
    y_true = jnp.load('./data/toy_EgY_X.npy')
    plt.figure()
    plt.scatter(x.squeeze(), BMC_x_mean_orig.squeeze())
    plt.plot(x_debug.squeeze(), mean, color='b')
    plt.plot(x_debug.squeeze(), y_true, color='red')
    plt.fill_between(x_debug.squeeze(), mean - std, mean + std, alpha=0.2, color='b')
    plt.savefig(f"./results/toy/figures/GP_toy_X_{n}_Y_{ny}.pdf")
    # plt.show()
    plt.close()
    pause = True
    return mean_true, std_true


def main():
    seed = int(time.time())
    # seed = 0
    rng_key = jax.random.PRNGKey(seed)

    # This is the joint distribution of x and y.
    covariance_matrix = jnp.array([[1, 0.8], [0.8, 1]])
    cov = 0.8

    # This is the number of x that we observe
    Nx_list = jnp.array([3, 5, 10, 20, 30])
    # Nx_list = jnp.array([3, 5])
    Ny_list = jnp.array([3, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # Ny_list = jnp.array([3, 5])
    # This is the value of x that we want to predict
    x_pred = 0.5

    # ------------- This is where standard Monte Carlo starts --------------#
    MC_list = []
    mu_y_x = cov * x_pred
    cov_y_x = 1 - cov * cov

    # The MC is based on the conditioning distribution on the true value.
    for ny in Ny_list:
        rng_key, _ = jax.random.split(rng_key)
        y = jax.random.normal(rng_key, shape=(ny, 1)) * jnp.sqrt(cov_y_x) + mu_y_x
        gY = g(y)
        int_MC = gY.mean()
        MC_list.append(int_MC)
    # ------------- This is where getting the true value starts --------------#

    # ------------- This is where standard Monte Carlo ends --------------#
    y = jax.random.normal(rng_key, shape=(1000000, 1)) * jnp.sqrt(cov_y_x) + mu_y_x
    gy = g(y)
    true_value = gy.mean()

    # ------------- This is where getting the true value ends --------------#

    # ------------- This is where importance sampling starts --------------#
    def gaussian_llk(x, mean, std):
        return (1.0 / (jnp.sqrt(2 * math.pi) * std)) * jnp.exp(-(x - mean) ** 2 / 2 / (std ** 2))

    Importance_dict = {}
    mu_y_x_pred = cov * x_pred
    cov_y_x_pred = 1 - cov * cov

    for nx in Nx_list:
        mu_y_x_obs = []
        cov_y_x_obs = []
        importance_sampling_MC_list = []
        rng_key, _ = jax.random.split(rng_key)
        x_obs = jax.random.normal(rng_key, shape=(nx, 1))
        for x in x_obs:
            mu_y_x_obs.append(cov * x)
            cov_y_x_obs.append(1 - cov * cov)

        for ny in Ny_list:
            importance_sampling_MC = 0
            for i in range(len(x_obs)):
                rng_key, _ = jax.random.split(rng_key)
                y = jax.random.normal(rng_key, shape=(ny, 1)) * jnp.sqrt(cov_y_x_obs[i]) + mu_y_x_obs[i]
                gY = g(y)
                p_y_x_pred = gaussian_llk(y, mu_y_x_pred, jnp.sqrt(cov_y_x_pred))
                p_y_x_obs = gaussian_llk(y, mu_y_x_obs[i], jnp.sqrt(cov_y_x_obs[i]))
                importance_sampling_MC += (gY * p_y_x_pred / p_y_x_obs).mean()
            importance_sampling_MC_list.append(importance_sampling_MC / len(x_obs))
        Importance_dict[f"{nx}"] = importance_sampling_MC_list

    # ------------- This is where importance sampling ends --------------#

    # ------------- This is where Conditional Bayesian Quadrature starts --------------#
    BMC_mean_dict = {}
    BMC_std_dict = {}

    for nx in Nx_list:
        mu_y_x_obs = []
        cov_y_x_obs = []
        rng_key, _ = jax.random.split(rng_key)
        x_obs = jax.random.normal(rng_key, shape=(nx, 1))
        for x in x_obs:
            mu_y_x_obs.append(cov * x)
            cov_y_x_obs.append(1 - cov * cov)

        temp_mean_list = jnp.array([])
        temp_std_list = jnp.array([])
        for ny in Ny_list:
            BMC_mean_x_list = jnp.array([])
            BMC_std_x_list = jnp.array([])
            for i in tqdm(range(len(x_obs))):
                rng_key, _ = jax.random.split(rng_key)
                y = jax.random.normal(rng_key, shape=(ny, 1)) * jnp.sqrt(cov_y_x_obs[i]) + mu_y_x_obs[i]
                gy = g(y)
                BMC_mean, BMC_std = CBQ(x_obs, y, gy, mu_y_x_obs[i], jnp.sqrt(cov_y_x_obs[i]), rng_key)
                BMC_mean_x_list = jnp.append(BMC_mean_x_list, BMC_mean)
                BMC_std_x_list = jnp.append(BMC_std_x_list, BMC_std)

            # The second GP - regress BMC_mean_x to x_obs
            mean, std = GP(jnp.array([[x_pred]]), x_obs, ny, BMC_mean_x_list[:, None], BMC_std_x_list[:, None], rng_key)
            temp_mean_list = jnp.append(temp_mean_list, mean)
            temp_std_list = jnp.append(temp_std_list, std)

        BMC_mean_dict[f"{nx}"] = temp_mean_list
        BMC_std_dict[f"{nx}"] = temp_std_list
        pause = True
    # ------------- This is where Conditional Bayesian Quadrature ends --------------#

    with open('./results/toy/BMC_mean', 'wb') as f:
        pickle.dump(BMC_mean_dict, f)
    with open('./results/toy/BMC_std', 'wb') as f:
        pickle.dump(BMC_std_dict, f)
    with open('./results/toy/importance_sampling', 'wb') as f:
        pickle.dump(Importance_dict, f)
    jnp.save('./results/toy/MC', jnp.array(MC_list))

    fig, axs = plt.subplots(len(Nx_list), 1, figsize=(20, 40))
    # axs = axs.flatten()


    for i, ax in enumerate(axs):
        Nx = Nx_list[i]
        axs[i].set_ylim(0, 1.5)
        axs[i].axhline(y=true_value, linestyle='--', color='black', label='true value')
        axs[i].plot(Ny_list, MC_list, color='b', label='MC')
        axs[i].plot(Ny_list, BMC_mean_dict[f"{Nx}"], color='r', label=f'BMC Nx = {Nx}')
        axs[i].fill_between(Ny_list, BMC_mean_dict[f"{Nx}"] - 2 * BMC_std_dict[f"{Nx}"],
                            BMC_mean_dict[f"{Nx}"] + 2 * BMC_std_dict[f"{Nx}"], color='r', alpha=0.2)
        axs[i].plot(Ny_list, Importance_dict[f"{Nx}"], color='g', label=f'IS Nx = {Nx}')
        axs[i].legend()
    plt.savefig("./results/toy/figures/all_methods.pdf")
    # plt.show()
    plt.close()
    return


if __name__ == '__main__':
    main()
    # true_value()
