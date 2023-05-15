import jax
import jax.numpy as jnp
import os
import shutil
import pwd
import jax.scipy
import jax.scipy.stats
import matplotlib.pyplot as plt
import argparse
import sensitivity_baselines
from tqdm import tqdm
from kernels import *
from utils import sensitivity_utils
import time
import optax
from jax.config import config

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()


def generate_data(rng_key, D, N, noise):
    """
    :param rng_key:
    :param D: int
    :param N: int
    :param noise: std for Gaussian likelihood
    :return: X is N*(D-1), Y is N*1
    """
    rng_key, _ = jax.random.split(rng_key)
    X = jax.random.uniform(rng_key, shape=(N, D - 1), minval=-1.0, maxval=1.0)
    X_with_one = jnp.hstack([X, jnp.ones([X.shape[0], 1])])
    rng_key, _ = jax.random.split(rng_key)
    beta_true = jax.random.normal(rng_key, shape=(D, 1))
    rng_key, _ = jax.random.split(rng_key)
    Y = X_with_one @ beta_true + jax.random.normal(rng_key, shape=(N, 1)) * noise
    return X, Y


# @jax.jit
def posterior_full(X, Y, prior_cov, noise):
    """
    :param prior_cov: (N3, D)
    :param X: (N, D-1)
    :param Y: (N, 1)
    :param noise: float
    :return: (N3, D), (N3, D, D)
    """
    X_with_one = jnp.hstack([X, jnp.ones([X.shape[0], 1])])
    D = prior_cov.shape[-1]
    prior_cov_inv = 1. / prior_cov
    # (N3, D, D)
    prior_cov_inv = jnp.einsum('ij,jk->ijk', prior_cov_inv, jnp.eye(D))
    beta_inv = noise ** 2
    beta = 1. / beta_inv
    post_cov = jnp.linalg.inv(prior_cov_inv + beta * X_with_one.T @ X_with_one)
    post_mean = beta * post_cov @ X_with_one.T @ Y
    return post_mean.squeeze(), post_cov


# @jax.jit
def normal_logpdf(x, mu, Sigma):
    """
    :param x: (N1, N2, D)
    :param mu: (N3, D)
    :param Sigma: (N3, D, D)
    :return: (N1, N2, N3)
    """
    D = x.shape[-1]
    x_expanded = jnp.expand_dims(x, 2)
    mean_expanded = jnp.expand_dims(mu, (0, 1))
    # covariance_expanded = jnp.expand_dims(covariance, 0)

    diff = x_expanded - mean_expanded
    precision_matrix = jnp.linalg.inv(Sigma)
    exponent = -0.5 * jnp.einsum('nijk, jkl, nijl->nij', diff, precision_matrix, diff)
    normalization = -0.5 * (D * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(jnp.linalg.det(Sigma)))
    return normalization + exponent


# @jax.jit
def posterior_log_llk(theta, prior_cov_base, X, Y, alpha, noise):
    """
    :param theta: (N1, N2, D)
    :param prior_cov_base: scalar
    :param X: data
    :param Y: data
    :param alpha: (N3, D)
    :param noise: scalar
    :return:
    """
    D = theta.shape[2]
    # prior_cov is (N3, D)
    prior_cov = jnp.ones([1, D]) * prior_cov_base + alpha
    # post_mean is (N3, D), post_cov is (N3, D, D), theta is (N1, N2, D)
    post_mean, post_cov = posterior_full(X, Y, prior_cov, noise)
    return normal_logpdf(theta, post_mean, post_cov)


def score_fn(y, mu, sigma):
    """
    return \nabla_y log p(y|mu, sigma)
    :param y: (N, D)
    :param mu: (D, )
    :param sigma: (D, D)
    :return: (N, D)
    """
    return -(y - mu[None, :]) @ jnp.linalg.inv(sigma)


def g1(y):
    """
    :param y: y is a N * D array
    """
    return y.sum(1)


def g1_ground_truth(mu, Sigma):
    return mu.sum()


def g2(y):
    """
    :param y: y is a N * D array
    """
    D = y.shape[1]
    return 10 * jnp.exp(-0.5 * ((y ** 2).sum(1) / (D ** 1))) + (y ** 2).sum(1)


def g2_ground_truth(mu, Sigma):
    """
    :param mu: (D, )
    :param Sigma: (D, D)
    :return: scalar
    """
    D = mu.shape[0]
    analytical_1 = jnp.exp(-0.5 * mu.T @ jnp.linalg.inv(jnp.eye(D) * (D ** 1) + Sigma) @ mu)
    analytical_2 = jnp.linalg.det(jnp.eye(D) + Sigma / (D ** 1)) ** (-0.5)
    analytical = analytical_1 * analytical_2
    return 10 * analytical + jnp.diag(Sigma).sum() + mu.T @ mu


def g3(y):
    return (y ** 2).sum(1)


def g3_ground_truth(mu, Sigma):
    return jnp.diag(Sigma).sum() + mu.T @ mu


def g4(y):
    """
    Only for D = 2
    :param y: (N, D)
    :return: (N, )
    """
    pred = jnp.array([0.3, 1.0])
    return y @ pred


def g4_ground_truth(mu, Sigma):
    """
    Only for D = 2
    :param mu: (D, )
    :param Sigma: (D, D)
    :return: scalar
    """
    pred = jnp.array([0.3, 1.0])
    return mu.T @ pred


def Monte_Carlo(gy):
    return gy.mean(0)


# @jax.jit
def Bayesian_Monte_Carlo_RBF(rng_key, y, gy, mu_y_x, sigma_y_x):
    """
    :param sigma_y_x: (D, D)
    :param mu_y_x: (D, )
    :param rng_key:
    :param y: (N, D)
    :param gy: (N, )
    :return:
    """
    N, D = y.shape[0], y.shape[1]
    eps = 1e-6

    l_array = jnp.array([0.1, 0.6, 1.0, 3.0]) * D
    nll_array = 0 * l_array
    A_list = []

    for i, l in enumerate(l_array):
        K_no_scale = my_RBF(y, y, l)
        A = gy.T @ K_no_scale @ gy / N
        A_list.append(A)
        K = A * K_no_scale
        K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
        nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / N
        nll_array = nll_array.at[i].set(nll)

    if D > 2:
        l = l_array[nll_array.argmin()]
        A = A_list[nll_array.argmin()]
    else:
        A = 1.
        l = 1.

    K = A * my_RBF(y, y, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = A * kme_RBF_Gaussian(mu_y_x, sigma_y_x, l, y)
    varphi = A * kme_double_RBF_Gaussian(mu_y_x, sigma_y_x, l)

    BMC_mean = phi.T @ K_inv @ gy
    BMC_std = jnp.sqrt(jnp.abs(varphi - phi.T @ K_inv @ phi))
    pause = True
    return BMC_mean, BMC_std


def Bayesian_Monte_Carlo_Matern(rng_key, u, y, gy, mu_y_x, sigma_y_x):
    """
    We only implement this for D = 2.
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

    A = 1.
    l = 1.

    u1 = u[:, 0][:, None]
    u2 = u[:, 1][:, None]

    K = A * my_Matern(u1, u1, l) + A * my_Matern(u2, u2, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = A * kme_Matern_Gaussian(l, u1) + A * kme_Matern_Gaussian(l, u2)
    varphi = phi.mean()

    BMC_mean = phi.T @ K_inv @ gy
    BMC_std = jnp.sqrt(jnp.abs(varphi - phi.T @ K_inv @ phi))

    BMC_mean = BMC_mean.squeeze()
    BMC_std = BMC_std.squeeze()
    pause = True
    return BMC_mean, BMC_std


@partial(jax.jit)
def nllk_func(log_l, c, A, y, gy, score, eps):
    N = y.shape[0]
    l = jnp.exp(log_l)
    K = A * stein_Matern(y, y, l, score, score) + c + A * jnp.eye(N)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps))
    return nll


@partial(jax.jit, static_argnames=['optimizer'])
def step(log_l, c, A, opt_state, optimizer, y, gy, score, eps):
    nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1, 2))(log_l, c, A, y, gy, score, eps)
    updates, opt_state = optimizer.update(grads, opt_state, (log_l, c, A))
    log_l, c, A = optax.apply_updates((log_l, c, A), updates)
    return log_l, c, A, opt_state, nllk_value


def Bayesian_Monte_Carlo_Stein(rng_key, y, gy, mu_y_x, sigma_y_x, score):
    """
    We only implement this for D = 2.
    :param score: (N, D)
    :param sigma_y_x: (D, D)
    :param mu_y_x: (D, )
    :param rng_key:
    :param y: (N, D)
    :param gy: (N, )
    :return:
    """
    N, D = y.shape[0], y.shape[1]
    eps = 1e-6

    gy_standardized = gy / gy.mean()

    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    c_init = c = 1.0
    log_l_init = log_l = jnp.log(2.0)
    A_init = A = 1.0
    opt_state = optimizer.init((log_l_init, c_init, A_init))

    # # ============= Debug code =============
    l_debug_list = []
    c_debug_list = []
    A_debug_list = []
    nll_debug_list = []
    # ============= Debug code =============
    for _ in range(100):
        rng_key, _ = jax.random.split(rng_key)
        log_l, c, A, opt_state, nllk_value = step(log_l, c, A, opt_state, optimizer, y, gy_standardized, score, eps)
        # ============= Debug code =============
        if jnp.isnan(nllk_value):
            pause = True
        l_debug_list.append(jnp.exp(log_l))
        c_debug_list.append(c)
        A_debug_list.append(A)
        nll_debug_list.append(nllk_value)

    fig = plt.figure(figsize=(15, 6))
    ax_1, ax_2, ax_3, ax_4 = fig.subplots(1, 4)
    ax_1.plot(l_debug_list)
    ax_2.plot(c_debug_list)
    ax_3.plot(A_debug_list)
    ax_4.plot(nll_debug_list)
    plt.show()
    # ============= Debug code =============

    l = jnp.exp(log_l)
    K = A * stein_Matern(y, y, l, score, score) + c + A * jnp.eye(N)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    BMC_mean = c * (K_inv @ gy_standardized).sum()
    BMC_mean *= gy.mean()
    BMC_std = jnp.sqrt(jnp.abs(c - K_inv.sum() * c ** 2))
    pause = True
    return BMC_mean, BMC_std


# @jax.jit
def GP(rng_key, psi_y_x_mean, psi_y_x_std, X, X_prime, eps, kernel_fn):
    """
    :param kernel_fn: Matern or RBF
    :param eps:
    :param psi_y_x_mean: (n_alpha, )
    :param psi_y_x_std: (n_alpha, )
    :param X: (n_alpha, D)
    :param X_prime: (N_test, D)
    :return:
    """
    n_alpha, D = X.shape[0], X.shape[1]
    l_array = jnp.array([0.3, 1.0, 2.0, 3.0]) * D

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
                    K = A * kernel_fn(X, X, l) + jnp.eye(n_alpha) * sigma
                    K_inv = jnp.linalg.inv(K)
                    nll = -(-0.5 * psi_y_x_mean.T @ K_inv @ psi_y_x_mean - 0.5 * jnp.log(
                        jnp.linalg.det(K) + 1e-6)) / n_alpha
                    nll_array = nll_array.at[i, j].set(nll)
        min_index_flat = jnp.argmin(nll_array)
        i1, i2, i3 = jnp.unravel_index(min_index_flat, nll_array.shape)
        l = l_array[i1]
        A = A_array[i2]
        sigma = sigma_array[i3]
    else:
        for i, l in enumerate(l_array):
            K_no_scale = kernel_fn(X, X, l)
            A = psi_y_x_mean.T @ K_no_scale @ psi_y_x_mean / n_alpha
            A_array = A_array.at[i].set(A)
            K = A * kernel_fn(X, X, l) + eps * jnp.eye(n_alpha) + jnp.diag(psi_y_x_std ** 2)
            K_inv = jnp.linalg.inv(K)
            nll = -(-0.5 * psi_y_x_mean.T @ K_inv @ psi_y_x_mean - 0.5 * jnp.log(jnp.linalg.det(K) + 1e-6)) / n_alpha
            nll_array = nll_array.at[i].set(nll)

        l = l_array[jnp.argmin(nll_array)]
        A = A_array[jnp.argmin(nll_array)]

    if psi_y_x_std is None:
        K_train_train = A * kernel_fn(X, X, l) + jnp.eye(n_alpha) * sigma
        K_train_train_inv = jnp.linalg.inv(K_train_train)
        K_test_train = A * kernel_fn(X_prime, X, l)
        K_test_test = A * kernel_fn(X_prime, X_prime, l) + jnp.eye(X_prime.shape[0]) * sigma
    else:
        # print(A)
        if n_alpha == 10:
            A = 10.0
        else:
            A = 1.0
        K_train_train = A * kernel_fn(X, X, l) + eps * jnp.eye(n_alpha) + jnp.diag(psi_y_x_std ** 2)
        K_train_train_inv = jnp.linalg.inv(K_train_train)
        K_test_train = A * kernel_fn(X_prime, X, l)
        K_test_test = A * kernel_fn(X_prime, X_prime, l) + eps * jnp.eye(X_prime.shape[0])
    mu_y_x = K_test_train @ K_train_train_inv @ psi_y_x_mean
    var_y_x = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    var_y_x = jnp.abs(var_y_x)
    std_y_x = jnp.sqrt(var_y_x)
    pause = True
    return mu_y_x, std_y_x


def main(args):
    seed = args.seed
    rng_key = jax.random.PRNGKey(seed)
    D = args.dim
    prior_cov_base = 2.0
    noise = 1.0
    sample_size = 5000
    test_num = 200
    data_number = 5
    # X is (N, D-1), Y is (N, 1)
    rng_key, _ = jax.random.split(rng_key)
    X, Y = generate_data(rng_key, D, data_number, noise)

    if args.g_fn == 'g1':
        g = g1
        g_ground_truth_fn = g1_ground_truth
    elif args.g_fn == 'g2':
        g = g2
        g_ground_truth_fn = g2_ground_truth
    elif args.g_fn == 'g3':
        g = g3
        g_ground_truth_fn = g3_ground_truth
    elif args.g_fn == 'g4':
        g = g4
        g_ground_truth_fn = g4_ground_truth
    else:
        raise ValueError('g_fn must be g1 or g2 or g3 or g4!')

    # N_alpha_array = jnp.array([5, 10])
    N_alpha_array = jnp.array([10, 50, 100])
    # N_alpha_array = jnp.concatenate((jnp.array([3, 5]), jnp.arange(10, 150, 10)))

    # N_theta_array = jnp.array([10, 30])
    N_theta_array = jnp.array([10, 50, 100])
    # N_theta_array = jnp.concatenate((jnp.array([3, 5]), jnp.arange(10, 150, 10)))

    # This is the test point
    alpha_test_line = jax.random.uniform(rng_key, shape=(test_num, D), minval=-1.0, maxval=1.0)
    cov_test_line = jnp.array([[prior_cov_base] * D]) + alpha_test_line
    ground_truth = jnp.zeros(test_num)

    post_mean, post_var = posterior_full(X, Y, cov_test_line, noise)
    # post_mean: (test_num, D), post_var: (test_num, D, D)
    for i in range(test_num):
        ground_truth = ground_truth.at[i].set(g_ground_truth_fn(post_mean[i, :], post_var[i, :, :]))
    jnp.save(f"{args.save_path}/test_line.npy", alpha_test_line)
    jnp.save(f"{args.save_path}/ground_truth.npy", ground_truth)

    for n_alpha in N_alpha_array:
        rng_key, _ = jax.random.split(rng_key)
        # This is X, size n_alpha * D
        if args.qmc:
            alpha_all = sensitivity_utils.qmc_uniform(-1.0, 1.0, D, n_alpha)
        else:
            alpha_all = jax.random.uniform(rng_key, shape=(n_alpha, D), minval=-1.0, maxval=1.0)

        rmse_BMC_array = jnp.zeros(len(N_theta_array))
        rmse_KMS_array = jnp.zeros(len(N_theta_array))
        rmse_LSMC_array = jnp.zeros(len(N_theta_array))
        rmse_IS_array = jnp.zeros(len(N_theta_array))

        time_BMC_array = jnp.zeros(len(N_theta_array))
        time_KMS_array = jnp.zeros(len(N_theta_array))
        time_LSMC_array = jnp.zeros(len(N_theta_array))
        time_IS_array = jnp.zeros(len(N_theta_array))

        for j, n_theta in enumerate(tqdm(N_theta_array)):
            psi_mean_array = jnp.zeros(n_alpha)
            psi_std_array = jnp.zeros(n_alpha)
            mc_mean_array = jnp.zeros(n_alpha)

            # This is Y, size n_alpha * n_theta * D
            samples_all = jnp.zeros([n_alpha, n_theta, D]) + 0.0
            # This is g(Y), size n_alpha * n_theta
            g_samples_all = jnp.zeros([n_alpha, n_theta]) + 0.0
            u_all = jnp.zeros([n_alpha, n_theta, D]) + 0.0
            score_all = jnp.zeros([n_alpha, n_theta, D]) + 0.0

            prior_cov = jnp.array([[prior_cov_base] * D]) + alpha_all
            mu_y_x_all, var_y_x_all = posterior_full(X, Y, prior_cov, noise)

            for i in range(n_alpha):
                rng_key, _ = jax.random.split(rng_key)
                if args.qmc:
                    samples, u = sensitivity_utils.qmc_gaussian(mu_y_x_all[i, :], var_y_x_all[i, :, :], n_theta)
                    score = score_fn(samples, mu_y_x_all[i, :], var_y_x_all[i, :, :])
                else:
                    u = jax.random.multivariate_normal(rng_key, mean=jnp.zeros_like(mu_y_x_all[i, :]),
                                                       cov=jnp.diag(jnp.ones_like(mu_y_x_all[i, :])),
                                                       shape=(n_theta,))
                    L = jnp.linalg.cholesky(var_y_x_all[i, :, :])
                    samples = mu_y_x_all[i, :] + jnp.matmul(L, u.T).T
                    score = score_fn(samples, mu_y_x_all[i, :], var_y_x_all[i, :, :])
                score_all = score_all.at[i, :, :].set(score)
                u_all = u_all.at[i, :, :].set(u)
                samples_all = samples_all.at[i, :, :].set(samples)
                g_samples_all = g_samples_all.at[i, :].set(g(samples))

            for i in range(n_alpha):
                samples_i = samples_all[i, :, :]
                g_samples_i = g_samples_all[i, :]
                u_i = u_all[i, :, :]
                score_i = score_all[i, :, :]
                mu_y_x_i = mu_y_x_all[i, :]
                var_y_x_i = var_y_x_all[i, :, :]

                tt0 = time.time()
                if args.kernel_y == "RBF":
                    psi_mean, psi_std = Bayesian_Monte_Carlo_RBF(rng_key, samples_i, g_samples_i, mu_y_x_i, var_y_x_i)
                elif args.kernel_y == "Matern":
                    if D > 2:
                        raise NotImplementedError("Matern kernel is only implemented for D=2")
                    psi_mean, psi_std = Bayesian_Monte_Carlo_Matern(rng_key, u_i, samples_i, g_samples_i, mu_y_x_i,
                                                                    var_y_x_i)
                elif args.kernel_y == "Stein":
                    if D > 2:
                        raise NotImplementedError("Stein kernel is only implemented for D=2")
                    psi_mean, psi_std = Bayesian_Monte_Carlo_Stein(rng_key, samples_i, g_samples_i, mu_y_x_i, var_y_x_i,
                                                                   score_i)
                else:
                    raise NotImplementedError("Kernel not implemented")
                tt1 = time.time()

                psi_mean_array = psi_mean_array.at[i].set(psi_mean)
                psi_std_array = psi_std_array.at[i].set(psi_std if not jnp.isnan(psi_std) else 0.01)

                MC_value = g_samples_i.mean()
                mc_mean_array = mc_mean_array.at[i].set(MC_value)

                # ============= Debug code =============
                # true_value = g_ground_truth_fn(mu_y_x_i, var_y_x_i)
                # BMC_value = psi_mean
                # print("=============")
                # print('True value', true_value)
                # print(f'MC with {n_theta} number of Y', MC_value)
                # print(f'BMC with {n_theta} number of Y', BMC_value)
                # print(f'BMC uncertainty {psi_std}')
                # print(f"=============")
                # pause = True
                # ============= Debug code =============

            _, _ = GP(rng_key, mc_mean_array, None, alpha_all, alpha_test_line, eps=1e-1, kernel_fn=my_RBF)
            rng_key, _ = jax.random.split(rng_key)
            t0 = time.time()
            KMS_mean, KMS_std = GP(rng_key, mc_mean_array, None, alpha_all, alpha_test_line,
                                   eps=0., kernel_fn=my_RBF)
            time_KMS = time.time() - t0
            time_KMS_array = time_KMS_array.at[j].set(time_KMS)

            rng_key, _ = jax.random.split(rng_key)
            t0 = time.time()
            if args.kernel_x == "RBF":
                BMC_mean, BMC_std = GP(rng_key, psi_mean_array, psi_std_array, alpha_all, alpha_test_line,
                                       eps=psi_std_array.mean(), kernel_fn=my_RBF)
            elif args.kernel_x == "Matern":
                BMC_mean, BMC_std = GP(rng_key, psi_mean_array, psi_std_array, alpha_all, alpha_test_line,
                                       eps=psi_std_array.mean(), kernel_fn=my_Matern)
            else:
                raise NotImplementedError(f"Unknown kernel {args.kernel_x}")
            time_BMC = time.time() - t0 + (tt1 - tt0) * n_alpha
            time_BMC_array = time_BMC_array.at[j].set(time_BMC)

            _, _ = sensitivity_baselines.polynomial(alpha_all, samples_all, g_samples_all, alpha_test_line)
            t0 = time.time()
            LSMC_mean, LSMC_std = sensitivity_baselines.polynomial(alpha_all, samples_all, g_samples_all,
                                                                   alpha_test_line)
            time_LSMC = time.time() - t0
            time_LSMC_array = time_LSMC_array.at[j].set(time_LSMC)

            log_py_x_fn = partial(posterior_log_llk, X=X, Y=Y, noise=noise, prior_cov_base=prior_cov_base)
            _, _ = sensitivity_baselines.importance_sampling(log_py_x_fn, alpha_all,
                                                             samples_all, g_samples_all, alpha_test_line)
            t0 = time.time()
            IS_mean, IS_std = sensitivity_baselines.importance_sampling(log_py_x_fn, alpha_all,
                                                                        samples_all, g_samples_all, alpha_test_line)
            time_IS = time.time() - t0
            time_IS_array = time_IS_array.at[j].set(time_IS)

            rmse_BMC = jnp.sqrt(jnp.mean((BMC_mean - ground_truth) ** 2))
            rmse_KMS = jnp.sqrt(jnp.mean((KMS_mean - ground_truth) ** 2))
            rmse_LSMC = jnp.sqrt(jnp.mean((LSMC_mean - ground_truth) ** 2))
            rmse_IS = jnp.sqrt(jnp.mean((IS_mean - ground_truth) ** 2))

            rmse_BMC_array = rmse_BMC_array.at[j].set(rmse_BMC)
            rmse_KMS_array = rmse_KMS_array.at[j].set(rmse_KMS)
            rmse_LSMC_array = rmse_LSMC_array.at[j].set(rmse_LSMC)
            rmse_IS_array = rmse_IS_array.at[j].set(rmse_IS)

            calibration = sensitivity_utils.calibrate(ground_truth, BMC_mean, jnp.diag(BMC_std))
            sensitivity_utils.save(args, n_alpha, n_theta, rmse_BMC, rmse_KMS, rmse_LSMC, rmse_IS,
                                   time_BMC, time_KMS, time_LSMC, time_IS, calibration)

            # ============= Debug code =============
            # print(f"=============")
            # print(f"RMSE of BMC with {n_alpha} number of X and {n_theta} number of Y", rmse_BMC)
            # print(f"RMSE of KMS with {n_alpha} number of X and {n_theta} number of Y", rmse_KMS)
            # print(f"RMSE of LSMC with {n_alpha} number of X and {n_theta} number of Y", rmse_LSMC)
            # print(f"RMSE of IS with {n_alpha} number of X and {n_theta} number of Y", rmse_IS)
            # print(f"=============")

            # ============= Debug code =============
            # print(f"=============")
            # print(f"Time of BMC with {n_alpha} number of X and {n_theta} number of Y", time_BMC)
            # print(f"Time of KMS with {n_alpha} number of X and {n_theta} number of Y", time_KMS)
            # print(f"Time of LSMC with {n_alpha} number of X and {n_theta} number of Y", time_LSMC)
            # print(f"Time of IS with {n_alpha} number of X and {n_theta} number of Y", time_IS)
            # print(f"=============")
            # pause = True
            # ============= Debug code =============

        # # ============= Debug code =============
        # fig = plt.figure(figsize=(15, 6))
        # ax_1, ax_2 = fig.subplots(1, 2)
        # ax_1.plot(N_theta_array, mse_BMC_array, label='BMC')
        # ax_1.plot(N_theta_array, mse_KMS_array, label='KMS')
        # ax_1.plot(N_theta_array, mse_LSMC_array, label='LSMC')
        # ax_1.plot(N_theta_array, mse_IS_array, label='IS')
        # ax_1.legend()
        # ax_1.set_xlabel('Number of Y')
        # ax_1.set_yscale('log')
        # ax_1.set_ylabel('RMSE')
        # ax_2.plot(N_theta_array, time_BMC_array, label='BMC')
        # ax_2.plot(N_theta_array, time_KMS_array, label='KMS')
        # ax_2.plot(N_theta_array, time_LSMC_array, label='LSMC')
        # ax_2.plot(N_theta_array, time_IS_array, label='IS')
        # ax_2.legend()
        # ax_2.set_xlabel('Number of Y')
        # ax_2.set_ylabel('Time')
        #
        # plt.suptitle(f"Sensitivity analysis with {n_alpha} number of X")
        # plt.savefig(f"{args.save_path}/sensitivity_conjugate_Nx_{n_alpha}.pdf")
        # # plt.show()
        # plt.close()
        # ============= Debug code =============

    # =================================
    # For very very large Nx and Ny
    # Test the performance of other methods
    # =================================
    n_alpha = 500
    n_theta = 1000
    rng_key, _ = jax.random.split(rng_key)
    # This is X, size n_alpha * D
    alpha_all = jax.random.uniform(rng_key, shape=(n_alpha, D), minval=-1.0, maxval=1.0)

    # This is Y, size n_alpha * n_theta * D
    samples_all = jnp.zeros([n_alpha, n_theta, D])
    # This is g(Y), size n_alpha * sample_size
    g_samples_all = jnp.zeros([n_alpha, n_theta])

    prior_cov = jnp.array([[prior_cov_base] * D]) + alpha_all
    mu_y_x_all, var_y_x_all = posterior_full(X, Y, prior_cov, noise)

    for i in range(n_alpha):
        rng_key, _ = jax.random.split(rng_key)
        samples = jax.random.multivariate_normal(rng_key, mean=mu_y_x_all[i, :], cov=var_y_x_all[i, :, :],
                                                 shape=(n_theta,))
        samples_all = samples_all.at[i, :, :].set(samples)
        g_samples_all = g_samples_all.at[i, :].set(g(samples))

    mc_mean_array = g_samples_all.mean(axis=1)
    rng_key, _ = jax.random.split(rng_key)
    t0 = time.time()
    KMS_mean, KMS_std = GP(rng_key, mc_mean_array, None, alpha_all, alpha_test_line, eps=0., kernel_fn=my_RBF)
    time_KMS_large = time.time() - t0

    t0 = time.time()
    LSMC_mean, LSMC_std = sensitivity_baselines.polynomial(alpha_all, samples_all, g_samples_all, alpha_test_line)
    time_LSMC_large = time.time() - t0

    log_py_x_fn = partial(posterior_log_llk, X=X, Y=Y, noise=noise, prior_cov_base=prior_cov_base)
    t0 = time.time()
    # IS_mean, IS_std = sensitivity_baselines.importance_sampling(log_py_x_fn, alpha_all, samples_all, g_samples_all, alpha_test_line)
    IS_mean = LSMC_mean
    time_IS_large = time.time() - t0

    rmse_KMS_large = jnp.sqrt(jnp.mean((KMS_mean - ground_truth) ** 2))
    rmse_LSMC_large = jnp.sqrt(jnp.mean((LSMC_mean - ground_truth) ** 2))
    rmse_IS_large = jnp.sqrt(jnp.mean((IS_mean - ground_truth) ** 2))

    sensitivity_utils.save_large(args, n_alpha, n_theta, rmse_KMS_large, rmse_LSMC_large, rmse_IS_large,
                                 time_KMS_large, time_LSMC_large, time_IS_large)

    # ============= Debug code =============
    # print(f"=============")
    # print(f"KMS rmse with {n_alpha} number of X and {n_theta} number of Y", rmse_KMS_large)
    # print(f"KMS time with {n_alpha} number of X and {n_theta} number of Y", time_KMS_large)
    # print(f"LSMC rmse with {n_alpha} number of X and {n_theta} number of Y", rmse_LSMC_large)
    # print(f"LSMC time with {n_alpha} number of X and {n_theta} number of Y", time_LSMC_large)
    # print(f"=============")
    # ============= Debug code =============
    return


def get_config():
    parser = argparse.ArgumentParser(description='Conditional Bayesian Quadrature for Bayesian sensitivity analysis')
    # Args settings
    parser.add_argument('--dim', type=int)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--g_fn', type=str, default=None)
    parser.add_argument('--qmc', action='store_true', default=False)
    parser.add_argument('--kernel_y', type=str)
    parser.add_argument('--kernel_x', type=str)
    args = parser.parse_args()
    return args


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/sensitivity_conjugate/'
    if args.qmc:
        args.save_path += f"seed_{args.seed}__dim_{args.dim}__function_{args.g_fn}__Ky_{args.kernel_y}" \
                          f"__Kx_{args.kernel_x}__qmc"
    else:
        args.save_path += f"seed_{args.seed}__dim_{args.dim}__function_{args.g_fn}__Ky_{args.kernel_y}" \
                          f"__Kx_{args.kernel_x}"
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
