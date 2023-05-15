import jax
import jax.numpy as jnp
import os
import pwd
import jax.scipy
import jax.scipy.stats
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import MCMC
import argparse
from sensitivity_baselines import *
from tqdm import tqdm
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from kernels import *
import optax
from utils import finance_utils, sensitivity_utils
import time
from jax.config import config

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

eps = 1e-6

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()


def generate_data(rng_key, D, num):
    rng_key, _ = jax.random.split(rng_key)
    x = jax.random.uniform(rng_key, shape=(num, D - 1), minval=-1.0, maxval=1.0)
    p = 1. / (1. + jnp.exp(- x.sum(-1)))
    rng_key, _ = jax.random.split(rng_key)
    Y = jax.random.bernoulli(rng_key, p)[:, None]
    # jnp.save(f'./data/sensitivity/data_y', Y)
    # jnp.save(f'./data/sensitivity/data_x', x)
    return x, Y


@jax.jit
def log_posterior(beta, x, y, prior_cov):
    """
    :param prior_cov: D*1 array
    :param beta: D*1 array
    :param x: N*2 array
    :param y: N*1 array
    :return:
    """
    D = prior_cov.shape[0]
    prior_cov = jnp.diag(prior_cov.squeeze())
    log_prior_beta = jax.scipy.stats.multivariate_normal.logpdf(beta.squeeze(),
                                                                mean=jnp.zeros([D]),
                                                                cov=prior_cov).sum()
    x_with_one = jnp.hstack([x, jnp.ones([x.shape[0], 1])])
    p = jax.nn.sigmoid(x_with_one @ beta)
    log_bern_llk = (y * jnp.log(p + eps) + (1 - y) * jnp.log(1 - p + eps)).sum()
    return (log_bern_llk + log_prior_beta).squeeze()


log_posterior_vmap = jax.vmap(log_posterior, in_axes=(0, None, None, None), out_axes=0)


def posterior(beta, x, y, prior_cov):
    """
    :param prior_cov: *1 array
    :param beta: Ny*3*1 array
    :param x: N*2 array
    :param y: N*1 array
    :return: Ny*1
    """
    return jnp.exp(log_posterior_vmap(beta, x, y, prior_cov))


def MCMC(rng_key, cov, nsamples, init_params, log_prob):
    rng_key, _ = jax.random.split(rng_key)

    @jax.jit
    def run_chain(rng_key, state):
        num_burnin_steps = int(1e3)
        # kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        #     tfp.mcmc.HamiltonianMonteCarlo(
        #         target_log_prob_fn=log_prob,
        #         num_leapfrog_steps=3,
        #         step_size=1.0),
        #         num_adaptation_steps=int(num_burnin_steps * 0.8))

        kernel = tfp.mcmc.NoUTurnSampler(log_prob, 1e-2)
        samples = tfp.mcmc.sample_chain(num_results=nsamples,
                                        num_burnin_steps=num_burnin_steps,
                                        current_state=state,
                                        kernel=kernel,
                                        trace_fn=None,
                                        seed=rng_key)
        return samples

    states = run_chain(rng_key, init_params)

    # # Debug code
    # D = states.shape[1]
    # fig = plt.figure(figsize=(5 * D, 10))
    # ax_list = fig.subplots(1, D)
    # prior_std = jnp.sqrt(cov.squeeze())
    #
    # for i, ax in enumerate(ax_list):
    #     x = jnp.linspace(0 - 3 * prior_std[i], 0 + 3 * prior_std[i], 100)
    #     beta_post = states[:, i, :]
    #     ax.plot(x, jax.scipy.stats.norm.pdf(x, 0, prior_std[i]), color='black', linewidth=5)
    #     ax.hist(np.array(beta_post), bins=10, alpha=0.8, density=True)
    # plt.show()

    pause = True
    return states


def g(y):
    """
    :param y: y is a N*3*1 array
    """
    return y.sum(1).squeeze(axis=-1)


def Monte_Carlo(gy):
    return gy.mean(0)

# @jax.jit
def Bayesian_Monte_Carlo(rng_key, y, gy, d_log_py, kernel_y):
    """
    :param rng_key:
    :param y: N * D * 1
    :param gy: N
    :param d_log_py: N * D * 1
    :param kernel_y: kernel function
    :return:
    """
    y = y[:, :, 0]
    N, D = y.shape[0], y.shape[1]
    d_log_py = d_log_py[:, :, 0]
    n = y.shape[0]
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    eps = 1e-6
    median_d = jnp.median(distance(y, y))
    gy_var = gy.var()
    c_init = c = 1.0 * gy_var
    log_l_init = log_l = jnp.log(median_d / jnp.sqrt(D))
    A_init = A = 1.0 * gy_var
    opt_state = optimizer.init((log_l_init, c_init, A_init))

    @jax.jit
    def nllk_func(log_l, c, A):
        l, c, A = jnp.exp(log_l), c, A
        n = y.shape[0]
        K = A * kernel_y(y, y, l, d_log_py, d_log_py) + c
        K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
        nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps))
        return nll

    @jax.jit
    def step(log_l, c, A, opt_state, rng_key):
        nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1, 2))(log_l, c, A)
        updates, opt_state = optimizer.update(grads, opt_state, (log_l, c, A))
        log_l, c, A = optax.apply_updates((log_l, c, A), updates)
        return log_l, c, A, opt_state, nllk_value

    # # Debug code
    # log_l_debug_list = []
    # c_debug_list = []
    # A_debug_list = []
    # nll_debug_list = []
    for _ in range(10000):
        rng_key, _ = jax.random.split(rng_key)
        log_l, c, A, opt_state, nllk_value = step(log_l, c, A, opt_state, rng_key)
        # # Debug code
        # if jnp.isnan(nllk_value):
        #     p = 1
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

    l, c, A = jnp.exp(log_l), c, A
    final_K = A * kernel_y(y, y, l, d_log_py, d_log_py) + c
    final_K_inv = jnp.linalg.inv(final_K + eps * jnp.eye(n))
    BMC_mean = c * (final_K_inv @ gy).sum()
    BMC_std = jnp.sqrt(c - final_K_inv.sum() * c * c)

    if jnp.isnan(BMC_std):
        BMC_std = 0.3
    pause = True
    return BMC_mean, BMC_std


@jax.jit
def GP(psi_y_x_mean, psi_y_x_std, X, x_prime):
    """
    :param psi_y_x_mean: n_alpha*1
    :param psi_y_x_std: n_alpha*1
    :param X: n_alpha*3
    :param x_prime: 1*3
    :return:
    """
    Nx = psi_y_x_mean.shape[0]
    Mu_standardized, Mu_mean, Mu_std = finance_utils.standardize(psi_y_x_mean)
    Sigma_standardized = psi_y_x_std / Mu_std
    X_standardized, X_mean, X_std = finance_utils.standardize(X)
    x_prime_standardized = (x_prime - X_mean) / X_std
    noise = 0.01
    lx = 0.5

    K_train_train = my_RBF(X_standardized, X_standardized, lx) + jnp.diag(
        Sigma_standardized) + noise * jnp.eye(Nx)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = my_RBF(x_prime_standardized, X_standardized, lx)
    K_test_test = my_RBF(x_prime_standardized, x_prime_standardized, lx) + noise
    mu_y_x_prime = K_test_train @ K_train_train_inv @ Mu_standardized
    var_y_x_prime = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    std_y_x_prime = jnp.sqrt(var_y_x_prime)

    mu_y_x_prime_original = mu_y_x_prime * Mu_std + Mu_mean
    std_y_x_prime_original = std_y_x_prime * Mu_std + jnp.mean(psi_y_x_std)
    return mu_y_x_prime_original, std_y_x_prime_original


def main(args):
    seed = args.seed
    rng_key = jax.random.PRNGKey(seed)
    D = args.dim
    prior_covariance = 5.0
    X, Y = generate_data(rng_key, D, 100)
    # X = jnp.load(f'./data/sensitivity/data_x.npy')
    # Y = jnp.load(f'./data/sensitivity/data_y.npy')

    # N_alpha_list = [5, 6]
    N_alpha_list = [3, 5, 10, 20, 30]
    N_beta_list = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # N_beta_list = [10, 30, 50, 100]
    N_MCMC = 5000

    cbq_mean_dict = {}
    cbq_std_dict = {}
    poly_mean_dict = {}
    poly_std_dict = {}
    IS_mean_dict = {}
    IS_std_dict = {}

    # This is the test point
    alpha_test = jax.random.uniform(rng_key, shape=(D, 1), minval=-1.0, maxval=1.0)
    cov_test = jnp.array([[prior_covariance] * D]).T + alpha_test
    log_prob = partial(log_posterior, x=X, y=Y, prior_cov=cov_test)
    grad_log_prob = jax.grad(log_prob, argnums=0)
    init_params = jnp.array([[0.1] * D]).T
    states_test = MCMC(rng_key, cov_test, N_MCMC * 2, init_params, log_prob)
    states_test = jnp.unique(states_test, axis=0)
    rng_key, _ = jax.random.split(rng_key)
    states_test = jax.random.permutation(rng_key, states_test)
    g_test_true = g(states_test).mean()

    for n_alpha in N_alpha_list:
        rng_key, _ = jax.random.split(rng_key)
        alpha_all = jax.random.uniform(rng_key, shape=(n_alpha, D), minval=-1.0, maxval=1.0)
        # This is X, size n_alpha*3
        cov_all = jnp.array([[prior_covariance] * D]) + alpha_all
        cbq_mean_array = jnp.array([])
        cbq_std_array = jnp.array([])
        poly_mean_array = jnp.array([])
        poly_std_array = jnp.array([])
        IS_mean_array = jnp.array([])
        IS_std_array = jnp.array([])

        states_all = {}
        g_states_all = {}
        for i in range(n_alpha):
            cov = cov_all[i, :][:, None]
            log_prob = partial(log_posterior, x=X, y=Y, prior_cov=cov)
            grad_log_prob = jax.grad(log_prob, argnums=0)

            init_params = jnp.array([[0.1] * D]).T
            states_temp = MCMC(rng_key, cov, N_MCMC, init_params, log_prob)
            states_temp = jnp.unique(states_temp, axis=0)
            rng_key, _ = jax.random.split(rng_key)
            states_temp = jax.random.permutation(rng_key, states_temp)
            states_all[f'{i}'] = states_temp
            g_states_all[f'{i}'] = g(states_temp)

        for n_beta in tqdm(N_beta_list):
            psi_mean_array = jnp.array([])
            psi_std_array = jnp.array([])
            logging = sensitivity_utils.init_logging()

            # This is Y and g(Y)
            states = jnp.zeros([n_alpha, n_beta, D, 1])
            g_states = jnp.zeros([n_alpha, n_beta])

            for i in range(n_alpha):
                rng_key, _ = jax.random.split(rng_key)
                ind = jax.random.permutation(rng_key, len(states_all[f'{i}']))[:n_beta]
                states_i = states_all[f'{i}'][ind, :, :]
                g_states_i = g(states_i)
                g_states_i_standardized, g_states_i_scale = sensitivity_utils.scale(g_states_i)
                states = states.at[i, :, :, :].set(states_i)
                g_states = g_states.at[i, :].set(g_states_i)
                d_log_pstates = grad_log_prob(states_i)

                psi_mean, psi_std = Bayesian_Monte_Carlo(rng_key, states_i,
                                                         g_states_i_standardized,
                                                         d_log_pstates,
                                                         stein_Gaussian)
                psi_mean_array = jnp.append(psi_mean_array, psi_mean * g_states_i_scale)
                psi_std_array = jnp.append(psi_std_array, psi_std * g_states_i_scale)

                true_value = g(states_all[f'{i}']).mean()
                BMC_value = psi_mean * g_states_i_scale
                MC_value = g_states_i.mean()
                # # Debug
                # print('True value', true_value)
                # print(f'MC with {n_beta} number of Y', MC_value)
                # print(f'BMC with {n_beta} number of Y', BMC_value)
                # print(f"=================")
                # pause = True
                logging = sensitivity_utils.update_log(args, n_alpha, n_beta, logging,
                                                       true_value, MC_value, BMC_value)

            BMC_mean, BMC_std = GP(psi_mean_array, psi_std_array, cov_all, cov_test.T)
            cbq_mean_array = jnp.append(cbq_mean_array, BMC_mean)
            cbq_std_array = jnp.append(cbq_std_array, BMC_std)

            mu_y_x_prime_poly, std_y_x_prime_poly = polynomial(cov_all, states, g_states, cov_test.T)
            poly_mean_array = jnp.append(poly_mean_array, mu_y_x_prime_poly)
            poly_std_array = jnp.append(poly_std_array, std_y_x_prime_poly)

            py_x_fn = partial(posterior, x=X, y=Y)
            mu_y_x_prime_IS, std_y_x_prime_IS = importance_sampling(py_x_fn, cov_all, states, g_states, cov_test)
            IS_mean_array = jnp.append(IS_mean_array, mu_y_x_prime_IS)
            IS_std_array = jnp.append(IS_std_array, std_y_x_prime_IS)

        cbq_mean_dict[f"{n_alpha}"] = cbq_mean_array
        cbq_std_dict[f"{n_alpha}"] = cbq_std_array
        poly_mean_dict[f"{n_alpha}"] = poly_mean_array
        poly_std_dict[f"{n_alpha}"] = poly_std_array
        IS_mean_dict[f"{n_alpha}"] = IS_mean_array
        IS_std_dict[f"{n_alpha}"] = IS_std_array

    MC_list = []
    for Ny in N_beta_list:
        rng_key, _ = jax.random.split(rng_key)
        MC_list.append(g(states_test[:Ny, :]).mean())
    sensitivity_utils.save_final_results(args, MC_list, cbq_mean_dict, cbq_std_dict, poly_mean_dict,
                                         IS_mean_dict, g_test_true, N_alpha_list, N_beta_list)
    return


def get_config():
    parser = argparse.ArgumentParser(description='Conditional Bayesian Quadrature for Bayesian sensitivity analysis')

    # Args settings
    parser.add_argument('--dim', type=int)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./data')
    args = parser.parse_args()
    return args


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/sensitivity/'
    args.save_path += f"seed_{args.seed}__dim_{args.dim}"
    os.makedirs(args.save_path, exist_ok=True)
    return args


if __name__ == '__main__':
    args = get_config()
    create_dir(args)
    print(f'Device is {jax.devices()}')
    main(args)
    save_path = args.save_path
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    os.rename(save_path, f"{save_path}__complete")
    print("\n------------------- DONE -------------------\n")
