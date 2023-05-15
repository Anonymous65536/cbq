import jax
import jax.numpy as jnp
from kernels import *
import time
from jax.config import config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pwd

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()


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


def g3(y):
    return (y ** 2).sum(1)


def g3_ground_truth(mu, Sigma):
    return jnp.diag(Sigma).sum() + mu.T @ mu


@jax.jit
def Bayesian_Monte_Carlo(y, gy, mu_y_x, sigma_y_x):
    """
    :param sigma_y_x: covariance
    :param mu_y_x: mean
    :param y: (N, D)
    :param gy: (N, )
    :return:
    """
    N, D = y.shape[0], y.shape[1]
    eps = 1e-6

    A = 1.
    l = 1.
    K = A * my_RBF(y, y, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = A * kme_RBF_Gaussian(mu_y_x, sigma_y_x, l, y)
    varphi = A * kme_double_RBF_Gaussian(mu_y_x, sigma_y_x, l)

    BMC_mean = phi.T @ K_inv @ gy
    BMC_std = jnp.sqrt(varphi - phi.T @ K_inv @ phi)
    pause = True
    return BMC_mean, BMC_std


@jax.jit
def GP(rng_key, psi_y_x_mean, psi_y_x_std, X, X_prime, eps):
    """
    :param eps:
    :param psi_y_x_mean: (n_alpha, )
    :param psi_y_x_std: (n_alpha, )
    :param X: (n_alpha, D)
    :param X_prime: (N_test, D)
    :return:
    """
    n_alpha, D = X.shape[0], X.shape[1]
    l = 1.0
    K_no_scale = my_Matern(X, X, l)
    A = psi_y_x_mean.T @ K_no_scale @ psi_y_x_mean / n_alpha
    A = 1.0

    K_train_train = A * my_Matern(X, X, l) + eps * jnp.eye(n_alpha) + jnp.diag(psi_y_x_std ** 2)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = A * my_Matern(X_prime, X, l)
    K_test_test = A * my_Matern(X_prime, X_prime, l) + eps * jnp.eye(X_prime.shape[0])

    mu_y_x = K_test_train @ K_train_train_inv @ psi_y_x_mean
    var_y_x = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    var_y_x = jnp.abs(var_y_x)
    std_y_x = jnp.sqrt(var_y_x)
    pause = True
    return mu_y_x, std_y_x


def standardize(X):
    mean = jnp.mean(X, axis=0)
    std = jnp.std(X, axis=0)
    X = (X - mean) / std
    return X, mean, std


def Bayesian_Monte_Carlo_fn(tree):
    y, gy, mu_y_x, sigma_y_x = tree
    return Bayesian_Monte_Carlo(y, gy, mu_y_x, sigma_y_x)


def main():
    # Compare the performance of BQ and CBQ
    # seed = int(time.time())
    seed = 10
    rng_key = jax.random.PRNGKey(seed)
    D = 2

    prior_cov_base = 2.0
    noise = 1.0
    data_number = 5
    # X is (N, D-1), Y is (N, 1)
    X, Y = generate_data(rng_key, D, data_number, noise)

    g = g3
    g_ground_truth_fn = g3_ground_truth

    # n_theta_array = jnp.concatenate((jnp.array([3]), jnp.arange(10, 200, 10)))
    # n_theta_array = jnp.concatenate((jnp.array([3]), jnp.arange(10, 20, 10)))
    n_theta_array = jnp.concatenate((jnp.array([3]), jnp.arange(10, 20, 10), jnp.arange(190, 210, 10)))

    time_BQ_array = 0. * n_theta_array
    time_CBQ_array = 0. * n_theta_array
    rmse_BQ_array = 0. * n_theta_array
    rmse_CBQ_array = 0. * n_theta_array

    for j, n_theta in enumerate(tqdm(n_theta_array)):
        n_alpha = n_theta

        rng_key, _ = jax.random.split(rng_key)
        # This is X, size n_alpha * D
        alpha_all = jax.random.uniform(rng_key, shape=(n_alpha, D), minval=-1.0, maxval=1.0)
        # This is Y, size n_alpha * n_theta * D
        samples_all = jnp.zeros([n_alpha, n_theta, D])
        # This is g(Y), size n_alpha * sample_size
        g_samples_all = jnp.zeros([n_alpha, n_theta])
        prior_cov = jnp.array([[prior_cov_base] * D]) + alpha_all
        mu_y_x_all, var_y_x_all = posterior_full(X, Y, prior_cov, noise)

        # alpha_test = alpha_all[0, :][None, :]
        # mu_y_x_test, var_y_x_test = mu_y_x_all[0, :], var_y_x_all[0, :, :]

        n_test = 50
        rng_key, _ = jax.random.split(rng_key)
        # alpha_test = jax.random.uniform(rng_key, shape=(n_test, D), minval=-1.0, maxval=1.0)
        alpha_test = jnp.linspace(-1.0, 1.0, n_test)[:, None] * jnp.ones([n_test, D])
        prior_cov_test = jnp.array([[prior_cov_base] * D]) + alpha_test
        mu_y_x_test, var_y_x_test = posterior_full(X, Y, prior_cov_test, noise)

        for i in range(n_alpha):
            rng_key, _ = jax.random.split(rng_key)
            samples = jax.random.multivariate_normal(rng_key, mean=mu_y_x_all[i, :], cov=var_y_x_all[i, :, :],
                                                     shape=(n_theta,))
            samples_all = samples_all.at[i, :, :].set(samples)
            g_samples_all = g_samples_all.at[i, :].set(g(samples))

        samples_all_BQ = samples_all.reshape([n_alpha * n_theta, D])
        rng_key, _ = jax.random.split(rng_key)
        permutation = jax.random.permutation(rng_key, n_alpha * n_theta)
        samples_all_BQ = samples_all_BQ[permutation, :]
        g_samples_all_BQ = g_samples_all.reshape([n_alpha * n_theta])
        g_samples_all_BQ = g_samples_all_BQ[permutation]

        # ==================== Ground Truth ====================
        ground_truth = jnp.zeros(n_test) * 0.0
        for i in range(n_test):
            ground_truth = ground_truth.at[i].set(g_ground_truth_fn(mu_y_x_test[i, :], var_y_x_test[i, :, :]))

        # ==================== BQ ====================
        BQ_mean_array = jnp.zeros(n_test) * 0.0
        BQ_std_array = jnp.zeros(n_test) * 0.0
        if n_theta <= 70:
            _, _ = Bayesian_Monte_Carlo(samples_all_BQ, g_samples_all_BQ,
                                        mu_y_x_test[0, :], var_y_x_test[0, :, :])
            t0 = time.time()
            for t in range(n_test):
                BQ_mean, BQ_std = Bayesian_Monte_Carlo(samples_all_BQ, g_samples_all_BQ,
                                                       mu_y_x_test[t, :], var_y_x_test[t, :, :])
                BQ_mean_array = BQ_mean_array.at[t].set(BQ_mean)
                BQ_std_array = BQ_std_array.at[t].set(BQ_std)
            BQ_time = time.time() - t0
            BQ_rmse = jnp.sqrt(((BQ_mean_array - ground_truth) ** 2).mean())
        else:
            BQ_rmse = None
            BQ_time = None
        rmse_BQ_array = rmse_BQ_array.at[j].set(BQ_rmse)
        time_BQ_array = time_BQ_array.at[j].set(BQ_time)

        # ==================== CBQ ====================
        rng_key, _ = jax.random.split(rng_key)
        psi_mean_array = jnp.zeros(n_alpha)
        psi_std_array = jnp.zeros(n_alpha)

        for i in range(n_alpha):
            samples_i = samples_all[i, :, :]
            g_samples_i = g_samples_all[i, :]
            mu_y_x_i = mu_y_x_all[i, :]
            var_y_x_i = var_y_x_all[i, :, :]

            t0 = time.time()
            psi_mean, psi_std = Bayesian_Monte_Carlo(samples_i, g_samples_i, mu_y_x_i, var_y_x_i)
            t1 = time.time()

            psi_mean_array = psi_mean_array.at[i].set(psi_mean)
            psi_std_array = psi_std_array.at[i].set(psi_std if not jnp.isnan(psi_std) else 0.01)

        _, _ = GP(rng_key, psi_mean_array, psi_std_array, alpha_all, alpha_test,
                  eps=psi_std_array.mean())

        t2 = time.time()
        CBQ_mean, CBQ_std = GP(rng_key, psi_mean_array, psi_std_array, alpha_all, alpha_test,
                               eps=psi_std_array.mean())
        t3 = time.time()
        CBQ_time = (t1 - t0) * n_alpha + (t3 - t2)

        CBQ_rmse = jnp.sqrt(((CBQ_mean - ground_truth) ** 2).mean())
        rmse_CBQ_array = rmse_CBQ_array.at[j].set(CBQ_rmse)
        time_CBQ_array = time_CBQ_array.at[j].set(CBQ_time)

        jnp.save(f"./ablations/BQ_CBQ/CBQ_mean_N_{n_alpha}_{seed}.npy", CBQ_mean)
        jnp.save(f"./ablations/BQ_CBQ/CBQ_std_N_{n_alpha}_{seed}.npy", CBQ_std)
        jnp.save(f"./ablations/BQ_CBQ/BQ_mean_N_{n_alpha}_{seed}.npy", BQ_mean_array)
        jnp.save(f"./ablations/BQ_CBQ/BQ_std_N_{n_alpha}_{seed}.npy", BQ_std_array)

        # ============= Debug code =============
        # print("=====================================")
        # print("BQ RMSE: ", BQ_rmse)
        # print("CBQ RMSE: ", CBQ_rmse)
        #
        # print("BQ time: ", BQ_time)
        # print("CBQ time: ", CBQ_time)
        # print("=====================================")
        pause = True
        # ============= Debug code =============

    # ==================== Plot ====================

    jnp.save(f"./ablations/BQ_CBQ/rmse_BQ_array_{seed}.npy", rmse_BQ_array)
    jnp.save(f"./ablations/BQ_CBQ/rmse_CBQ_array_{seed}.npy", rmse_CBQ_array)
    jnp.save(f"./ablations/BQ_CBQ/time_BQ_array_{seed}.npy", time_BQ_array)
    jnp.save(f"./ablations/BQ_CBQ/time_CBQ_array_{seed}.npy", time_CBQ_array)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(n_theta_array, rmse_BQ_array, label="BQ")
    axs[0].plot(n_theta_array, rmse_CBQ_array, label="CBQ")
    axs[0].set_xlabel("Sample Numbers")
    axs[0].set_ylabel("RMSE")
    axs[0].legend()
    axs[0].set_yscale("log")

    axs[1].plot(n_theta_array, time_BQ_array, label="BQ")
    axs[1].plot(n_theta_array, time_CBQ_array, label="CBQ")
    axs[1].set_xlabel("Sample Numbers")
    axs[1].set_ylabel("Time")
    axs[1].legend()
    axs[1].set_yscale("log")
    plt.savefig(f"./ablations/BQ_CBQ/ablation_{seed}.png")
    plt.show()
    pause = True


if __name__ == "__main__":
    os.makedirs(f"./ablations/BQ_CBQ/", exist_ok=True)
    main()
