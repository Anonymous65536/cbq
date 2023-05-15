import matplotlib.pyplot as plt
import math
import sklearn
from tqdm import tqdm
import torch
import jax.numpy as jnp
import jax
import optax
import time
from functools import partial
from sklearn.gaussian_process.kernels import Matern

from IPython.display import set_matplotlib_formats

set_matplotlib_formats("pdf", "png")
# plt.tight_layout()
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
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath, amssymb}']


# @jax.jit
def jax_dist(x, y):
    return jnp.abs(x - y).squeeze()


distance = jax.vmap(jax_dist, in_axes=(None, 1), out_axes=1)

sign_func = jax.vmap(jnp.greater, in_axes=(None, 1), out_axes=1)


# @jax.jit
def my_Matern(x, y, l):
    r = distance(x, y).squeeze()
    part1 = 1 + math.sqrt(3) * r / l
    part2 = jnp.exp(-math.sqrt(3) * r / l)
    return part1 * part2


# @jax.jit
def dx_Matern(x, y, l):
    sign = sign_func(x, y).squeeze().astype(float) * 2 - 1
    r = distance(x, y).squeeze()
    part1 = jnp.exp(-math.sqrt(3) / l * r) * (math.sqrt(3) / l * sign)
    part2 = (-math.sqrt(3) / l * sign) * jnp.exp(-math.sqrt(3) / l * r) * (1 + math.sqrt(3) / l * r)
    return part1 + part2


# @jax.jit
def dy_Matern(x, y, l):
    sign = -(sign_func(x, y).squeeze().astype(float) * 2 - 1)
    r = distance(x, y).squeeze()
    part1 = jnp.exp(-math.sqrt(3) / l * r) * (math.sqrt(3) / l * sign)
    part2 = (-math.sqrt(3) / l * sign) * jnp.exp(-math.sqrt(3) / l * r) * (1 + math.sqrt(3) / l * r)
    return part1 + part2


### Important! Be caution about the gradient here, the derivative of the absolute value is taken to be 0 at 0.
# @jax.jit
def dxdy_Matern(x, y, l):
    r = distance(x, y).squeeze()
    const = math.sqrt(3) / l
    part1 = const * const * jnp.exp(-const * r)
    part2 = -const * const * jnp.exp(-const * r) * (1 + const * r)
    part3 = const * jnp.exp(-const * r) * const
    return part1 + part2 + part3


# @jax.jit
def stein_kernel(x, y, l):
    d_log_px = -x
    d_log_py = -y

    K = my_Matern(x, y, l)
    dx_K = dx_Matern(x, y, l)
    dy_K = dy_Matern(x, y, l)
    dxdy_K = dxdy_Matern(x, y, l)
    part1 = d_log_px @ d_log_py.T * K
    part2 = d_log_py.T * dx_K
    part3 = d_log_px * dy_K
    part4 = dxdy_K
    return part1 + part2 + part3 + part4


def GP():
    seed = int(time.time())
    rng_key = jax.random.PRNGKey(seed)
    C = 5
    l = 2.0
    i = 10
    eps = 1e-6
    x_train = jax.random.normal(rng_key, shape=(i,))[:, None]
    fx_train = jnp.exp(jnp.sin(C * 2 * x_train) ** 2 - (2 * x_train) ** 2)

    x_test = jnp.linspace(-3, 3, 100)[:, None]
    fx_test = jnp.exp(jnp.sin(C * 2 * x_test) ** 2 - (2 * x_test) ** 2)
    K_x_test_x_train = stein_kernel(x_test, x_train, l)
    K_x_train_x_train = stein_kernel(x_train, x_train, l) + eps * jnp.eye(i)

    gp_mean = K_x_test_x_train @ jnp.linalg.inv(K_x_train_x_train) @ fx_train

    plt.figure()
    plt.scatter(x_train, fx_train)
    plt.plot(x_test, fx_test)
    plt.plot(x_test, gp_mean, color='r')
    plt.show()
    return

@jax.jit
def f(x):
    return 2 * jnp.exp(2 * jnp.sin(10 * x) ** 2 - 4 * x ** 2)


def main():
    seed = int(time.time())
    # seed = 0
    rng_key = jax.random.PRNGKey(seed)
    # n_list = [200]
    n_list = jnp.concatenate((jnp.arange(3, 9), jnp.arange(1, 20) * 10))
    # n_list = jnp.array([200])
    eps = 1e-6

    x = jnp.linspace(-5, 5, 100)
    fx = f(x)
    plt.figure()
    plt.plot(x, fx)
    plt.title("This is the function that we want to integrate.")
    plt.show()

    l_list = jnp.array([])
    c_list = jnp.array([])
    MC_list = jnp.array([])
    BMC_mean_list = jnp.array([])
    BMC_std_list = jnp.array([])

    x = jax.random.normal(rng_key, shape=(100000,))
    fx = f(x)
    true_value = fx.mean()

    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)

    for n in tqdm(n_list):
        x = jax.random.normal(rng_key, shape=(n,))[:, None]
        fx = f(x)
        MC_list = jnp.append(MC_list, fx.mean())

        c_init = c = 1.0
        log_l_init = log_l = jnp.log(0.3)
        A_init = A = 1.0 / jnp.sqrt(n)
        opt_state = optimizer.init((log_l_init, c_init, A_init))

        @jax.jit
        def nllk_func(log_l, c, A):
            l = jnp.exp(log_l)
            n = x.shape[0]
            K = A * stein_kernel(x, x, l) + c
            K_inv = jnp.linalg.inv(K)
            nll = -(-0.5 * fx.T @ K_inv @ fx - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / n
            return nll[0][0]

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

        for _ in range(1000):
            rng_key, _ = jax.random.split(rng_key)
            log_l, c, A, opt_state, nllk_value = step(log_l, c, A, opt_state, rng_key)
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
        final_K = A * stein_kernel(x, x, l) + c
        final_K_inv = jnp.linalg.inv(final_K)
        BMC_mean = c * (final_K_inv @ fx).sum()
        BMC_std = jnp.sqrt(c * c - final_K_inv.sum() * c * c)
        BMC_mean_list = jnp.append(BMC_mean_list, BMC_mean)
        BMC_std_list = jnp.append(BMC_std_list, BMC_std)

        l_list = jnp.append(l_list, l)
        c_list = jnp.append(c_list, c)

    fig = plt.figure(figsize=(10, 5))
    ax_1, ax_2 = fig.subplots(1, 2)
    ax_1.plot(l_list)
    ax_1.set_title("l")
    ax_2.plot(c_list)
    ax_2.set_title("c")
    plt.show()

    fig = plt.figure(figsize=(20, 10))
    ax_1 = fig.subplots(1, 1)
    ax_1.plot(n_list, BMC_mean_list, label='BMC', color='blue')
    # ax_1.fill_between(jnp.array(n_list), BMC_mean_list - BMC_std_list,
    #                   BMC_mean_list + BMC_std_list, color='blue', alpha=0.2)
    ax_1.plot(n_list, MC_list, label='MC', color='orange')
    ax_1.axhline(y=true_value, linestyle='--', color='r')
    ax_1.legend()
    ax_1.set_xlabel("The number of observations.")
    ax_1.set_ylabel("Estimated Integral")
    ax_1.set_ylim([0., 3.])
    plt.show()

    pause = True
    return


if __name__ == '__main__':
    # GP()
    main()
