import jax
import jax.numpy as jnp
import time
from jax.scipy.stats import norm
import matplotlib.pyplot as plt


def Geometric_Brownian(n, dt, rng_key, sigma=0.3, S0=1):
    rng_key, _ = jax.random.split(rng_key)
    dWt = jax.random.normal(rng_key, shape=(int(n),)) * jnp.sqrt(dt)
    dlnSt = sigma * dWt - 0.5 * (sigma ** 2) * dt
    St = jnp.exp(jnp.cumsum(dlnSt) + jnp.log(S0))
    return St


def callBS(t, s, K, sigma):
    part_one = jnp.log(s / K) / (sigma * jnp.sqrt(t)) + 0.5 * sigma * jnp.sqrt(t)
    part_two = jnp.log(s / K) / (sigma * jnp.sqrt(t)) - 0.5 * sigma * jnp.sqrt(t)
    return s * norm.cdf(part_one) - K * norm.cdf(part_two)


def BSM_butterfly_analytic():
    seed = int(time.time())
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

    # St = Geometric_Brownian(100, t / 100, sigma=sigma, S0=S0)[-1]

    epsilon = jax.random.normal(rng_key)
    St = S0 * jnp.exp(sigma * jnp.sqrt(t) * epsilon - 0.5 * (sigma ** 2) * t)
    psiST_St_1 = callBS(T - t, St, K1, sigma) + callBS(T - t, St, K2, sigma) \
                 - 2 * callBS(T - t, St, (K1 + K2) / 2, sigma)
    psiST_St_2 = callBS(T - t, (1 + s) * St, K1, sigma) + callBS(T - t, (1 + s) * St, K2, sigma) \
                 - 2 * callBS(T - t, (1 + s) * St, (K1 + K2) / 2, sigma)
    L_inner = psiST_St_1 - psiST_St_2
    # print(max(L_inner, 0))
    print(L_inner)

    # Verifies the result from BSM agree with standard Monte Carlo
    # iter_num = 10000
    # L_MC = 0
    # for i in range(iter_num):
    #     epsilon = np.random.normal(0, 1)
    #     ST = St * np.exp(sigma * np.sqrt((T-t)) * epsilon - 0.5 * (sigma ** 2) * (T-t))
    #     psi_ST_1 = max(ST - K1, 0) + max(ST - K2, 0) - 2 * max(ST - (K1+K2) / 2, 0)
    #     psi_ST_2 = max((1+s) * ST - K1, 0) + max((1+s) * ST - K2, 0) - 2 * max((1+s) * ST - (K1+K2) / 2, 0)
    #     L_MC += (psi_ST_1 - psi_ST_2)
    # print(L_MC / iter_num)
    return


def price_visualize(St, N, rng_key, K1=50, K2=150, s=-0.2, sigma=0.3, T=2, t=1):
    """
    :param St: the price St at time t
    :return: The function returns the price ST at time T sampled from the conditional
    distribution p(ST|St), and the loss \psi(ST) - \psi((1+s)ST) due to the shock. Their shape is Nx * Ny
    """
    output_shape = (St.shape[0], N)
    rng_key, _ = jax.random.split(rng_key)
    epsilon = jax.random.normal(rng_key, shape=output_shape)
    ST = St * jnp.exp(sigma * jnp.sqrt((T - t)) * epsilon - 0.5 * (sigma ** 2) * (T - t))
    psi_ST_1 = jnp.maximum(ST - K1, 0) + jnp.maximum(ST - K2, 0) - 2 * jnp.maximum(ST - (K1 + K2) / 2, 0)
    psi_ST_2 = jnp.maximum((1 + s) * ST - K1, 0) + jnp.maximum((1 + s) * ST - K2, 0) - 2 * jnp.maximum(
        (1 + s) * ST - (K1 + K2) / 2, 0)

    loss_list = []
    ST_list = []
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs = axs.flatten()

    St_dummy = St[0]
    log_ST_min = jnp.log(St_dummy) + sigma * jnp.sqrt((T - t)) * (-3) - 0.5 * (sigma ** 2) * (T - t)
    log_ST_max = jnp.log(St_dummy) + sigma * jnp.sqrt((T - t)) * (+3) - 0.5 * (sigma ** 2) * (T - t)
    ST_min = jnp.exp(log_ST_min)
    ST_max = jnp.exp(log_ST_max)
    ST_samples = jnp.linspace(ST_min, ST_max, 100)
    normal_samples = (jnp.log(ST_samples) - jnp.log(St_dummy) + 0.5 * (sigma ** 2) * (T - t)) / sigma
    density = norm.pdf(normal_samples)
    axs[0].plot(ST_samples, density)
    axs[0].set_title(r"The pdf for $p(S_T|S_t)$")
    axs[0].set_xlabel(r"$S_T$")

    for _ in range(1000):
        rng_key, _ = jax.random.split(rng_key)
        epsilon = jax.random.normal(rng_key, shape=(1, 1))
        ST = St_dummy * jnp.exp(sigma * jnp.sqrt((T - t)) * epsilon - 0.5 * (sigma ** 2) * (T - t))
        psi_ST_1 = jnp.maximum(ST - K1, 0) + jnp.maximum(ST - K2, 0) - 2 * jnp.maximum(ST - (K1 + K2) / 2, 0)
        psi_ST_2 = jnp.maximum((1 + s) * ST - K1, 0) + jnp.maximum((1 + s) * ST - K2, 0) - 2 * np.maximum(
            (1 + s) * ST - (K1 + K2) / 2, 0)
        loss_list.append(psi_ST_1 - psi_ST_2)
        ST_list.append(ST)

    ST_dummy = jnp.array(ST_list).squeeze()
    loss_dummy = jnp.array(loss_list).squeeze()
    axs[1].scatter(ST_dummy, loss_dummy)
    axs[1].set_title(r"$\psi(S_T) - \psi((1+s)S_T)$")
    axs[1].set_xlabel(r"$S_T$")
    plt.suptitle(rf"$S_t$ is {St_dummy[0]}")
    plt.savefig(f"./results/finance_dummy.pdf")
    # plt.show()
    return ST, psi_ST_1 - psi_ST_2


def calibrate(ground_truth, BMC_mean, BMC_std):
    """
    Calibrate the BMC mean and std
    :param ground_truth: (N, )
    :param BMC_mean: (N, )
    :param BMC_std: (N, )
    :return:
    """
    BMC_std /= 10
    confidence_level = jnp.arange(0.0, 1.01, 0.05)
    prediction_interval = jnp.zeros(len(confidence_level))
    for i, c in enumerate(confidence_level):
        z_score = norm.ppf(1 - (1 - c) / 2)  # Two-tailed z-score for the given confidence level
        prob = jnp.less(jnp.abs(ground_truth - BMC_mean), z_score * BMC_std)
        prediction_interval = prediction_interval.at[i].set(prob.mean())

    plt.figure()
    plt.plot(confidence_level, prediction_interval, label="Model calibration", marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal calibration", color="black")
    plt.xlabel("Confidence")
    plt.ylabel("Coverage")
    plt.title("Calibration plot")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.close()
    # plt.show()
    return prediction_interval

def scale(Z):
    s = Z.std()
    standardized = Z / s
    return standardized, s


def standardize(Z):
    mean = Z.mean(0)
    std = Z.std(0)
    standardized = (Z - mean) / std
    return standardized, mean, std
