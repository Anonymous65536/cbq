import jax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp


@partial(jax.jit, static_argnums=(2, 3, 4))
def leapfrog(q, p, dVdq, path_len, step_size):
    """Leapfrog integrator for Hamiltonian Monte Carlo.
    Parameters
    ----------
    q : np.floatX
        Initial position
    p : np.floatX
        Initial momentum
    dVdq : callable
        Gradient of the velocity
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be
    Returns
    -------
    q, p : np.floatX, np.floatX
        New position and momentum
    """
    q, p = jnp.copy(q), jnp.copy(p)

    p -= step_size * dVdq(q)[0] / 2  # half step
    for _ in range(int(path_len / step_size) - 1):
        q += step_size * p  # whole step
        p -= step_size * dVdq(q)[0]  # whole step
    q += step_size * p  # whole step
    p -= step_size * dVdq(q)[0] / 2  # half step

    # momentum flip at end
    return q, -p


def HMC(rng_key, n_samples, negative_log_prob, grad_neg_log_prob, initial_position, path_len, step_size):
    """Run Hamiltonian Monte Carlo sampling.
    Parameters
    ----------
    n_samples : int
        Number of samples to return
    negative_log_prob : callable
        The negative log probability to sample from
    initial_position : np.array
        A place to start sampling from.
    path_len : float
        How long each integration path is. Smaller is faster and more correlated.
    step_size : float
        How long each integration step is. Smaller is slower and more accurate.

    Returns
    -------
    np.array
        Array of length `n_samples`.
    """
    # autograd magic

    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = jax.random.normal(rng_key, shape=(n_samples,) + initial_position.shape[:1])

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    for p0 in tqdm(momentum):
        p0 = p0[:, None]
        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog(
            samples[-1],
            p0,
            grad_neg_log_prob,
            path_len=path_len,
            step_size=step_size,
        )

        # Check Metropolis acceptance criterion
        start_log_p = negative_log_prob(samples[-1]) + 0.5 * jnp.sum((p0 * p0).sum())
        new_log_p = negative_log_prob(q_new) + 0.5 * jnp.sum((p_new * p_new).sum())
        rng_key, _ = jax.random.split(rng_key)
        if jnp.log(jax.random.uniform(rng_key)) < start_log_p - new_log_p:
            samples.append(q_new)
        else:
            samples.append(jnp.copy(samples[-1]))

    return jnp.array(samples[1:])


def main():
    seed = 0
    rng_key = jax.random.PRNGKey(seed)

    num = 30
    rng_key, _ = jax.random.split(rng_key)
    x = jax.random.uniform(rng_key, shape=(num, 2), minval=-1.0, maxval=1.0)
    rng_key, _ = jax.random.split(rng_key)
    w = jax.random.normal(rng_key, shape=(2, 1))
    rng_key, _ = jax.random.split(rng_key)
    epsilon = jax.random.normal(rng_key, shape=(num, 1))
    y = x @ w + epsilon

    def neg_log_posterior(w, x, y):
        log_prior_w = jax.scipy.stats.norm.logpdf(w, 0., 1.).sum()
        log_likelihood = jax.scipy.stats.norm.logpdf(y - x @ w, 0., 1.0).sum()
        return -(log_prior_w + log_likelihood).squeeze()

    true_post_var = jnp.linalg.inv(jnp.eye(2) + 10 * (x.T @ x))
    true_post_mean = 10 * true_post_var @ x.T @ y

    negative_log_prob = partial(neg_log_posterior, x=x, y=y)
    grad_neg_log_prob = jax.grad(negative_log_prob, argnums=(0,))
    initial_position = jnp.array([[0.0, 0.0]]).T
    n_samples = 500
    post_samples = HMC(rng_key, n_samples, negative_log_prob, grad_neg_log_prob,
                       initial_position, path_len=1, step_size=0.01)
    post_samples = post_samples.squeeze()

    N = 50
    w1 = jnp.linspace(-2, 2, N)
    w2 = jnp.linspace(-2, 2, N)
    W1, W2 = jnp.meshgrid(w1, w2)
    W = jnp.concatenate((W1[:, :, None], W2[:, :, None]), axis=2)
    W = W.reshape(N * N, 2)
    coe = (2 * jnp.pi * jnp.linalg.det(true_post_var)) ** (-2 / 2)
    Z_prior = jnp.zeros((N * N, 1))
    Z_posterior = jnp.zeros((N * N, 1))
    for i in tqdm(range(N * N)):
        W_i = W[i, :].T[:, None]
        temp = coe * jnp.exp(-0.5 * (W_i - true_post_mean).T @ jnp.linalg.inv(true_post_var) @ (W_i - true_post_mean))
        Z_posterior = Z_posterior.at[i, :].set(temp.squeeze())
        temp = (2 * jnp.pi) ** (-1) * W_i.T @ W_i
        Z_prior = Z_prior.at[i, :].set(temp.squeeze())
    Z_posterior = Z_posterior.reshape(N, N)
    Z_prior = Z_prior.reshape(N, N)

    plt.figure()
    plt.contour(W1, W2, Z_posterior)
    plt.contour(W1, W2, Z_prior)
    plt.scatter(post_samples[50:, 0], post_samples[50:, 1])
    plt.show()

    def log_posterior(w, x, y):
        log_prior_w = jax.scipy.stats.norm.logpdf(w, 0., 1.).sum()
        log_likelihood = jax.scipy.stats.norm.logpdf(y - x @ w, 0., 1.0).sum()
        return (log_prior_w + log_likelihood).squeeze()
    log_prob = partial(log_posterior, x=x, y=y)

    rng_key, _ = jax.random.split(rng_key)
    init_params = jnp.array([[0.0, 0.0]]).T
    def run_chain(rng_key, state):
        kernel = tfp.mcmc.NoUTurnSampler(log_prob, 1e-3)
        return tfp.mcmc.sample_chain(500,
                                     current_state=state,
                                     kernel=kernel,
                                     seed=rng_key)
    states, _ = run_chain(rng_key, init_params)

    plt.figure()
    plt.contour(W1, W2, Z_posterior)
    plt.contour(W1, W2, Z_prior)
    plt.scatter(post_samples[50:, 0], post_samples[50:, 1])
    plt.scatter(states[50:, 0, 0], states[50:, 1, 0])
    plt.show()


if __name__ == '__main__':
    main()
