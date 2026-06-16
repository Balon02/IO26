import jax
import jax.numpy as jnp


# https://www.sfu.ca/~ssurjano/Code/ackleym.html
def ackley(xs: jnp.ndarray, a = 20, b = 0.2, c = jnp.pi * 2):
    sum1 = jnp.sum(xs ** 2)
    sum2 = jnp.sum(jnp.cos(c) * xs)
    term1 = -a * jnp.exp(-b * jnp.sqrt(sum1 / xs.shape[0]))
    term2 = - jnp.exp(sum2 / xs.shape[0])
    return term1 + term2 + a + jnp.exp(1)