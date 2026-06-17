import jax
import jax.numpy as jnp

# funkcje tłumaczone z matlabowych implementacji, przeanalizowane i zwektoryzowane gdzie się dało, przygotowane do akceleracji

# https://www.sfu.ca/~ssurjano/Code/ackleym.html
@jax.jit(static_argnames = ['a', 'b', 'c'])
def ackley(xs: jnp.ndarray, a = 20, b = 0.2, c = jnp.pi * 2):
    sum1 = jnp.sum(xs ** 2)
    sum2 = jnp.sum(jnp.cos(c * xs))
    term1 = -a * jnp.exp(-b * jnp.sqrt(sum1 / xs.shape[0]))
    term2 = - jnp.exp(sum2 / xs.shape[0])
    return term1 + term2 + a + jnp.e

ackley_domain = (-32.768, 32.768)

# https://www.sfu.ca/~ssurjano/Code/michalm.html
@jax.jit(static_argnames = 'm')
def michalewicz(xs: jnp.ndarray, m = 10):
    return -jnp.sum(jnp.sin(xs) * (jnp.sin(jnp.arange(1, xs.shape[0] + 1) * xs ** 2 / jnp.pi)) ** (2 * m)) # arange zastępuje tutaj zależność od pozycji w pętli z oryginalnej implementacji; gdyby zużycie pamięci było krytyczne można by tutaj użyć jax.lax.scan i lecieć w pętli jak w oryginale, żeby uniknąć materializacji tego sztucznego wektora indeksów w pamięci

michalewicz_domain = (0.0, jnp.pi)

# https://www.sfu.ca/~ssurjano/Code/rastrm.html
@jax.jit()
def rastrigin(xs: jnp.ndarray):
    return 10 * xs.shape[0] + jnp.sum(xs ** 2 - 10 * jnp.cos(2 * jnp.pi * xs))

rastrigin_domain = (-5.12, 5.12)

# https://www.sfu.ca/~ssurjano/langer.html
@jax.jit()
def langermann( # bardzo ciekawy kształt w 2D dla podanych stałych; niestety losowe dobieranie tych stałych dla > 2 wymiarów skutkuje nierozwiązywalnym problemem
        xs: jnp.ndarray, 
        # pomijam tutaj argument m = 5 z oryginału, bo wydaje mi się, że służy tylko sprawdzeniu kształtu wektora c oraz pierwszego wymiaru macierzy A; zakładam, że w matlabowej wersji była to kwestia implementacyjna związana ze specyfiką języka, a nie istotna część funkcji
        c = jnp.array([1, 2, 5, 2, 3]),
        A = jnp.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
    ):
    # pomijam sprawdzenie zgodności kształtu, jax ma swoje tracebacki od tego
    inner = jnp.sum((xs[None, :] - A) ** 2, axis=1) # tutaj działa auto broadcast, analogicznie do numpy
    return jnp.sum(c * jnp.exp(-inner / jnp.pi) * jnp.cos(jnp.pi * inner)) # outer

langermann_domain = (0.0, 10.0)

# https://www.sfu.ca/~ssurjano/Code/griewankm.html
@jax.jit()
def griewank(xs: jnp.ndarray):
    sum = jnp.sum(xs ** 2 / 4000)
    prod = jnp.prod(jnp.cos(xs / jnp.sqrt(jnp.arange(1, xs.shape[0] + 1)))) # ta sama akcja z indeksami co w funkcji Michalewicza w 18 linii kodu
    return sum - prod + 1

griewank_domain = (-600.0, 600.0)

# https://www.sfu.ca/~ssurjano/Code/rosenm.html
@jax.jit()
def rosenbrock(xs: jnp.ndarray):
    return jnp.sum(100 * (xs[1:] - xs[:-1] ** 2) ** 2 + (xs[:-1] - 1) ** 2)

rosenbrock_domain = (-5.0, 10.0)
