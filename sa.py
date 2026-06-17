import jax
import jax.numpy as jnp

# jako, że zrobiłem pierwsze zadanie używając jaxa i jako tako miało to sens, to zadanie też zrobiłem w jaxie
# tutaj nie ma to kompletnie żadnego sensu (moim zdaniem), bo algorytm dużo bardziej polega na ilości iteracji, niż złożoności tego co się w środku każdej dzieje
# uzysk z xla będzie minimalny (tak przypuszczam), ale jak już sobie postanowiłem, że robię wszystko w jaxie, to robię wszystko w jaxie
# niech się Pan przygotuje na masę bezsensownych rozwiązań na maskach zamiast standardowego data slicingu, którego użylibyśmy w numpy, bo nie wymusza statycznych kształtów jak xla w funkcjach kompilowanych przez jaxa

@jax.jit()
def cost(dist: jnp.ndarray, route: jnp.ndarray):
    roll = jnp.roll(route, -1)
    return jnp.sum(dist[route, roll])

@jax.jit()
def swap(key, route: jnp.ndarray):
    i, j = jax.random.randint(key, 2, minval = 0, maxval = route.shape[0])
    i_val = route[i]
    route = route.at[i].set(route[j])
    route = route.at[j].set(i_val)
    return route

@jax.jit()
def opt_reversal(key, route: jnp.ndarray):
    i, j = jax.random.randint(key, 2, minval = 0, maxval = route.shape[0])
    idx = jnp.arange(route.shape[0])
    segment_idx = i + j - idx
    new_route = jnp.where(jnp.greater_equal(idx, i) & jnp.less_equal(idx, j), route[segment_idx], route) # pierwsza przekombinowana implementacja na maskach
    return new_route

@jax.jit()
def insert_relocate(key, route: jnp.ndarray):
    i, j = jax.random.randint(key, 2, minval = 0, maxval = route.shape[0])
    k = jnp.arange(route.shape[0])
    src = k
    src = src + (jnp.less(i, j) & jnp.greater_equal(k, i) & jnp.less(k, j))
    src = src - (jnp.greater(i, j) & jnp.greater(k, j) & jnp.less_equal(k, i))
    src = jnp.where(jnp.equal(k, j), i, src)
    return route[src], (i, j) # druga przekombinowana implementacja na maskach

@jax.jit()
def block_move(key, route: jnp.ndarray):
    # trzecia przekombinowana implementacja na maskach
    key, k_len, k_start, k_dest = jax.random.split(key, 4)
    length = jax.random.randint(k_len, (), 1, route.shape[0])
    start = jax.random.randint(k_start, (), 0, route.shape[0] - length + 1)

    

def simulated_annealing(
        dist: jnp.ndarray,
        iterations: int = 10000,
        initial_temperature: float = 1000.0,
        cooling_rate: float = 0.995,
        min_temperature: float = 1e-3,
        seed: int = 123
    ):
    key, route_init_key = jax.random.split(jax.random.key(seed), 2)

    init_route = jax.random.permutation(route_init_key, jnp.arange(dist.shape[0]))
    init_cost = cost(dist, init_route)

    state = {
        "key": key,
        "route": init_route,
        "cost": init_cost,
        "best_route": init_route,
        "best_cost": init_cost,
        "temperature": initial_temperature,
        "iteration": iterations,
    }

    @jax.jit()
    def step(state):
        key = state["key"]
        route = state["route"]
        cost = state["cost"]


