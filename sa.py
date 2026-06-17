import jax
import jax.numpy as jnp

# jako, że zrobiłem pierwsze zadanie używając jaxa i jako tako miało to sens, to zadanie też zrobiłem w jaxie
# tutaj nie ma to kompletnie żadnego sensu (moim zdaniem), bo algorytm dużo bardziej polega na ilości iteracji, niż złożoności tego co się w środku każdej dzieje
# uzysk z xla będzie minimalny (tak przypuszczam), ale jak już sobie postanowiłem, że robię wszystko w jaxie, to robię wszystko w jaxie
# 

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
    pair = jax.random.choice(key, route.shape[0], shape = (2,), replace=False)
    i = jnp.minimum(pair[0], pair[1])
    j = jnp.maximum(pair[0], pair[1])
    reversed_route = route[::-1]
    map = jnp.arange(route.shape[0])
    apply_mask = jnp.where(jnp.greater_equal(map, i) & jnp.less(map, j))
    return jnp.where(apply_mask, reversed_route, route)

@jax.jit()
def insert_relocate(key, route: jnp.ndarray):
    i, j = jax.random.randint(key, 2, minval = 0, maxval = route.shape[0])
    iless_roll = jnp.roll(route, -i)[1:]
    return jnp.concatenate([route[j][None], iless_roll, route[i][None]]) # wydaje mi się, że nie trzeba tego odkręcać, bo droga i tak zatacza koło; jnp.roll powinien być niezauważalny z punktu widzenia algorytmu

@jax.jit()
def block_move(key, route: jnp.ndarray):
    key_len, key_start, key_jump = jax.random.split(key, 3)
    length = jax.random.randint(key_len, (), minval=1, maxval=route.shape[0] - 1) # ruch o 1 byłby równoznaczny z brakiem ruchu
    start = jax.random.randint(key_start, (), minval=0, maxval=route.shape[0] - length + 1)
    left = route.shape[0] - length
    jump = jax.random.randint(key_jump, (), minval = 1, maxval = left)

    base_idx_map = jnp.arange(route.shape[0])

    block_aligned_idx_map = jnp.roll(jnp.arange(route.shape[0]), -start) # ciąg w postaci [wylosowany_blok], [reszta]
    base_mask = jnp.where(base_idx_map <= start, 1, 0)

    # resztę dzielimy na to co ma wylądować przed i po wg. wylosowanego skoku
    end_rolled_idx_map = jnp.roll(base_idx_map, route.shape[0] - length + jump) # ciąg w postaci [reszta], [segment który ma trafić przed wylosowany ciąg]
    prefix_mask = jnp.where(base_idx_map <= start)

    start_aligned_idx_map = jnp.roll(base_idx_map, -start) # stąd bierzemy indeksy bloku

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


