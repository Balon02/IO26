import jax
import jax.numpy as jnp

# jako, że zrobiłem pierwsze zadanie używając jaxa i jako tako miało to sens, to zadanie też zrobiłem w jaxie
# tutaj nie ma to kompletnie żadnego sensu (moim zdaniem), bo algorytm dużo bardziej polega na ilości iteracji, niż złożoności tego co się w środku każdej dzieje
# uzysk z xla będzie minimalny (tak przypuszczam), ale jak już sobie postanowiłem, że robię wszystko w jaxie, to robię wszystko w jaxie
# sporym utrudnieniem jest fakt, że w funkcjach przepuszczonych przez xla nie można używać dynamicznie krojonych tablic (kształty muszą być statyczne)
# wynikają z tego takie wynaturzenia, jakie można zobaczyć w moich operatorach zmiany trasy (nadużywanie funkcji roll i where)
# tak czy siak działa :)

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
    distance = j - i
    left = route.shape[0] - distance
    i_aligned = jnp.roll(route, -i)
    i_reversed = i_aligned[::-1]
    i_reversed_aligned = jnp.roll(i_reversed, -left)
    apply_mask = jnp.arange(route.shape[0]) < distance
    return jnp.where(apply_mask, i_reversed_aligned, i_aligned) # nie odkręcam, bo to nie ma znaczenia dla tsp; to jest cykliczne tak czy siak

@jax.jit()
def insert_relocate(key, route: jnp.ndarray):
    i, j = jax.random.choice(key, route.shape[0], shape = (2,), replace=False)
    iless_roll = jnp.roll(route, -i)[1:]
    distance = (j - i - 1) % (route.shape[0] - 1)
    j_aligned_roll = jnp.roll(iless_roll, distance)
    return jnp.concatenate([route[i][None], j_aligned_roll]) # tak samo, pomijam odkręcenie do oryginalnego porządku, bo nie ma to znaczenia

@jax.jit()
def block_move(key, route: jnp.ndarray):
    key_len, key_start, key_jump = jax.random.split(key, 3)
    length = jax.random.randint(key_len, (), minval=1, maxval=route.shape[0] - 1) # ruch o 1 byłby równoznaczny z brakiem ruchu
    start = jax.random.randint(key_start, (), minval=0, maxval=route.shape[0] - length + 1)
    left = route.shape[0] - length
    jump = jax.random.randint(key_jump, (), minval = 1, maxval = left)

    # pomysł jest taki, żeby przekręcić do postaci która zaczyna się od segmentu wybranego do zamiany, potem podzielić to co po nim zostaje na 2 wg. jump i zamienić te 2 końcowe części
    base_idx_map = jnp.arange(route.shape[0])
    result_idx_map = jnp.roll(base_idx_map, -start) # sekwencja zaczyna się od wybranego bloku do zamiany
    
    rest_len = route.shape[0] - length
    rest_pos = base_idx_map - length
    shifted_rest_pos = (rest_pos + jump) % rest_len
    source_pos = length + shifted_rest_pos

    source_pos = jnp.where(base_idx_map < length, base_idx_map, source_pos)

    return route[result_idx_map[source_pos]]

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
        "temperature": initial_temperature
    }

    operators = [swap, opt_reversal, insert_relocate, block_move]

    @jax.jit()
    def step(state):
        key = state["key"]
        route = state["route"]
        cost = state["cost"]
        temperature = state["temperature"]

        key, operator_switch_key, operator_key, accept_key = jnp.split(key, 4)
        
        # wybór operatora
        operator_id = jax.random.randint(operator_switch_key, (), minval=0, maxval=len(operators))
        operator = jax.lax.switch(operator_id, operators)

        # zastosowanie operatora
        candidate_route = operator(operator_key, route)

        # ocena
        candidate_cost = cost(candidate_route)
        cost_delta = candidate_cost - cost

        # wybór sekwencji
        always_accept = cost_delta <= 0
        random_accept = jax.random.uniform(accept_key, ()) < jnp.exp(-cost_delta / temperature)
        accept = always_accept | random_accept
        route = jax.lax.cond(accept, candidate_route, route)
        cost = jax.lax.cond(accept, candidate_cost, cost)

        # zmniejszenie temperatury
        temperature = jnp.min(temperature * cooling_rate, min_temperature)

        state = {
            "key": key,
            "route": route,
            "cost": cost,
            "temperature": temperature
        }

        return state
    
    results = [state]
    for i in range(iterations):
        state = simulated_annealing(state)
        results.append(state)
        print(f'{state} @ {i} iter')
