import jax
import jax.numpy as jnp

from typing import Callable, Tuple

def differential_evolution(
        objective: Callable,
        domain: Tuple[float, float],
        n_dim: int = 2,
        population_size: int = 100,
        generations: int = 3000,
        mutation_factor: float = 0.5,
        crossover_prob: float = 0.9,
        seed: int = 123
    ):
    init_key = jax.random.key(seed)
    init_population = jax.random.uniform(init_key, (population_size, n_dim), minval = domain[0], maxval = domain[1])
    init_fitness = jax.vmap(objective)(init_population)
    init_best = jnp.argmin(init_fitness) # mniej = lepiej
    init_worst = jnp.argmax(init_fitness)
    init_mean = jnp.mean(init_fitness)

    state = {
        "key": init_key,
        "population": init_population,
        "fitness": init_fitness,
        "best": init_best,
        "worst": init_worst,
        "mean": init_mean
    }

    @jax.jit
    def step(state):
        key = state["key"]
        population = state["population"]
        fitness = state["fitness"]
        key, sampling_key, crossover_key, forced_muation_key = jax.random.split(key, 4)

        key_map = jax.random.split(sampling_key, population_size)
        index_map = jnp.arange(population_size)
        
        @jax.vmap
        def get_sampling_pools(key, idx):
            # pool = jax.random.permutation(key, jnp.delete(jnp.arange(population_size), idx))[:3] # pierwsza wersja, logicznie spójna, ale niepotrzebnie skomplikowana dla xla (jax.jit)
            scores = jax.random.uniform(key, (population_size,))
            scores = scores.at[idx].set(jnp.inf)
            pool = jnp.argsort(scores)[:3] # rozwiązanie v2, dziwne ale w teorii bardziej przyjazna dla xla
            return population[pool]
        
        sampling_pools = get_sampling_pools(key_map, index_map)
        mutants = sampling_pools[:, 0] + mutation_factor * (sampling_pools[:, 1] - sampling_pools[:, 2]) # a + F * (b - c)
        
        # selekcja
        crossover_mask = jnp.less(jax.random.uniform(crossover_key, (population_size, n_dim)), crossover_prob) # gdzie wybrać cechę po mutacji / oryginał
        ensure_mutation_mask = jnp.greater(jnp.sum(crossover_mask, axis=1), 0) # część populacji u której żadne mutacje nie przeszły
        forced_mutation_idxs = jax.random.randint(forced_muation_key, (population_size,), minval = 0, maxval = n_dim) # wylosowanie po 1 indeksie do mutacji 'na wszelki wypadek' dla każdego osobnika
        forced_mutation_mask = jnp.zeros_like(crossover_mask).at[jnp.arange(population_size), forced_mutation_idxs].set(1) # pełna maska tych 'na wszelkich wypadków'
        crossover_mask = jnp.where(ensure_mutation_mask, crossover_mask, forced_mutation_mask) # merge masek z wymuszonym wyborem tylko tam, gdzie jest potrzebny
            # chyba to trochę przekombinowałem, ale w ten sposób odsetek wyboru cech mutant vs oryginał będzie bliższy crossover_prob niż gdybyśmy 'na chama' nałożyli i jedną i drugą maskę

        trials = jnp.where(crossover_mask, mutants, population) # nałożenie maski
        trial_fitness = jax.vmap(objective)(trials)
        improved = jnp.less_equal(trial_fitness, fitness)

        # ocena
        population = jnp.where(improved[:, None], trials, population)
        fitness = jnp.where(improved, trial_fitness, fitness)
        best = jnp.argmin(fitness)
        worst = jnp.argmax(fitness)
        mean = jnp.mean(fitness)

        state = {
            "key": key,
            "population": population,
            "fitness": fitness,
            "best": best,
            "worst": worst,
            "mean": mean
        }

        return state
    
    results = [state]

    for generation in range(generations):
        state = step(state)
        results.append(state)
        print(f'POKOLENIE{generation}')
        print(f'NAJLEPSZY:{state["fitness"][state["best"]]} ({state["population"][state["best"]]})')
        print(f'NAJGORSZY:{state["fitness"][state["worst"]]} ({state["population"][state["worst"]]})')
        print(f'ŚREDNIA:{state["mean"]:.2f}')

    return results