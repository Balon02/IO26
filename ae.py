import jax
import jax.numpy as jnp

from typing import Callable, Tuple

def evolutionary_algorithm(
        objective: Callable,
        domain: Tuple[float, float],
        n_dim: int = 2,
        population_size: int = 100,
        generations: int = 100,
        mutation_std: float = 0.05,
        parent_selection_pool_size: int = 5,
        seed: int = 123
    ):
    mutation_std = mutation_std * (domain[1] - domain[0])

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
        key, parent_key, pairing_key, crossover_key, mutation_key = jax.random.split(key, 5)

        # wybór rodziców
        parent_key_map = jax.random.split(parent_key, population_size)
        
        @jax.vmap
        def get_parent(key): # takie coś wymyśliłem, żeby nie było powtórek w puli kandydatów (chociaż pewnie z powtórkami też by działało, a kod byłby prostszy)
            pool = jax.random.permutation(key, population_size)[:parent_selection_pool_size]
            best_idx_pool =  jnp.argmin(fitness[pool])
            best_idx = pool[best_idx_pool]
            return population[best_idx]
        
        parents = get_parent(parent_key_map)
        
        # krzyżowanie
        pairs = jax.random.permutation(pairing_key, parents)
        alpha = jax.random.uniform(crossover_key, (population_size, 1))
        children = alpha * parents + (1.0 - alpha) * pairs
        
        # mutacja
        children = children + mutation_std * jax.random.normal(mutation_key, children.shape)
        children = jnp.clip(children, domain[0], domain[1])

        # selekcja
        children_fitness = jax.vmap(objective)(children)
        combined_population = jnp.concatenate([population, children])
        combined_fitness = jnp.concatenate([fitness, children_fitness])
        best_idxs = jnp.argsort(combined_fitness)[:population_size]

        population = combined_population[best_idxs]
        fitness = combined_fitness[best_idxs]
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
