from __future__ import annotations

import argparse
import contextlib
import html
import inspect
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ae import evolutionary_algorithm
from de import differential_evolution
from ndim_fn import (
    ackley,
    ackley_domain,
    griewank,
    griewank_domain,
    langermann,
    langermann_domain,
    michalewicz,
    michalewicz_domain,
    rastrigin,
    rastrigin_domain,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Plotly is required to generate the report. Run `uv sync` after the "
        "pyproject.toml dependency update, then rerun `uv run python zadanie_1.py`."
    ) from exc


Objective = Callable[[jnp.ndarray], jnp.ndarray]


OBJECTIVES: dict[str, tuple[Objective, tuple[float, float]]] = {
    "ackley": (ackley, ackley_domain),
    "michalewicz": (michalewicz, michalewicz_domain),
    "rastrigin": (rastrigin, rastrigin_domain),
    "langermann": (langermann, langermann_domain),
    "griewank": (griewank, griewank_domain),
}

HIGH_DIM = 50
HIGH_DIM_GENERATION_MULTIPLIER = 3


def make_high_dim_objectives(seed: int) -> dict[str, tuple[Objective, tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    langermann_c = jnp.asarray(rng.uniform(1.0, 5.0, size=(5,)), dtype=jnp.float32)
    langermann_a = jnp.asarray(
        rng.uniform(langermann_domain[0], langermann_domain[1], size=(5, HIGH_DIM)),
        dtype=jnp.float32,
    )

    @jax.jit
    def langermann_50d(xs: jnp.ndarray) -> jnp.ndarray:
        return langermann(xs, c=langermann_c, A=langermann_a)

    return {
        "ackley": (ackley, ackley_domain),
        "michalewicz": (michalewicz, michalewicz_domain),
        "rastrigin": (rastrigin, rastrigin_domain),
        "langermann": (langermann_50d, langermann_domain),
        "griewank": (griewank, griewank_domain),
    }


def default_generations() -> int:
    ae_default = inspect.signature(evolutionary_algorithm).parameters["generations"].default
    de_default = inspect.signature(differential_evolution).parameters["generations"].default
    if ae_default != de_default:
        raise ValueError(
            "AE and DE have different default generation counts; choose an explicit "
            "50D generation count before building the report."
        )
    return int(ae_default)


def history_to_numpy(results: list[dict]) -> dict[str, np.ndarray]:
    population = np.stack([np.asarray(state["population"]) for state in results])
    fitness = np.stack([np.asarray(state["fitness"]) for state in results])
    best_idx = np.argmin(fitness, axis=1)
    worst_idx = np.argmax(fitness, axis=1)
    generation_idx = np.arange(len(results))

    return {
        "population": population,
        "fitness": fitness,
        "best_idx": best_idx,
        "worst_idx": worst_idx,
        "best": fitness[generation_idx, best_idx],
        "worst": fitness[generation_idx, worst_idx],
        "mean": np.mean(fitness, axis=1),
    }


def save_history(path: Path, history: dict[str, np.ndarray], domain: tuple[float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        population=history["population"],
        fitness=history["fitness"],
        best_idx=history["best_idx"],
        worst_idx=history["worst_idx"],
        best=history["best"],
        worst=history["worst"],
        mean=history["mean"],
        domain=np.asarray(domain, dtype=float),
    )


def load_history(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def evaluate_objective_grid(
    objective: Objective,
    domain: tuple[float, float],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low, high = float(domain[0]), float(domain[1])
    axis = np.linspace(low, high, grid_size)
    x_grid, y_grid = np.meshgrid(axis, axis)
    points = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)

    values = jax.vmap(objective)(jnp.asarray(points))
    z_grid = np.asarray(values).reshape(grid_size, grid_size)
    return x_grid, y_grid, z_grid


def make_metrics_figure(
    function_name: str,
    ae_history: dict[str, np.ndarray],
    de_history: dict[str, np.ndarray],
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Najlepsza wartość funkcji celu",
            "Najgorsza wartość funkcji celu",
            "Średnia wartość funkcji celu",
        ),
    )
    metrics = [
        ("best", "najlepsza"),
        ("worst", "najgorsza"),
        ("mean", "średnia"),
    ]

    for row, (metric_key, metric_name) in enumerate(metrics, start=1):
        fig.add_trace(
            go.Scatter(
                y=ae_history[metric_key],
                mode="lines",
                name=f"AE: {metric_name}",
                legendgroup="AE",
                line={"color": "#2563eb"},
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                y=de_history[metric_key],
                mode="lines",
                name=f"DE: {metric_name}",
                legendgroup="DE",
                line={"color": "#dc2626"},
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )

    fig.update_layout(
        title=f"{function_name}: porównanie przebiegu AE i DE",
        height=780,
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Pokolenie", row=3, col=1)
    fig.update_yaxes(title_text="Wartość funkcji celu", row=1, col=1)
    fig.update_yaxes(title_text="Wartość funkcji celu", row=2, col=1)
    fig.update_yaxes(title_text="Wartość funkcji celu", row=3, col=1)
    return fig


def frame_indices(n_generations: int, frame_stride: int) -> np.ndarray:
    indices = np.arange(0, n_generations, frame_stride)
    if indices[-1] != n_generations - 1:
        indices = np.append(indices, n_generations - 1)
    return indices


def make_3d_population_figure(
    function_name: str,
    algorithm_name: str,
    objective: Objective,
    domain: tuple[float, float],
    history: dict[str, np.ndarray],
    grid_size: int,
    frame_stride: int,
) -> go.Figure:
    x_grid, y_grid, z_grid = evaluate_objective_grid(objective, domain, grid_size)
    population = history["population"]
    fitness = history["fitness"]
    generations = frame_indices(population.shape[0], frame_stride)
    first_generation = int(generations[0])
    first_population = population[first_generation]
    first_fitness = fitness[first_generation]
    low, high = float(domain[0]), float(domain[1])

    surface = go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        surfacecolor=y_grid,
        colorscale="Viridis",
        cmin=low,
        cmax=high,
        opacity=0.72,
        colorbar={"title": "x2"},
        name="Powierzchnia funkcji celu",
        showscale=True,
    )
    population_trace = go.Scatter3d(
        x=first_population[:, 0],
        y=first_population[:, 1],
        z=first_fitness,
        mode="markers",
        marker={
            "size": 4,
            "color": first_population[:, 1],
            "colorscale": "Viridis",
            "cmin": low,
            "cmax": high,
            "line": {"color": "#111827", "width": 1},
        },
        name=f"Populacja {algorithm_name}",
        text=[f"wartość funkcji celu={value:.6g}" for value in first_fitness],
        hovertemplate="x1=%{x:.5g}<br>x2=%{y:.5g}<br>%{text}<extra></extra>",
    )

    frames = [
        go.Frame(
            name=str(int(generation)),
            data=[
                go.Scatter3d(
                    x=population[generation, :, 0],
                    y=population[generation, :, 1],
                    z=fitness[generation],
                    mode="markers",
                    marker={
                        "size": 4,
                        "color": population[generation, :, 1],
                        "colorscale": "Viridis",
                        "cmin": low,
                        "cmax": high,
                        "line": {"color": "#111827", "width": 1},
                    },
                    text=[f"wartość funkcji celu={value:.6g}" for value in fitness[generation]],
                    hovertemplate="x1=%{x:.5g}<br>x2=%{y:.5g}<br>%{text}<extra></extra>",
                )
            ],
            traces=[1],
        )
        for generation in generations
    ]

    steps = [
        {
            "method": "animate",
            "label": str(int(generation)),
            "args": [
                [str(int(generation))],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for generation in generations
    ]

    fig = go.Figure(data=[surface, population_trace], frames=frames)
    fig.update_layout(
        title=f"{function_name}: ruch populacji {algorithm_name}",
        height=760,
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        scene={
            "xaxis_title": "x1",
            "yaxis_title": "x2",
            "zaxis_title": "f(x1, x2)",
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.05}},
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 0,
                "buttons": [
                    {
                        "label": "Odtwórz",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 80, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pauza",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Pokolenie: "},
                "pad": {"t": 45},
                "steps": steps,
            }
        ],
    )
    return fig


def fig_to_section(fig: go.Figure, include_plotlyjs: bool) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        config={"responsive": True, "displaylogo": False},
    )


def build_report(
    report_path: Path,
    histories: dict[str, dict[str, dict[str, np.ndarray]]],
    high_dim_histories: dict[str, dict[str, dict[str, np.ndarray]]],
    objectives: dict[str, tuple[Objective, tuple[float, float]]],
    grid_size: int,
    frame_stride: int,
) -> None:
    sections: list[str] = []
    include_plotlyjs = True

    for function_name, algorithm_histories in histories.items():
        objective, domain = objectives[function_name]
        ae_history = algorithm_histories["ae"]
        de_history = algorithm_histories["de"]
        metrics_fig = make_metrics_figure(function_name, ae_history, de_history)
        ae_3d_fig = make_3d_population_figure(
            function_name,
            "AE",
            objective,
            domain,
            ae_history,
            grid_size,
            frame_stride,
        )
        de_3d_fig = make_3d_population_figure(
            function_name,
            "DE",
            objective,
            domain,
            de_history,
            grid_size,
            frame_stride,
        )

        section_parts = [
            f"<section><h2>{html.escape(function_name)}</h2>",
            "<h3>Metryki 2D</h3>",
            fig_to_section(metrics_fig, include_plotlyjs),
            "<h3>Ruch populacji 3D: AE</h3>",
            fig_to_section(ae_3d_fig, False),
            "<h3>Ruch populacji 3D: DE</h3>",
            fig_to_section(de_3d_fig, False),
        ]

        section_parts.append("</section>")
        sections.append("\n".join(section_parts))
        include_plotlyjs = False

    high_dim_parts = [
        "<section>",
        "<h2>Porównanie osiągów AE i DE dla 50 wymiarów</h2>",
        "<p>",
        "Poniższe wykresy pokazują przebieg najlepszej, najgorszej i średniej "
        "wartości funkcji celu dla uruchomień w 50 wymiarach. Ta część nie ma "
        "wizualizacji 3D, bo przestrzeń rozwiązań jest 50-wymiarowa.",
        "</p>",
    ]
    for function_name, algorithm_histories in high_dim_histories.items():
        metrics_fig = make_metrics_figure(
            f"{function_name} 50D",
            algorithm_histories["ae"],
            algorithm_histories["de"],
        )
        high_dim_parts.extend(
            [
                f"<h3>{html.escape(function_name)}</h3>",
                fig_to_section(metrics_fig, include_plotlyjs),
            ]
        )
        include_plotlyjs = False

    high_dim_parts.append("</section>")
    sections.append("\n".join(high_dim_parts))

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Raport optymalizacji AE/DE</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: #111827;
      background: #f8fafc;
    }}
    header, section {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 24px;
    }}
    h1, h2, h3 {{
      margin: 0 0 16px;
    }}
    h2 {{
      padding-top: 18px;
      border-top: 1px solid #d1d5db;
      text-transform: capitalize;
    }}
    p {{
      max-width: 900px;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Raport optymalizacji AE/DE</h1>
    <p>
      Każda funkcja jest optymalizowana w dwóch wymiarach. Wykresy 2D
      porównują najlepszą, najgorszą i średnią wartość funkcji celu dla AE
      oraz DE. Wykresy 3D pokazują powierzchnię funkcji celu i położenie całej
      populacji w kolejnych pokoleniach. Kolory powierzchni i populacji kodują
      drugą zmienną wejściową, x2.
    </p>
  </header>
  {"".join(sections)}
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(document, encoding="utf-8")


def run_one_optimizer(
    algorithm_name: str,
    function_name: str,
    objective: Objective,
    domain: tuple[float, float],
    n_dim: int,
    seed: int,
    log_dir: Path,
    verbose_optimizers: bool,
    generations: int | None = None,
) -> dict[str, np.ndarray]:
    if algorithm_name == "ae":
        runner = evolutionary_algorithm
    elif algorithm_name == "de":
        runner = differential_evolution
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    kwargs = {"n_dim": n_dim, "seed": seed}
    if generations is not None:
        kwargs["generations"] = generations

    if verbose_optimizers:
        results = runner(objective, domain, **kwargs)
    else:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{function_name}_{algorithm_name}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file):
                results = runner(objective, domain, **kwargs)

    return history_to_numpy(results)


def run_experiments(
    run_dir: Path,
    log_dir: Path,
    n_dim: int,
    seed: int,
    verbose_optimizers: bool,
    objectives: dict[str, tuple[Objective, tuple[float, float]]] = OBJECTIVES,
    run_label: str = "",
    generations: int | None = None,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    histories: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    run_prefix = f"{run_label} " if run_label else ""
    file_prefix = f"{run_label}_" if run_label else ""

    for function_index, (function_name, (objective, domain)) in enumerate(objectives.items()):
        histories[function_name] = {}
        for algorithm_offset, algorithm_name in enumerate(("ae", "de")):
            run_seed = seed + 1000 * function_index + algorithm_offset
            print(f"Running {algorithm_name.upper()} on {run_prefix}{function_name}...")
            history = run_one_optimizer(
                algorithm_name=algorithm_name,
                function_name=f"{file_prefix}{function_name}",
                objective=objective,
                domain=domain,
                n_dim=n_dim,
                seed=run_seed,
                log_dir=log_dir,
                verbose_optimizers=verbose_optimizers,
                generations=generations,
            )
            save_history(run_dir / f"{file_prefix}{function_name}_{algorithm_name}.npz", history, domain)
            histories[function_name][algorithm_name] = history

    return histories


def load_histories(
    run_dir: Path,
    objectives: dict[str, tuple[Objective, tuple[float, float]]] = OBJECTIVES,
    run_label: str = "",
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    histories: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    file_prefix = f"{run_label}_" if run_label else ""
    for function_name in objectives:
        histories[function_name] = {
            "ae": load_history(run_dir / f"{file_prefix}{function_name}_ae.npz"),
            "de": load_history(run_dir / f"{file_prefix}{function_name}_de.npz"),
        }
    return histories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AE/DE experiments in 2D and generate a Plotly HTML report."
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/zadanie_1"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs/zadanie_1_logs"))
    parser.add_argument("--report", type=Path, default=Path("reports/zadanie_1_report.html"))
    parser.add_argument("--n-dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid-size", type=int, default=70)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--verbose-optimizers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_dim != 2:
        raise ValueError("The main report visualizes 2D objective domains, so --n-dim must be 2.")
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2.")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1.")

    high_dim_objectives = make_high_dim_objectives(args.seed)
    high_dim_generations = HIGH_DIM_GENERATION_MULTIPLIER * default_generations()

    if args.skip_runs:
        histories = load_histories(args.run_dir)
        high_dim_histories = load_histories(
            args.run_dir,
            objectives=high_dim_objectives,
            run_label=f"{HIGH_DIM}d",
        )
    else:
        histories = run_experiments(
            run_dir=args.run_dir,
            log_dir=args.log_dir,
            n_dim=args.n_dim,
            seed=args.seed,
            verbose_optimizers=args.verbose_optimizers,
        )
        high_dim_histories = run_experiments(
            run_dir=args.run_dir,
            log_dir=args.log_dir,
            n_dim=HIGH_DIM,
            seed=args.seed,
            verbose_optimizers=args.verbose_optimizers,
            objectives=high_dim_objectives,
            run_label=f"{HIGH_DIM}d",
            generations=high_dim_generations,
        )

    print("Building Plotly report...")
    build_report(
        report_path=args.report,
        histories=histories,
        high_dim_histories=high_dim_histories,
        objectives=OBJECTIVES,
        grid_size=args.grid_size,
        frame_stride=args.frame_stride,
    )
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
