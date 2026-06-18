from __future__ import annotations

import html
import urllib.error
import urllib.request
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from sa import simulated_annealing

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Plotly is required to generate the report. Run `uv sync`, then rerun "
        "`uv run python zadanie_2.py`."
    ) from exc


TSPLIB_URLS = {
    "eil51": "https://raw.githubusercontent.com/mastqe/tsplib/master/eil51.tsp",
    "berlin52": "https://raw.githubusercontent.com/mastqe/tsplib/master/berlin52.tsp",
    "st70": "https://raw.githubusercontent.com/mastqe/tsplib/master/st70.tsp",
    "eil76": "https://raw.githubusercontent.com/mastqe/tsplib/master/eil76.tsp",
    "pr76": "https://raw.githubusercontent.com/mastqe/tsplib/master/pr76.tsp",
}

DATA_DIR = Path("tsplib")
RUN_DIR = Path("runs/zadanie_2")
REPORT_PATH = Path("reports/zadanie_2_report.html")

ITERATIONS = 10000
SEED = 123
MAX_ROUTE_FRAMES = 180

OPERATOR_LABELS = {
    0: "zamiana dwóch miast",
    1: "odwrócenie fragmentu (2-opt)",
    2: "przeniesienie miasta",
    3: "przeniesienie bloku",
}


def download_if_missing(name: str, url: str, data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{name}.tsp"
    if path.exists():
        return path

    print(f"Downloading {name} from TSPLIB mirror...")
    request = urllib.request.Request(url, headers={"User-Agent": "IO26-zadanie-2"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            path.write_bytes(response.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not download {name}. Save the file manually as {path} or rerun "
            "with network access enabled."
        ) from exc

    return path


def header_value(tokens: list[str], key: str) -> str:
    key = key.upper()
    for idx, token in enumerate(tokens):
        if token.upper() != key:
            continue
        value_idx = idx + 1
        if value_idx < len(tokens) and tokens[value_idx] == ":":
            value_idx += 1
        if value_idx >= len(tokens):
            raise ValueError(f"Missing value for TSPLIB header field {key}.")
        return tokens[value_idx]
    raise ValueError(f"Missing TSPLIB header field {key}.")


def parse_tsplib_euc_2d(path: Path) -> tuple[str, np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tokens = text.replace(":", " : ").split()

    name = header_value(tokens, "NAME")
    edge_weight_type = header_value(tokens, "EDGE_WEIGHT_TYPE").upper()
    if edge_weight_type != "EUC_2D":
        raise ValueError(f"{name} uses {edge_weight_type}; this script expects EUC_2D.")

    dimension = int(header_value(tokens, "DIMENSION"))
    try:
        coord_start = tokens.index("NODE_COORD_SECTION") + 1
    except ValueError as exc:
        raise ValueError(f"{name} does not contain NODE_COORD_SECTION.") from exc

    coords_by_id: list[tuple[int, float, float]] = []
    idx = coord_start
    while idx < len(tokens) and tokens[idx].upper() != "EOF":
        node_id = int(float(tokens[idx]))
        x = float(tokens[idx + 1])
        y = float(tokens[idx + 2])
        coords_by_id.append((node_id, x, y))
        idx += 3

    if len(coords_by_id) != dimension:
        raise ValueError(
            f"{name} declares {dimension} nodes, but {len(coords_by_id)} coordinates were parsed."
        )

    coords_by_id.sort(key=lambda item: item[0])
    coords = np.asarray([(x, y) for _, x, y in coords_by_id], dtype=np.float32)
    return name, coords


def euc_2d_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    euclidean = np.sqrt(np.sum(diff * diff, axis=-1))
    return np.floor(euclidean + 0.5).astype(np.float32)


def history_to_numpy(results: list[dict]) -> dict[str, np.ndarray]:
    return {
        "route": np.stack([np.asarray(state["route"]) for state in results]).astype(np.int32),
        "cost": np.asarray([state["cost"] for state in results], dtype=np.float32),
        "best_route": np.stack([np.asarray(state["best_route"]) for state in results]).astype(np.int32),
        "best_cost": np.asarray([state["best_cost"] for state in results], dtype=np.float32),
        "temperature": np.asarray([state["temperature"] for state in results], dtype=np.float32),
        "operator_id": np.asarray([state["operator_id"] for state in results], dtype=np.int32),
        "accepted": np.asarray([state["accepted"] for state in results], dtype=bool),
        "candidate_cost": np.asarray([state["candidate_cost"] for state in results], dtype=np.float32),
        "cost_delta": np.asarray([state["cost_delta"] for state in results], dtype=np.float32),
    }


def save_run(path: Path, name: str, coords: np.ndarray, dist: np.ndarray, history: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, name=name, coords=coords, dist=dist, **history)


def load_run(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def closed_route_xy(coords: np.ndarray, route: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = coords[route.astype(np.int32)]
    closed = np.vstack([ordered, ordered[0]])
    return closed[:, 0], closed[:, 1]


def route_frame_indices(n_steps: int) -> np.ndarray:
    if n_steps <= MAX_ROUTE_FRAMES:
        return np.arange(n_steps)

    indices = np.unique(np.linspace(0, n_steps - 1, MAX_ROUTE_FRAMES, dtype=np.int32))
    if indices[-1] != n_steps - 1:
        indices = np.append(indices, n_steps - 1)
    return indices


def make_route_figure(problem_name: str, coords: np.ndarray, history: dict[str, np.ndarray]) -> go.Figure:
    routes = history["route"]
    best_routes = history["best_route"]
    costs = history["cost"]
    best_costs = history["best_cost"]
    temperatures = history["temperature"]
    operator_ids = history["operator_id"]
    accepted = history["accepted"]
    frame_ids = route_frame_indices(routes.shape[0])

    first_idx = int(frame_ids[0])
    x_route, y_route = closed_route_xy(coords, routes[first_idx])
    x_best, y_best = closed_route_xy(coords, best_routes[first_idx])

    city_trace = go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers",
        marker={"size": 5, "color": "#111827"},
        name="Miasta",
        text=[str(idx + 1) for idx in range(coords.shape[0])],
        hovertemplate="miasto %{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
    )
    route_trace = go.Scatter(
        x=x_route,
        y=y_route,
        mode="lines",
        line={"color": "#2563eb", "width": 2},
        name="Aktualna trasa",
        hoverinfo="skip",
    )
    best_trace = go.Scatter(
        x=x_best,
        y=y_best,
        mode="lines",
        line={"color": "#dc2626", "width": 3},
        name="Najlepsza trasa",
        hoverinfo="skip",
    )

    frames = []
    for step_idx in frame_ids:
        step_idx = int(step_idx)
        x_route, y_route = closed_route_xy(coords, routes[step_idx])
        x_best, y_best = closed_route_xy(coords, best_routes[step_idx])
        operator_label = (
            "stan początkowy"
            if step_idx == 0
            else OPERATOR_LABELS.get(int(operator_ids[step_idx]), "nieznany operator")
        )
        accepted_label = "tak" if bool(accepted[step_idx]) else "nie"
        frames.append(
            go.Frame(
                name=str(step_idx),
                data=[
                    go.Scatter(x=x_route, y=y_route, mode="lines"),
                    go.Scatter(x=x_best, y=y_best, mode="lines"),
                ],
                traces=[1, 2],
                layout={
                    "title": (
                        f"{problem_name}: krok {step_idx}, koszt={costs[step_idx]:.0f}, "
                        f"najlepszy={best_costs[step_idx]:.0f}, "
                        f"T={temperatures[step_idx]:.4g}, operator={operator_label}, "
                        f"zaakceptowano={accepted_label}"
                    )
                },
            )
        )

    steps = [
        {
            "method": "animate",
            "label": str(int(step_idx)),
            "args": [
                [str(int(step_idx))],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for step_idx in frame_ids
    ]

    fig = go.Figure(data=[city_trace, route_trace, best_trace], frames=frames)
    fig.update_layout(
        title=f"{problem_name}: krok 0, koszt={costs[0]:.0f}, najlepszy={best_costs[0]:.0f}",
        height=720,
        template="plotly_white",
        margin={"l": 45, "r": 20, "t": 70, "b": 55},
        xaxis_title="x",
        yaxis_title="y",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": -0.08,
                "buttons": [
                    {
                        "label": "Odtwórz",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 90, "redraw": True},
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
                "currentvalue": {"prefix": "Krok: "},
                "pad": {"t": 45},
                "steps": steps,
            }
        ],
    )
    return fig


def rolling_acceptance(accepted: np.ndarray, window: int = 200) -> np.ndarray:
    accepted_float = accepted.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32)
    numerator = np.convolve(accepted_float, kernel, mode="same")
    denominator = np.convolve(np.ones_like(accepted_float), kernel, mode="same")
    return numerator / np.maximum(denominator, 1.0)


def make_cost_figure(problem_name: str, history: dict[str, np.ndarray]) -> go.Figure:
    x = np.arange(history["cost"].shape[0])
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=(
            "Koszt trasy i temperatura",
            "Lokalny odsetek zaakceptowanych ruchów",
        ),
    )
    fig.add_trace(
        go.Scatter(x=x, y=history["cost"], mode="lines", name="aktualny koszt", line={"color": "#2563eb"}),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=history["best_cost"], mode="lines", name="najlepszy koszt", line={"color": "#dc2626"}),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=history["temperature"],
            mode="lines",
            name="temperatura",
            line={"color": "#059669", "dash": "dot"},
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x[1:],
            y=rolling_acceptance(history["accepted"][1:]),
            mode="lines",
            name="zaakceptowane ruchy",
            line={"color": "#7c3aed"},
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=f"{problem_name}: przebieg optymalizacji",
        height=720,
        template="plotly_white",
        margin={"l": 70, "r": 60, "t": 85, "b": 55},
    )
    fig.update_xaxes(title_text="Iteracja", row=2, col=1)
    fig.update_yaxes(title_text="Koszt", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Temperatura", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Odsetek", range=[0, 1], row=2, col=1)
    return fig


def make_operator_figure(problem_name: str, history: dict[str, np.ndarray]) -> go.Figure:
    operators = history["operator_id"][1:]
    accepted = history["accepted"][1:]
    labels = [OPERATOR_LABELS[idx] for idx in range(len(OPERATOR_LABELS))]
    counts = np.bincount(operators, minlength=len(OPERATOR_LABELS)).astype(np.float32)
    accepted_counts = np.bincount(operators[accepted], minlength=len(OPERATOR_LABELS)).astype(np.float32)
    acceptance_rate = accepted_counts / np.maximum(counts, 1.0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=labels, y=counts, name="użycia operatora", marker_color="#2563eb"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=acceptance_rate,
            mode="lines+markers",
            name="odsetek zaakceptowanych ruchów",
            line={"color": "#dc2626"},
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"{problem_name}: operatory zmiany trasy",
        height=470,
        template="plotly_white",
        margin={"l": 70, "r": 70, "t": 75, "b": 105},
    )
    fig.update_yaxes(title_text="Liczba użyć", secondary_y=False)
    fig.update_yaxes(title_text="Odsetek akceptacji", range=[0, 1], secondary_y=True)
    return fig


def summary_table(name: str, coords: np.ndarray, history: dict[str, np.ndarray]) -> str:
    initial_cost = float(history["cost"][0])
    best_cost = float(history["best_cost"][-1])
    improvement = 100.0 * (initial_cost - best_cost) / initial_cost
    accepted_count = int(np.sum(history["accepted"][1:]))
    attempted = int(history["accepted"].shape[0] - 1)
    acceptance_rate = accepted_count / max(attempted, 1)

    rows = [
        ("Liczba miast", str(coords.shape[0])),
        ("Liczba iteracji", str(attempted)),
        ("Koszt początkowy", f"{initial_cost:.0f}"),
        ("Najlepszy znaleziony koszt", f"{best_cost:.0f}"),
        ("Poprawa względem startu", f"{improvement:.2f}%"),
        ("Zaakceptowane ruchy", f"{accepted_count} / {attempted}"),
        ("Odsetek akceptacji", f"{100.0 * acceptance_rate:.2f}%"),
    ]
    body = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    return f"""
<table class="summary-table" aria-label="Podsumowanie {html.escape(name)}">
  <tbody>
    {body}
  </tbody>
</table>
"""


def fig_to_html(fig: go.Figure, include_plotlyjs: bool) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=True if include_plotlyjs else False,
        config={"responsive": True, "displaylogo": False},
    )


def build_report(
    report_path: Path,
    runs: dict[str, dict[str, np.ndarray]],
) -> None:
    sections: list[str] = []
    include_plotlyjs = True

    for problem_name, run in runs.items():
        coords = run["coords"]
        route_fig = make_route_figure(problem_name, coords, run)
        cost_fig = make_cost_figure(problem_name, run)
        operator_fig = make_operator_figure(problem_name, run)
        sections.append(
            "\n".join(
                [
                    f"<section><h2>{html.escape(problem_name)}</h2>",
                    "<p>",
                    "Wykres trasy pokazuje aktualne rozwiązanie oraz najlepsze rozwiązanie "
                    "znalezione do danej iteracji. Punkt startowy cyklu nie jest wyróżniany, "
                    "ponieważ w TSP istotny jest kształt zamkniętej trasy, a nie miejsce "
                    "rozpoczęcia zapisu permutacji.",
                    "</p>",
                    summary_table(problem_name, coords, run),
                    "<h3>Symulacja krokowa trasy</h3>",
                    fig_to_html(route_fig, include_plotlyjs),
                    "<h3>Przebieg kosztu</h3>",
                    fig_to_html(cost_fig, False),
                    "<h3>Statystyki operatorów</h3>",
                    fig_to_html(operator_fig, False),
                    "</section>",
                ]
            )
        )
        include_plotlyjs = False

    document = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Raport zadania 2: symulowane wyżarzanie dla TSP</title>
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
    }}
    p {{
      max-width: 900px;
      line-height: 1.5;
    }}
    .summary-table {{
      border-collapse: collapse;
      margin: 18px 0 26px;
      min-width: 420px;
      background: #ffffff;
    }}
    .summary-table th,
    .summary-table td {{
      border: 1px solid #d1d5db;
      padding: 9px 12px;
      text-align: left;
    }}
    .summary-table th {{
      background: #eef2ff;
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Raport zadania 2: symulowane wyżarzanie dla TSP</h1>
    <p>
      Raport przedstawia pięć instancji z biblioteki TSPLIB. Każda instancja
      jest rozwiązywana algorytmem symulowanego wyżarzania z czterema
      operatorami zmiany trasy: zamianą dwóch miast, odwróceniem fragmentu,
      przeniesieniem jednego miasta oraz przeniesieniem bloku miast.
      Wizualizacja krokowa używa próbkowania iteracji, żeby plik HTML pozostał
      responsywny, natomiast wykresy kosztu korzystają z pełnej historii
      zapisanego przebiegu.
    </p>
  </header>
  {"".join(sections)}
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(document, encoding="utf-8")


def run_problem(name: str, tsp_path: Path, run_dir: Path = RUN_DIR) -> dict[str, np.ndarray]:
    parsed_name, coords = parse_tsplib_euc_2d(tsp_path)
    dist = euc_2d_distance_matrix(coords)
    print(f"Solving {parsed_name} ({coords.shape[0]} cities)...")
    results = simulated_annealing(jnp.asarray(dist), iterations=ITERATIONS, seed=SEED)
    history = history_to_numpy(results)
    save_run(run_dir / f"{name}.npz", parsed_name, coords, dist, history)
    return {"name": np.asarray(parsed_name), "coords": coords, "dist": dist, **history}


def main() -> None:
    tsp_paths = {
        name: download_if_missing(name, url)
        for name, url in TSPLIB_URLS.items()
    }

    runs = {
        name: run_problem(name, path)
        for name, path in tsp_paths.items()
    }

    print("Building Plotly report...")
    build_report(REPORT_PATH, runs)
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
