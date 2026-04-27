from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd


def _require_plotly() -> tuple[object, object]:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception as exc:  # pragma: no cover
        raise ImportError("Instale plotly para usar visualizações interativas.") from exc
    return px, go


def _require_widgets() -> object:
    try:
        import ipywidgets as widgets
    except Exception as exc:  # pragma: no cover
        raise ImportError("Instale ipywidgets para usar o dashboard interativo.") from exc
    return widgets


def _display_plotly_figure(fig: object) -> None:
    from IPython.display import HTML, display

    try:
        display(fig)
    except Exception:
        html = fig.to_html(include_plotlyjs="cdn", full_html=False)
        display(HTML(html))


def filter_probabilistic_edges(
    summary: pd.DataFrame,
    *,
    min_probability: float = 0.5,
    max_lag: int | None = None,
    source: str | None = None,
    target: str | None = None,
) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame(columns=list(summary.columns) if summary is not None else [])

    frame = summary.copy()
    if "edge_probability" in frame.columns:
        frame = frame[frame["edge_probability"] >= float(min_probability)]

    if max_lag is not None and "lag" in frame.columns:
        frame = frame[pd.to_numeric(frame["lag"], errors="coerce") <= int(max_lag)]

    if source and source != "Todos":
        frame = frame[frame["source"].astype(str) == str(source)]
    if target and target != "Todos":
        frame = frame[frame["target"].astype(str) == str(target)]

    return frame.reset_index(drop=True)


def plot_probabilistic_causal_graph(
    summary: pd.DataFrame,
    *,
    min_probability: float = 0.5,
    max_lag: int | None = None,
    source: str | None = None,
    target: str | None = None,
    title: str = "Grafo causal probabilístico",
):
    _, go = _require_plotly()
    filtered = filter_probabilistic_edges(
        summary,
        min_probability=min_probability,
        max_lag=max_lag,
        source=source,
        target=target,
    )

    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (sem arestas com o filtro atual)")
        return fig

    nodes = sorted(set(filtered["source"].astype(str)) | set(filtered["target"].astype(str)))
    n_nodes = len(nodes)
    if n_nodes == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (sem nós)")
        return fig

    angle_step = (2.0 * math.pi) / n_nodes
    positions = {
        node: (math.cos(i * angle_step), math.sin(i * angle_step))
        for i, node in enumerate(nodes)
    }

    fig = go.Figure()

    # Usamos intensidade de cor e largura para refletir força e confiança da conexão.
    for _, edge in filtered.iterrows():
        source_name = str(edge["source"])
        target_name = str(edge["target"])
        x0, y0 = positions[source_name]
        x1, y1 = positions[target_name]

        probability = float(edge.get("edge_probability", 0.0))
        confidence = float(edge.get("confidence", 0.0)) if pd.notna(edge.get("confidence", np.nan)) else 0.0
        lag = int(edge.get("lag", 0))
        width = 1.0 + 5.0 * probability
        color = f"rgba(30, 136, 229, {0.2 + 0.8 * confidence})"

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"width": width, "color": color},
                hoverinfo="text",
                text=(
                    f"{source_name} -> {target_name}<br>"
                    f"lag={lag}<br>"
                    f"prob={probability:.3f}<br>"
                    f"conf={confidence:.3f}"
                ),
                showlegend=False,
            )
        )

    node_x = [positions[node][0] for node in nodes]
    node_y = [positions[node][1] for node in nodes]
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=nodes,
            textposition="top center",
            marker={"size": 18, "color": "#ef6c00", "line": {"width": 1, "color": "#ffffff"}},
            hoverinfo="text",
            hovertext=nodes,
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


def plot_method_consistency_heatmap(consistency_matrix: pd.DataFrame, *, title: str = "Consistência entre métodos"):
    px, _ = _require_plotly()
    if consistency_matrix is None or consistency_matrix.empty:
        return px.imshow(np.array([[0.0]]), title=f"{title} (sem dados)")

    fig = px.imshow(
        consistency_matrix,
        text_auto=".2f",
        color_continuous_scale="YlGnBu",
        zmin=0.0,
        zmax=1.0,
        title=title,
    )
    fig.update_layout(template="plotly_white")
    return fig


def create_interactive_ensemble_dashboard(
    probabilistic_summary: pd.DataFrame,
    *,
    consistency_matrix: pd.DataFrame | None = None,
):
    widgets = _require_widgets()
    from IPython.display import display

    if probabilistic_summary is None:
        probabilistic_summary = pd.DataFrame()

    min_prob = widgets.FloatSlider(
        value=0.6,
        min=0.0,
        max=1.0,
        step=0.01,
        description="Prob. mín.",
        continuous_update=False,
    )

    max_lag_value = int(pd.to_numeric(probabilistic_summary.get("lag", pd.Series([0])), errors="coerce").max())
    max_lag_value = max(0, max_lag_value if np.isfinite(max_lag_value) else 0)
    lag_slider = widgets.IntSlider(
        value=max_lag_value,
        min=0,
        max=max_lag_value,
        step=1,
        description="Lag máx.",
        continuous_update=False,
    )

    all_nodes = sorted(
        set(probabilistic_summary.get("source", pd.Series(dtype=object)).dropna().astype(str).tolist())
        | set(probabilistic_summary.get("target", pd.Series(dtype=object)).dropna().astype(str).tolist())
    )
    node_options = ["Todos", *all_nodes]

    source_dropdown = widgets.Dropdown(options=node_options, value="Todos", description="Origem")
    target_dropdown = widgets.Dropdown(options=node_options, value="Todos", description="Destino")

    output_graph = widgets.Output()
    output_table = widgets.Output()
    output_consistency = widgets.Output()

    def _refresh(*_args: object) -> None:
        filtered = filter_probabilistic_edges(
            probabilistic_summary,
            min_probability=min_prob.value,
            max_lag=lag_slider.value,
            source=source_dropdown.value,
            target=target_dropdown.value,
        )

        fig = plot_probabilistic_causal_graph(
            probabilistic_summary,
            min_probability=min_prob.value,
            max_lag=lag_slider.value,
            source=source_dropdown.value,
            target=target_dropdown.value,
            title="Grafo causal filtrado",
        )

        with output_graph:
            output_graph.clear_output(wait=True)
            _display_plotly_figure(fig)

        with output_table:
            output_table.clear_output(wait=True)
            display(filtered.head(50))

        with output_consistency:
            output_consistency.clear_output(wait=True)
            if consistency_matrix is not None and not consistency_matrix.empty:
                heatmap = plot_method_consistency_heatmap(consistency_matrix)
                _display_plotly_figure(heatmap)

    min_prob.observe(_refresh, names="value")
    lag_slider.observe(_refresh, names="value")
    source_dropdown.observe(_refresh, names="value")
    target_dropdown.observe(_refresh, names="value")

    controls = widgets.HBox([min_prob, lag_slider, source_dropdown, target_dropdown])
    dashboard = widgets.VBox([controls, output_graph, output_table, output_consistency])

    _refresh()
    display(dashboard)
    return dashboard
