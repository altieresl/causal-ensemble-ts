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


def _display_scrollable_dataframe(frame: pd.DataFrame, *, max_rows: int = 50) -> None:
    from IPython.display import HTML, display

    if frame is None:
        frame = pd.DataFrame()

    preview = frame.head(max_rows)
    table_html = preview.to_html(index=False, escape=False, na_rep="")
    display(
        HTML(
            f"""
            <div style="max-height: 420px; overflow: auto; border: 1px solid #d0d7de; border-radius: 6px;">
              <div style="min-width: max-content; padding: 4px 8px;">
                {table_html}
              </div>
            </div>
            """
        )
    )


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
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=max(1.0, width * 0.45),
            arrowcolor=color,
            opacity=0.85,
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
    output_dag = widgets.Output()
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
        dag = plot_temporal_dag(
            probabilistic_summary,
            min_probability=min_prob.value,
            max_lag=lag_slider.value,
            source=source_dropdown.value,
            target=target_dropdown.value,
            title="DAG temporal filtrado",
        )

        with output_graph:
            output_graph.clear_output(wait=True)
            _display_plotly_figure(fig)

        with output_dag:
            output_dag.clear_output(wait=True)
            _display_plotly_figure(dag)

        with output_table:
            output_table.clear_output(wait=True)
            _display_scrollable_dataframe(filtered, max_rows=50)

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
    dashboard = widgets.VBox([controls, output_graph, output_dag, output_table, output_consistency])

    _refresh()
    display(dashboard)
    return dashboard

def create_advanced_expert_dashboard(
    processed_data: pd.DataFrame,
    candidate_methods: dict,
    candidate_method_kwargs: dict,
    method_weights: dict,
    all_nodes: list[str],
    pipeline_callback, # Function to run the selection pipeline
    initial_expert_knowledge: pd.DataFrame | list[dict] | None = None,
    initial_quick_mode: bool = True,
    initial_n_bootstrap: int | None = None,
    initial_parallel_jobs: int = 2,
):
    """
    Cria um dashboard avançado que integra a inserção de conhecimento especialista,
    ajuste de parâmetros e permite rodar/re-rodar o pipeline visualizando os resultados interativamente.
    """
    widgets = _require_widgets()
    from IPython.display import display
    import json

    if not all_nodes:
        all_nodes = list(processed_data.columns)
    if not all_nodes:
        raise ValueError("all_nodes nao pode ser vazio quando processed_data nao tem colunas.")

    # 1. Configurações e Parâmetros
    html_title_params = widgets.HTML("<h3>Parametros do modelo</h3>")
    initial_quick_mode = bool(initial_quick_mode)
    default_bootstraps = 4 if initial_quick_mode else 8
    quick_mode_cb = widgets.Checkbox(value=initial_quick_mode, description="Quick mode", indent=False)
    parallel_jobs_ui = widgets.IntSlider(
        value=max(1, min(8, int(initial_parallel_jobs))),
        min=1,
        max=8,
        description="Jobs (Threads):",
        style={'description_width': 'initial'},
    )
    n_bootstrap_ui = widgets.IntSlider(
        value=max(1, min(50, int(initial_n_bootstrap or default_bootstraps))),
        min=1,
        max=50,
        description="Bootstraps:",
        style={'description_width': 'initial'},
    )
    
    def on_quick_mode_change(change):
        if change.new:
            n_bootstrap_ui.value = 4
        else:
            n_bootstrap_ui.value = 8
    quick_mode_cb.observe(on_quick_mode_change, names="value")
    
    params_controls = widgets.VBox([
        quick_mode_cb,
        n_bootstrap_ui,
        parallel_jobs_ui
    ])

    # 2. Conhecimento Especialista
    html_title_expert = widgets.HTML("<h3>Conhecimento especialista</h3><p>Adicione regras causais para orientar ou vetar arestas.</p>")
    
    # Layout para que os títulos (description) caibam sem serem cortados
    style = {'description_width': '200px'}
    layout = widgets.Layout(width='500px')
    
    source_dd = widgets.Dropdown(options=all_nodes, description="1. No origem (causa):", style=style, layout=layout)
    target_default = all_nodes[1] if len(all_nodes) > 1 else all_nodes[0]
    target_dd = widgets.Dropdown(options=all_nodes, value=target_default, description="2. No destino (efeito):", style=style, layout=layout)
    lag_input = widgets.IntText(value=0, description="3. Defasagem temporal (Lag):", style=style, layout=layout)
    
    relation_dd = widgets.Dropdown(
        options=[
            ("Forte (aumenta prob.)", "strong"), 
            ("Fraca (reduz prob.)", "weak"), 
            ("Inversa (Sinal negativo)", "inverse"), 
            ("Nenhuma (Proibida)", "none")
        ], 
        description="4. Forca da relacao:", style=style, layout=layout
    )
    
    constraint_dd = widgets.Dropdown(
        options=[("Flexivel (soft)", "soft"), ("Absoluta (hard)", "hard")], 
        description="5. Tipo de restricao:", style=style, layout=layout
    )
    
    conf_input = widgets.BoundedFloatText(value=0.9, min=0.0, max=1.0, step=0.05, description="6. Confianca (0 a 1):", style=style, layout=layout)
    prior_input = widgets.BoundedFloatText(value=0.9, min=0.0, max=1.0, step=0.05, description="7. Prob. previa (0 a 1):", style=style, layout=layout)
    
    add_rule_btn = widgets.Button(description="Adicionar regra", button_style="info", layout=widgets.Layout(width='500px'))
    clear_rules_btn = widgets.Button(description="Limpar regras", button_style="danger", layout=widgets.Layout(width='500px'))
    
    rules_text = widgets.Textarea(
        value="[]",
        placeholder="Nenhuma regra adicionada...",
        description="Regras atuais:",
        disabled=True,
        layout=widgets.Layout(width="100%", height="170px")
    )
    rule_status = widgets.HTML("")
    
    if initial_expert_knowledge is None:
        current_rules = []
    else:
        current_rules = pd.DataFrame(initial_expert_knowledge).to_dict("records")

    def update_rules_ui():
        rules_text.value = json.dumps(current_rules, indent=2)
        rule_status.value = f"<b>{len(current_rules)}</b> regra(s) na lista."

    def on_add_rule(_):
        if lag_input.value < 0:
            rule_status.value = "<span style='color:#b00020'>Lag nao pode ser negativo.</span>"
            return
        rule = {
            "source": source_dd.value,
            "target": target_dd.value,
            "lag": int(lag_input.value),
            "relation": relation_dd.value,
            "confidence": float(conf_input.value),
            "constraint": constraint_dd.value,
            "prior_probability": float(prior_input.value),
        }
        current_rules.append(rule)
        update_rules_ui()
        ui.pipeline_result = None

    def on_clear_rules(_):
        current_rules.clear()
        update_rules_ui()
        ui.pipeline_result = None

    add_rule_btn.on_click(on_add_rule)
    clear_rules_btn.on_click(on_clear_rules)

    rule_controls = widgets.VBox([
        source_dd,
        target_dd,
        lag_input,
        relation_dd,
        constraint_dd,
        conf_input,
        prior_input,
        widgets.HTML("<hr>"),
        add_rule_btn,
        clear_rules_btn,
        rule_status,
        widgets.HTML("<br>"),
        rules_text
    ])
    update_rules_ui()

    # 3. Execução e Output
    html_title_run = widgets.HTML("<h3>Execucao do ensemble</h3>")
    run_btn = widgets.Button(description="Rodar pipeline com estas regras", button_style="success", layout=widgets.Layout(width="500px", height="50px"))

    log_output = widgets.Output() # Para os prints
    dash_output = widgets.Output() # Para o dashboard filtrado

    def on_run_pipeline(_):
        run_btn.disabled = True
        with log_output:
            log_output.clear_output(wait=True)
            print("Inicializando descoberta causal... Isso pode demorar.")
            
        with dash_output:
            dash_output.clear_output(wait=True)
            
        try:
            # Chama o wrapper do pipeline
            callback_result = pipeline_callback(
                quick_mode=quick_mode_cb.value,
                n_bootstrap=n_bootstrap_ui.value,
                parallel_jobs=parallel_jobs_ui.value,
                expert_knowledge=current_rules,
                processed_data=processed_data,
                candidate_methods=candidate_methods,
                candidate_method_kwargs=candidate_method_kwargs,
                method_weights=method_weights
            )
            if isinstance(callback_result, Mapping):
                result_summary = callback_result.get("probabilistic_summary", pd.DataFrame())
                consistency = callback_result.get("consistency", pd.DataFrame())
            else:
                result_summary, consistency = callback_result
            ui.pipeline_result = callback_result
            ui.result_summary = result_summary
            ui.consistency_matrix = consistency
            ui.current_rules = current_rules
            
            with log_output:
                print("Finalizado com sucesso. Veja o dashboard interativo abaixo.")
                
            with dash_output:
                dash_output.clear_output(wait=True)
                create_interactive_ensemble_dashboard(
                    result_summary,
                    consistency_matrix=consistency
                )
        except Exception as e:
            with log_output:
                print(f"Erro durante a execucao: {e}")
        finally:
            run_btn.disabled = False

    run_btn.on_click(on_run_pipeline)

    params_tab = widgets.VBox([html_title_params, params_controls])
    expert_tab = widgets.VBox([html_title_expert, rule_controls])
    run_tab = widgets.VBox([html_title_run, run_btn, widgets.HTML("<br>"), log_output])
    results_tab = widgets.VBox([dash_output])
    ui = widgets.Tab(children=[params_tab, expert_tab, run_tab, results_tab])
    for index, title in enumerate(["Parametros", "Especialista", "Execucao", "Resultados"]):
        ui.set_title(index, title)
    ui.current_rules = current_rules
    ui.pipeline_result = None
    ui.result_summary = pd.DataFrame()
    ui.consistency_matrix = pd.DataFrame()
    ui.quick_mode_control = quick_mode_cb
    ui.bootstrap_control = n_bootstrap_ui
    ui.parallel_jobs_control = parallel_jobs_ui
    ui.expert_source_control = source_dd
    ui.expert_target_control = target_dd
    ui.expert_lag_control = lag_input
    ui.add_expert_rule_button = add_rule_btn

    def _invalidate_result(change):
        if change.get("name") == "value":
            ui.pipeline_result = None

    quick_mode_cb.observe(_invalidate_result, names="value")
    n_bootstrap_ui.observe(_invalidate_result, names="value")
    parallel_jobs_ui.observe(_invalidate_result, names="value")

    display(ui)
    return ui


def plot_temporal_dag(
    summary: pd.DataFrame,
    *,
    min_probability: float = 0.5,
    max_lag: int | None = None,
    source: str | None = None,
    target: str | None = None,
    max_edges: int | None = 12,
    title: str = "DAG temporal",
):
    _, go = _require_plotly()
    filtered = filter_probabilistic_edges(
        summary,
        min_probability=min_probability,
        max_lag=max_lag,
        source=source,
        target=target,
    ).copy()

    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (sem arestas com o filtro atual)")
        return fig

    filtered["lag"] = pd.to_numeric(filtered.get("lag"), errors="coerce").fillna(0).astype(int)
    filtered = filtered[filtered["lag"] >= 0]
    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (sem lags validos)")
        return fig

    sort_columns = [column for column in ["edge_probability", "confidence", "lag"] if column in filtered.columns]
    if sort_columns:
        filtered = filtered.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    if max_edges is not None:
        filtered = filtered.head(int(max_edges)).reset_index(drop=True)

    variables = sorted(set(filtered["source"].astype(str)) | set(filtered["target"].astype(str)))
    max_lag_value = int(filtered["lag"].max()) if not filtered.empty else 0
    lag_levels = list(range(max_lag_value, -1, -1))

    x_positions = {lag: index for index, lag in enumerate(lag_levels)}
    y_positions = {variable: len(variables) - index - 1 for index, variable in enumerate(variables)}

    fig = go.Figure()

    for lag in lag_levels:
        x_coord = x_positions[lag]
        fig.add_shape(
            type="line",
            x0=x_coord,
            x1=x_coord,
            y0=-0.7,
            y1=max(len(variables) - 0.3, 0.3),
            line={"color": "rgba(120, 144, 156, 0.25)", "width": 1, "dash": "dot"},
        )
        label = "t" if lag == 0 else f"t-{lag}"
        fig.add_annotation(
            x=x_coord,
            y=len(variables) + 0.05,
            text=label,
            showarrow=False,
            font={"size": 13, "color": "#455a64"},
        )

    for _, edge in filtered.iterrows():
        source_name = str(edge["source"])
        target_name = str(edge["target"])
        lag = int(edge["lag"])
        probability = float(edge.get("edge_probability", 0.0))
        confidence_value = edge.get("confidence", np.nan)
        confidence = float(confidence_value) if pd.notna(confidence_value) else 0.0
        width = 1.5 + 5.5 * probability
        color = f"rgba(0, 121, 107, {0.25 + 0.75 * max(confidence, probability)})"

        x0 = x_positions[lag]
        x1 = x_positions[0]
        y0 = y_positions[source_name]
        y1 = y_positions[target_name]

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"width": width, "color": color},
                hoverinfo="text",
                text=(
                    f"{source_name}(t-{lag}) -> {target_name}(t)<br>"
                    f"prob={probability:.3f}<br>"
                    f"conf={confidence:.3f}"
                ),
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=max(1.2, width * 0.45),
            arrowcolor=color,
            opacity=0.9,
        )

    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    for variable in variables:
        y_coord = y_positions[variable]
        for lag in lag_levels:
            if lag != 0 and not ((filtered["source"].astype(str) == variable) & (filtered["lag"] == lag)).any():
                continue
            x_coord = x_positions[lag]
            label = f"{variable} (t)" if lag == 0 else f"{variable} (t-{lag})"
            node_x.append(x_coord)
            node_y.append(y_coord)
            node_text.append(label)
            node_hover.append(label)

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker={"size": 28, "color": "#f4a261", "line": {"width": 1.5, "color": "#ffffff"}},
            hoverinfo="text",
            hovertext=node_hover,
            showlegend=False,
        )
    )
    for x_coord, y_coord, label in zip(node_x, node_y, node_text):
        fig.add_annotation(
            x=x_coord,
            y=y_coord + 0.32,
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font={"size": 11, "color": "#1f2933"},
            bgcolor="rgba(255, 255, 255, 0.82)",
            bordercolor="rgba(255, 255, 255, 0.0)",
            borderpad=2,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 30, "r": 30, "t": 80, "b": 30},
        plot_bgcolor="#fcfcf8",
        paper_bgcolor="#ffffff",
    )
    return fig
