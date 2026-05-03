import numpy as np
import pandas as pd

def generate_synthetic_timeseries(n_samples=500, noise_std=0.1):
    """
    Gera um dataset sintético de séries temporais com um Ground Truth causal claro.
    
    A estrutura causal (Ground Truth) é:
    - X_t = 0.7 * X_{t-1} + e_X
    - Y_t = 0.5 * Y_{t-1} + 0.8 * X_{t-2} + e_Y
    - Z_t = 0.6 * Z_{t-1} + 0.9 * Y_{t-1} - 0.5 * X_{t-1} + e_Z
    
    Ground Truth Edges Esperados:
    - (X, X, lag=1)
    - (Y, Y, lag=1)
    - (Z, Z, lag=1)
    - (X, Y, lag=2)
    - (Y, Z, lag=1)
    - (X, Z, lag=1)
    """
    np.random.seed(42)
    
    # Inicializa arrays
    X = np.zeros(n_samples)
    Y = np.zeros(n_samples)
    Z = np.zeros(n_samples)
    
    # Gerar dados
    for t in range(2, n_samples):
        X[t] = 0.7 * X[t-1] + np.random.normal(0, noise_std)
        Y[t] = 0.5 * Y[t-1] + 0.8 * X[t-2] + np.random.normal(0, noise_std)
        Z[t] = 0.6 * Z[t-1] + 0.9 * Y[t-1] - 0.5 * X[t-1] + np.random.normal(0, noise_std)
        
    df = pd.DataFrame({
        "X": X,
        "Y": Y,
        "Z": Z
    })
    
    ground_truth_edges = [
        {"source": "X", "target": "X", "lag": 1},
        {"source": "Y", "target": "Y", "lag": 1},
        {"source": "Z", "target": "Z", "lag": 1},
        {"source": "X", "target": "Y", "lag": 2},
        {"source": "Y", "target": "Z", "lag": 1},
        {"source": "X", "target": "Z", "lag": 1},
    ]
    gt_df = pd.DataFrame(ground_truth_edges)
    
    return df, gt_df

def compute_structural_metrics(predicted_summary: pd.DataFrame, ground_truth: pd.DataFrame, prob_threshold=0.5):
    """
    Calcula as métricas estruturais (Precision, Recall, F1, SHD) para validação do ensemble.
    Ignora a métrica de "Reversed Edges" direta no SHD simplificado e lida como FP + FN para séries temporais direcionais.
    
    Args:
        predicted_summary: pd.DataFrame vindo do ensemble (edge_probability).
        ground_truth: DataFrame de gabarito com as colunas ["source", "target", "lag"].
        prob_threshold: Limiar de probabilidade para considerar uma aresta prevista como válida.
        
    Retorna:
        Dict com as métricas calculadas.
    """
    # Filtrar previsões pelo limiar escolhido
    if "edge_probability" in predicted_summary.columns:
        preds = predicted_summary[predicted_summary["edge_probability"] >= prob_threshold]
    else:
        preds = predicted_summary # assume que já está filtrado
        
    # Converter para sets para facilitar intersecção
    def _to_set(df):
        if df.empty:
            return set()
        return set(tuple(x) for x in df[["source", "target", "lag"]].values)
    
    pred_set = _to_set(preds)
    gt_set = _to_set(ground_truth)
    
    # Cálculo das métricas básicas
    true_positives = len(pred_set.intersection(gt_set))
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # SHD - Structural Hamming Distance em grafo direcionado é FN + FP + Reversed
    # Aqui vamos simplificar que reversed edge tbm é (FP + FN) já embutido.
    shd = false_positives + false_negatives
    
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "structural_hamming_distance": shd
    }

def inject_noise_regime_change(df, index_change=250, noise_multiplier=3.0):
    """
    Injeta uma quebra de regime e ruído artificial no dado para testar robustez.
    """
    df_noisy = df.copy()
    # Aumentar severamente o ruído a partir do indice de quebra
    for col in df_noisy.columns:
        noise = np.random.normal(0, np.std(df_noisy[col]) * noise_multiplier, len(df_noisy) - index_change)
        df_noisy.loc[index_change:, col] += noise
        
    return df_noisy
