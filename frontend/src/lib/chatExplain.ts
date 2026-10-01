import type { ChatDecision } from "../api/types";

export interface AxisReading {
  key: "stationarity" | "linearity" | "non_gaussian";
  label: string;
  pct: number | null; // % de séries do dataset com a propriedade, segundo o chat
  datasetMajority: boolean | null; // o dataset tem a propriedade em ao menos metade das séries?
  required: boolean | null; // o algoritmo exige a propriedade?
  violated: boolean | null; // exige e o dataset não tem (null = não dá para dizer)
}

export interface DecisionExplanation {
  axes: AxisReading[];
  violated: string[];
  /** Frase que liga a leitura às premissas e ao voto, escrita a partir dos booleanos declarados pelo chat. */
  verdict: string;
}

const AXES: { key: AxisReading["key"]; label: string; noun: string }[] = [
  { key: "stationarity", label: "Estacionariedade", noun: "estacionariedade" },
  { key: "linearity", label: "Linearidade", noun: "linearidade" },
  { key: "non_gaussian", label: "Erros não gaussianos", noun: "erros não gaussianos" },
];

function reading(decision: ChatDecision, key: AxisReading["key"], label: string): AxisReading {
  const pct = {
    stationarity: decision.stationary_fraction_pct,
    linearity: decision.linear_fraction_pct,
    non_gaussian: decision.non_gaussian_fraction_pct,
  }[key];
  const majority = {
    stationarity: decision.dataset_is_majority_stationary,
    linearity: decision.dataset_is_majority_linear,
    non_gaussian: decision.dataset_is_majority_non_gaussian,
  }[key];
  const required = {
    stationarity: decision.algorithm_requires_stationarity,
    linearity: decision.algorithm_requires_linearity,
    non_gaussian: decision.algorithm_requires_non_gaussian_errors,
  }[key];
  const violated = required == null || majority == null ? null : required && !majority;
  return { key, label, pct: pct ?? null, datasetMajority: majority ?? null, required: required ?? null, violated };
}

/** Reconstrói o raciocínio do voto: leitura do perfil × premissas exigidas → decisão. */
export function explainDecision(decision: ChatDecision): DecisionExplanation {
  const axes = AXES.map(({ key, label }) => reading(decision, key, label));
  const violated = axes.filter((axis) => axis.violated).map((axis) => axis.label.toLowerCase());
  const unknown = axes.some((axis) => axis.violated === null);

  let verdict: string;
  if (violated.length > 0) {
    const details = AXES.filter((_, i) => axes[i].violated).map(
      ({ noun }, _i) => `exige ${noun} e o dataset não a tem na maioria das séries`,
    );
    verdict = `Votou ${decision.include ? "incluir" : "excluir"}. O chat leu que o algoritmo ${details.join("; e ")}.`;
  } else if (unknown) {
    verdict = `Votou ${decision.include ? "incluir" : "excluir"}, mas o chat não declarou todos os campos da leitura; a derivação abaixo é parcial.`;
  } else {
    verdict = `Votou ${decision.include ? "incluir" : "excluir"}: nenhuma premissa exigida pelo algoritmo é violada pelo perfil do dataset.`;
  }
  if (decision.synthesis_corrected) {
    verdict += " A decisão final foi recalculada em Python a partir dos próprios booleanos do chat, porque a síntese dele os contradizia.";
  }
  return { axes, violated, verdict };
}
