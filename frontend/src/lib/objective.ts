// Objetivos de análise (portados do dashboard do notebook): cada objetivo define quais relações
// direcionais (origem → destino) o pipeline avalia.

export type ObjectiveType =
  | "full_structure"
  | "causes_of_target"
  | "effects_of_source"
  | "compare_directions"
  | "specific_relations";

export type Relation = [string, string];

export type Objective = {
  type: ObjectiveType;
  primary_variable: string | null;
  secondary_variable: string | null;
};

export const OBJECTIVE_LABELS: Record<ObjectiveType, string> = {
  full_structure: "Explorar estrutura geral",
  causes_of_target: "Investigar causas de uma variável",
  effects_of_source: "Investigar efeitos de uma variável",
  compare_directions: "Comparar as duas direções",
  specific_relations: "Escolher relações específicas",
};

export const needsPrimary = (type: ObjectiveType) =>
  type === "causes_of_target" || type === "effects_of_source" || type === "compare_directions";

export const relationKey = ([source, target]: Relation) => `${source}→${target}`;

export const allRelations = (nodes: string[]): Relation[] =>
  nodes.flatMap((source) => nodes.filter((target) => target !== source).map((target): Relation => [source, target]));

export function relationsFor(objective: Objective, nodes: string[], specific: Relation[]): Relation[] {
  const { primary_variable: primary, secondary_variable: secondary } = objective;
  switch (objective.type) {
    case "full_structure":
      return allRelations(nodes);
    case "causes_of_target":
      return primary ? nodes.filter((n) => n !== primary).map((n): Relation => [n, primary]) : [];
    case "effects_of_source":
      return primary ? nodes.filter((n) => n !== primary).map((n): Relation => [primary, n]) : [];
    case "compare_directions":
      return primary && secondary && primary !== secondary ? [[primary, secondary], [secondary, primary]] : [];
    case "specific_relations": {
      const valid = new Set(allRelations(nodes).map(relationKey));
      return specific.filter((relation) => valid.has(relationKey(relation)));
    }
  }
}

export function describeObjective(objective: Objective): string {
  const { primary_variable: p, secondary_variable: s } = objective;
  switch (objective.type) {
    case "full_structure":
      return "Todas as relações entre variáveis diferentes serão avaliadas.";
    case "causes_of_target":
      return `Investiga quais variáveis podem anteceder ou influenciar ${p ?? "…"}.`;
    case "effects_of_source":
      return `Investiga quais variáveis podem ser influenciadas por ${p ?? "…"}.`;
    case "compare_directions":
      return `Compara ${p ?? "…"} → ${s ?? "…"} e ${s ?? "…"} → ${p ?? "…"}.`;
    case "specific_relations":
      return "Escolha manualmente os pares direcionais na grade abaixo.";
  }
}
