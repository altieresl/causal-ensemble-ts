import { allRelations, relationsFor, type Objective } from "./objective";

const nodes = ["A", "B", "C"];
const objective = (over: Partial<Objective>): Objective => ({
  type: "full_structure", primary_variable: null, secondary_variable: null, ...over,
});

describe("relationsFor", () => {
  it("estrutura geral = todos os pares direcionados distintos", () => {
    expect(relationsFor(objective({}), nodes, [])).toHaveLength(6);
    expect(allRelations(["A"])).toEqual([]);
  });
  it("causas de um alvo: todas as demais → alvo", () => {
    expect(relationsFor(objective({ type: "causes_of_target", primary_variable: "B" }), nodes, [])).toEqual([
      ["A", "B"], ["C", "B"],
    ]);
  });
  it("efeitos de uma fonte: fonte → todas as demais", () => {
    expect(relationsFor(objective({ type: "effects_of_source", primary_variable: "B" }), nodes, [])).toEqual([
      ["B", "A"], ["B", "C"],
    ]);
  });
  it("comparar direções gera as duas direções; iguais não geram nada", () => {
    const type = "compare_directions" as const;
    expect(relationsFor(objective({ type, primary_variable: "A", secondary_variable: "B" }), nodes, [])).toEqual([
      ["A", "B"], ["B", "A"],
    ]);
    expect(relationsFor(objective({ type, primary_variable: "A", secondary_variable: "A" }), nodes, [])).toEqual([]);
  });
  it("relações específicas descartam pares inválidos (autoaresta/coluna desconhecida)", () => {
    const specific: [string, string][] = [["A", "B"], ["A", "A"], ["Z", "A"]];
    expect(relationsFor(objective({ type: "specific_relations" }), nodes, specific)).toEqual([["A", "B"]]);
  });
  it("sem variável principal não há relações", () => {
    expect(relationsFor(objective({ type: "causes_of_target" }), nodes, [])).toEqual([]);
  });
});
