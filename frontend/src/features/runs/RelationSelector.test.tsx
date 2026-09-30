import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";

import { relationsFor, type Objective, type Relation } from "../../lib/objective";
import { RelationSelector } from "./RelationSelector";

const nodes = ["A", "B", "C"];

function Harness({ initial, onCount }: { initial: Objective; onCount?: (n: number) => void }) {
  const [objective, setObjective] = useState(initial);
  const [specific, setSpecific] = useState<Relation[]>([]);
  const relations = relationsFor(objective, nodes, specific);
  onCount?.(relations.length);
  return (
    <RelationSelector
      nodes={nodes}
      objective={objective}
      onObjectiveChange={setObjective}
      specific={specific}
      onSpecificChange={setSpecific}
      relationCount={relations.length}
    />
  );
}

const full: Objective = { type: "full_structure", primary_variable: null, secondary_variable: null };

describe("RelationSelector", () => {
  it("estrutura geral define 6 relações e não mostra a grade", () => {
    render(<Harness initial={full} />);
    expect(screen.getByText(/6/)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("'causas de uma variável' pede a variável principal e reduz as relações", () => {
    render(<Harness initial={full} />);
    fireEvent.change(screen.getByLabelText("Objetivo"), { target: { value: "causes_of_target" } });
    expect(screen.getByLabelText("Variável principal")).toHaveValue("A");
    expect(screen.getByText(/Investiga quais variáveis podem anteceder/)).toBeInTheDocument();
    expect(screen.getByText("2", { selector: "strong" })).toBeInTheDocument();
  });

  it("relações específicas: grade começa vazia, avisa e permite marcar pares", () => {
    render(<Harness initial={{ ...full, type: "specific_relations" }} />);
    expect(screen.getByText("Selecione ao menos 1 relação para analisar.")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("A → B"));
    expect(screen.queryByText("Selecione ao menos 1 relação para analisar.")).not.toBeInTheDocument();
    expect(screen.getByLabelText("A → B")).toBeChecked();
    fireEvent.click(screen.getByText("Selecionar todas"));
    expect(screen.getByLabelText("C → A")).toBeChecked();
    fireEvent.click(screen.getByText("Limpar seleção"));
    expect(screen.getByLabelText("C → A")).not.toBeChecked();
  });
});
