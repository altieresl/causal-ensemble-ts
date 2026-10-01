import { act, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";

import { Tabs } from "./Tabs";
import { ConfirmButton, ProgressBar, ValueBar } from "./ui";

describe("ConfirmButton", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("exige dois cliques e desarma sozinho", () => {
    const onConfirm = vi.fn();
    render(<ConfirmButton onConfirm={onConfirm}>Remover</ConfirmButton>);
    fireEvent.click(screen.getByRole("button", { name: "Remover" }));
    expect(onConfirm).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Confirmar?" })).toBeInTheDocument();

    act(() => vi.advanceTimersByTime(4100));
    expect(screen.getByRole("button", { name: "Remover" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Remover" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirmar?" }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });
});

describe("Tabs", () => {
  const TABS = ["A", "B", "C"] as const;
  function Harness() {
    const [tab, setTab] = useState<(typeof TABS)[number]>("A");
    return (
      <Tabs tabs={TABS} value={tab} onChange={setTab} label="Teste">
        <p>conteúdo {tab}</p>
      </Tabs>
    );
  }

  it("navega com setas, Home e End e mantém o foco na aba ativa", () => {
    render(<Harness />);
    const list = screen.getByRole("tablist", { name: "Teste" });
    fireEvent.keyDown(list, { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "B" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("conteúdo B")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "B" })).toHaveFocus();
    fireEvent.keyDown(list, { key: "End" });
    expect(screen.getByRole("tab", { name: "C" })).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(list, { key: "ArrowRight" }); // volta ao início
    expect(screen.getByRole("tab", { name: "A" })).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(list, { key: "ArrowLeft" }); // dá a volta para o fim
    expect(screen.getByRole("tab", { name: "C" })).toHaveAttribute("aria-selected", "true");
  });
});

describe("ProgressBar / ValueBar", () => {
  it("é determinada com total e indeterminada sem", () => {
    const { rerender } = render(<ProgressBar done={1} total={4} label="p" />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "25");
    rerender(<ProgressBar label="p" />);
    expect(screen.getByRole("progressbar")).not.toHaveAttribute("aria-valuenow");
  });

  it("ValueBar limita o preenchimento a 0–100% e trata ausente", () => {
    const { container, rerender } = render(<ValueBar value={1.7} />);
    expect(container.querySelector<HTMLElement>(".bar-fill")?.style.width).toBe("100%");
    rerender(<ValueBar value={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});
