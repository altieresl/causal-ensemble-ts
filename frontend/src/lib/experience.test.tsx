import { fireEvent, render, screen } from "@testing-library/react";

import { WelcomeProfile } from "../components/WelcomeProfile";
import { AdvancedOnly, BeginnerHint, ExperienceProvider, ExperienceToggle } from "./experience";

const KEY = "causal-discovery-ts.experience";

function Page() {
  return (
    <ExperienceProvider>
      <ExperienceToggle />
      <WelcomeProfile />
      <BeginnerHint>dica de iniciante</BeginnerHint>
      <AdvancedOnly>
        <p>controle avançado</p>
      </AdvancedOnly>
    </ExperienceProvider>
  );
}

describe("perfil de experiência", () => {
  beforeEach(() => window.localStorage.clear());

  it("na primeira visita pergunta o perfil e começa simples", () => {
    render(<Page />);
    expect(screen.getByText("Como você prefere usar a ferramenta?")).toBeInTheDocument();
    expect(screen.getByText("dica de iniciante")).toBeInTheDocument();
    expect(screen.queryByText("controle avançado")).not.toBeInTheDocument();
  });

  it("escolher avançado mostra os controles, some com as dicas e é lembrado", () => {
    const { unmount } = render(<Page />);
    fireEvent.click(screen.getByRole("button", { name: /^Avançado\s*Todos os parâmetros/ }));
    expect(screen.queryByText("Como você prefere usar a ferramenta?")).not.toBeInTheDocument();
    expect(screen.getByText("controle avançado")).toBeInTheDocument();
    expect(screen.queryByText("dica de iniciante")).not.toBeInTheDocument();
    expect(window.localStorage.getItem(KEY)).toBe("advanced");
    unmount();

    render(<Page />);
    expect(screen.getByRole("button", { name: "Avançado" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("controle avançado")).toBeInTheDocument();
  });

  it("o seletor da barra alterna entre os perfis", () => {
    window.localStorage.setItem(KEY, "advanced");
    render(<Page />);
    fireEvent.click(screen.getByRole("button", { name: "Iniciante" }));
    expect(screen.getByText("dica de iniciante")).toBeInTheDocument();
    expect(window.localStorage.getItem(KEY)).toBe("beginner");
  });

  it("funciona mesmo se o armazenamento do navegador falhar", () => {
    const spy = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("bloqueado");
    });
    render(<Page />);
    expect(screen.getByText("dica de iniciante")).toBeInTheDocument();
    spy.mockRestore();
  });
});
