import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

/**
 * Perfil de experiência do usuário:
 * - iniciante: menos controles (valores padrão do notebook), mais explicações em linguagem simples;
 * - avançado: todos os parâmetros, abas e números brutos.
 * Não muda nenhum cálculo — só o que é mostrado e o que pode ser ajustado.
 */
export type ExperienceLevel = "beginner" | "advanced";

const STORAGE_KEY = "causal-discovery-ts.experience";

function readStored(): ExperienceLevel | null {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value === "beginner" || value === "advanced" ? value : null;
  } catch {
    return null; // armazenamento bloqueado (modo privado etc.): segue sem lembrar
  }
}

interface ExperienceApi {
  level: ExperienceLevel;
  /** O usuário já escolheu explicitamente? (senão mostramos as boas-vindas) */
  chosen: boolean;
  isBeginner: boolean;
  setLevel: (level: ExperienceLevel) => void;
}

const ExperienceContext = createContext<ExperienceApi>({
  level: "advanced",
  chosen: true,
  isBeginner: false,
  setLevel: () => undefined,
});

export const useExperience = () => useContext(ExperienceContext);

export function ExperienceProvider({ children, initial }: { children: ReactNode; initial?: ExperienceLevel }) {
  const [stored, setStored] = useState<ExperienceLevel | null>(() => initial ?? readStored());
  const setLevel = useCallback((level: ExperienceLevel) => {
    setStored(level);
    try {
      window.localStorage.setItem(STORAGE_KEY, level);
    } catch {
      /* sem persistência: vale só nesta sessão */
    }
  }, []);
  const api = useMemo<ExperienceApi>(() => {
    const level = stored ?? "beginner"; // até escolher, a interface começa simples
    return { level, chosen: stored !== null, isBeginner: level === "beginner", setLevel };
  }, [stored, setLevel]);
  return <ExperienceContext.Provider value={api}>{children}</ExperienceContext.Provider>;
}

/** Só renderiza no perfil avançado. */
export function AdvancedOnly({ children }: { children: ReactNode }) {
  return useExperience().isBeginner ? null : <>{children}</>;
}

/** Explicação extra exibida só para iniciantes. */
export function BeginnerHint({ children }: { children: ReactNode }) {
  return useExperience().isBeginner ? <p className="hint">{children}</p> : null;
}

export function ExperienceToggle() {
  const { level, setLevel } = useExperience();
  return (
    <div className="segmented small" role="group" aria-label="Perfil de uso">
      <button type="button" aria-pressed={level === "beginner"} onClick={() => setLevel("beginner")} title="Menos opções, mais explicações">
        Iniciante
      </button>
      <button type="button" aria-pressed={level === "advanced"} onClick={() => setLevel("advanced")} title="Todos os parâmetros e detalhes">
        Avançado
      </button>
    </div>
  );
}
