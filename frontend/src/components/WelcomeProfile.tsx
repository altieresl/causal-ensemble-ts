import { useExperience } from "../lib/experience";

/** Primeira visita: o usuário escolhe como quer usar a ferramenta (pode trocar depois na barra superior). */
export function WelcomeProfile() {
  const { chosen, setLevel } = useExperience();
  if (chosen) return null;
  return (
    <section className="card welcome" aria-labelledby="welcome-title">
      <h2 id="welcome-title">Como você prefere usar a ferramenta?</h2>
      <p className="muted">Dá para trocar a qualquer momento no seletor “Iniciante / Avançado” da barra superior.</p>
      <div className="grid">
        <button type="button" className="choice" onClick={() => setLevel("beginner")}>
          <strong>Iniciante</strong>
          <span>
            Poucos controles, com os valores padrão já testados. Explicações em linguagem simples sobre o que cada
            resultado significa.
          </span>
        </button>
        <button type="button" className="choice" onClick={() => setLevel("advanced")}>
          <strong>Avançado</strong>
          <span>
            Todos os parâmetros (lags, bootstraps, limiares, paralelismo, conhecimento especialista) e todas as abas de
            resultado com os números brutos.
          </span>
        </button>
      </div>
    </section>
  );
}
