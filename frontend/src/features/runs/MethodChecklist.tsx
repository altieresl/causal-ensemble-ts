import { useEffect } from "react";

import { useMethods } from "../../api/hooks";
import { ErrorBox, Spinner } from "../../components/ui";
import { useExperience } from "../../lib/experience";

interface Props {
  value: string[];
  onChange: (methods: string[]) => void;
  minimum?: number;
}

/** Seleção de métodos candidatos; começa com todos os métodos registrados no backend. */
export function MethodChecklist({ value, onChange, minimum = 2 }: Props) {
  const methods = useMethods();
  const { isBeginner } = useExperience();
  useEffect(() => {
    if (methods.data && value.length === 0) onChange(methods.data.map((m) => m.name));
    // inicializa uma única vez, quando a lista chega
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [methods.data]);

  if (methods.isPending) return <Spinner />;
  if (methods.isError) return <ErrorBox error={methods.error} />;
  if (isBeginner) {
    return (
      <p className="muted">
        Todos os {methods.data.length} métodos disponíveis entram como candidatos; a análise escolhe a melhor combinação.
      </p>
    );
  }
  return (
    <>
      <div className="checks">
        {methods.data.map((m) => (
          <label key={m.name} className="check">
            <input
              type="checkbox"
              checked={value.includes(m.name)}
              onChange={() => onChange(value.includes(m.name) ? value.filter((x) => x !== m.name) : [...value, m.name])}
            />
            {m.name}
          </label>
        ))}
      </div>
      {value.length < minimum && <p className="error">Selecione ao menos {minimum} métodos.</p>}
    </>
  );
}
