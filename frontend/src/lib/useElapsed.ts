import { useEffect, useState } from "react";

/** Segundos decorridos desde `start` (ISO), atualizando a cada segundo enquanto `running`. */
export function useElapsedSeconds(start: string | null, running: boolean): number | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running]);
  return start ? Math.max(0, Math.round((now - Date.parse(start)) / 1000)) : null;
}

export function formatElapsed(seconds: number | null): string {
  if (seconds == null) return "—";
  const minutes = Math.floor(seconds / 60);
  return minutes > 0 ? `${minutes}min ${String(seconds % 60).padStart(2, "0")}s` : `${seconds}s`;
}
