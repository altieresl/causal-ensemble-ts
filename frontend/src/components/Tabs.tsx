import { useId, useRef, type KeyboardEvent, type ReactNode } from "react";

interface TabsProps<T extends string> {
  tabs: readonly T[];
  value: T;
  onChange: (tab: T) => void;
  label: string;
  children: ReactNode;
}

/** Abas acessíveis (WAI-ARIA): setas, Home/End e foco gerenciado. */
export function Tabs<T extends string>({ tabs, value, onChange, label, children }: TabsProps<T>) {
  const base = useId();
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});

  const onKeyDown = (event: KeyboardEvent) => {
    const index = tabs.indexOf(value);
    let next = index;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    else return;
    event.preventDefault();
    onChange(tabs[next]);
    refs.current[tabs[next]]?.focus();
  };

  return (
    <div className="stack">
      <div role="tablist" aria-label={label} className="tabs" onKeyDown={onKeyDown}>
        {tabs.map((tab) => (
          <button
            key={tab}
            ref={(element) => {
              refs.current[tab] = element;
            }}
            role="tab"
            id={`${base}-${tab}`}
            aria-selected={value === tab}
            aria-controls={`${base}-panel`}
            tabIndex={value === tab ? 0 : -1}
            className={value === tab ? "tab active" : "tab"}
            onClick={() => onChange(tab)}
          >
            {tab}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`${base}-panel`} aria-labelledby={`${base}-${value}`} className="stack">
        {children}
      </div>
    </div>
  );
}
