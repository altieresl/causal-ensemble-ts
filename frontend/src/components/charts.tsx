import { formatNumber } from "../lib/format";

export interface BarSeries {
  key: string;
  label: string;
  className: string;
}

interface BarChartProps {
  title: string;
  groups: { label: string; values: Record<string, number | null | undefined> }[];
  series: BarSeries[];
  max?: number;
}

const W = 720;
const H = 260;
const PAD = { top: 16, right: 12, bottom: 70, left: 36 };

/** Barras agrupadas em SVG (valores 0..max). Cores por classe CSS (`bar-0..bar-2`), com legenda. */
export function BarChart({ title, groups, series, max = 1 }: BarChartProps) {
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const groupW = innerW / Math.max(groups.length, 1);
  const barW = Math.min(22, (groupW * 0.8) / series.length);
  const y = (value: number) => PAD.top + innerH * (1 - Math.min(Math.max(value / max, 0), 1));

  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(tick * max)} y2={y(tick * max)} className="chart-grid" />
            <text x={PAD.left - 6} y={y(tick * max)} textAnchor="end" dominantBaseline="central" className="chart-tick">
              {formatNumber(tick * max, 2)}
            </text>
          </g>
        ))}
        {groups.map((group, groupIndex) => {
          const start = PAD.left + groupIndex * groupW + (groupW - barW * series.length) / 2;
          return (
            <g key={group.label}>
              {series.map((serie, serieIndex) => {
                const value = group.values[serie.key];
                if (value == null) return null;
                return (
                  <rect
                    key={serie.key}
                    x={start + serieIndex * barW}
                    y={y(value)}
                    width={barW - 2}
                    height={PAD.top + innerH - y(value)}
                    className={serie.className}
                  >
                    <title>{`${group.label} — ${serie.label}: ${formatNumber(value, 3)}`}</title>
                  </rect>
                );
              })}
              <text
                transform={`translate(${PAD.left + groupIndex * groupW + groupW / 2},${H - PAD.bottom + 12}) rotate(35)`}
                className="chart-tick"
              >
                {group.label}
              </text>
            </g>
          );
        })}
      </svg>
      <ul className="legend">
        {series.map((serie) => (
          <li key={serie.key}>
            <span className={`swatch ${serie.className}`} aria-hidden /> {serie.label}
          </li>
        ))}
      </ul>
    </figure>
  );
}

interface StripPlotProps {
  title: string;
  rows: { label: string; values: number[] }[];
}

/** Um ponto por réplica em cada estratégia (0..1), com a média marcada. */
export function StripPlot({ title, rows }: StripPlotProps) {
  const labelW = 150;
  const rowH = 26;
  const width = 640;
  const height = rows.length * rowH + 30;
  const x = (value: number) => labelW + (width - labelW - 16) * Math.min(Math.max(value, 0), 1);
  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} x2={x(tick)} y1={4} y2={rows.length * rowH + 4} className="chart-grid" />
            <text x={x(tick)} y={height - 8} textAnchor="middle" className="chart-tick">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}
        {rows.map((row, index) => {
          const cy = 4 + index * rowH + rowH / 2;
          const mean = row.values.length ? row.values.reduce((a, b) => a + b, 0) / row.values.length : null;
          return (
            <g key={row.label}>
              <text x={labelW - 8} y={cy} textAnchor="end" dominantBaseline="central" className="chart-tick">
                {row.label}
              </text>
              {row.values.map((value, i) => (
                <circle key={i} cx={x(value)} cy={cy} r={4} className="dot" opacity={0.6}>
                  <title>{`${row.label}: ${formatNumber(value, 3)}`}</title>
                </circle>
              ))}
              {mean != null && (
                <line x1={x(mean)} x2={x(mean)} y1={cy - 9} y2={cy + 9} className="mean-mark">
                  <title>{`média ${formatNumber(mean, 3)}`}</title>
                </line>
              )}
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
