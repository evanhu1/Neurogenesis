import { useMemo } from 'react';

type SessionOverviewPanelProps = {
  metricsText: string;
};

type MetricItem = { label: string; value: string };

export function SessionOverviewPanel({ metricsText }: SessionOverviewPanelProps) {
  const metrics = useMemo((): MetricItem[] => {
    const mapping: Record<string, string> = {
      turn: 'Turn',
      organisms: 'Organisms',
      species_alive: 'Species',
    };
    const items: MetricItem[] = [];
    for (const line of metricsText.split('\n')) {
      const eqIdx = line.indexOf('=');
      if (eqIdx === -1) continue;
      const key = line.slice(0, eqIdx);
      const label = mapping[key];
      if (label) items.push({ label, value: line.slice(eqIdx + 1) });
    }
    return items;
  }, [metricsText]);

  return (
    <div className="mb-2">
      <h1 className="text-xs font-semibold uppercase tracking-[0.2em] text-accent/80">
        NeuroGenesis
      </h1>
      <div className="mt-1 flex gap-4 font-mono text-[11px]">
        {metrics.map((m) => (
          <div key={m.label}>
            <span className="text-ink/35">{m.label}</span>{' '}
            <span className="text-ink/80">{m.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
