import { useMemo } from 'react';

type SessionOverviewPanelProps = {
  sessionMeta: string;
  metricsText: string;
};

export function SessionOverviewPanel({
  sessionMeta,
  metricsText,
}: SessionOverviewPanelProps) {
  const runtimeStatsText = useMemo(() => {
    const visibleMetricPrefixes = ['turn=', 'organisms=', 'species_alive='];
    const filtered = metricsText
      .split('\n')
      .filter((line) => visibleMetricPrefixes.some((prefix) => line.startsWith(prefix)));
    return filtered.length > 0 ? filtered.join('\n') : 'No metrics';
  }, [metricsText]);

  return (
    <>
      <h1 className="text-2xl font-semibold tracking-tight">Neurogenesis</h1>
      {sessionMeta ? (
        <div className="mt-2 text-[11px] font-mono text-ink/65">{sessionMeta}</div>
      ) : null}
      <pre className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-100/80 p-3 font-mono text-xs">
        {runtimeStatsText}
      </pre>
    </>
  );
}
