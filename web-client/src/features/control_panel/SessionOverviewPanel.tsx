import { useMemo } from 'react';

type SessionOverviewPanelProps = {
  metricsText: string;
};

export function SessionOverviewPanel({ metricsText }: SessionOverviewPanelProps) {
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
      <pre className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-100/80 p-3 font-mono text-xs">
        {runtimeStatsText}
      </pre>
    </>
  );
}
