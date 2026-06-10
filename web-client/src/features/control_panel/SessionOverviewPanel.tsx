import type { LiveMetricsData } from '../../types';

type SessionOverviewPanelProps = {
  liveMetrics: LiveMetricsData | null;
  isRunning: boolean;
  isFastMode: boolean;
};

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/5 bg-surface/50 px-2 py-1.5">
      <div className="text-[9px] font-semibold uppercase tracking-[0.14em] text-ink/35">
        {label}
      </div>
      <div className="mt-0.5 truncate font-mono text-[13px] text-ink/90">{value}</div>
    </div>
  );
}

export function SessionOverviewPanel({
  liveMetrics,
  isRunning,
  isFastMode,
}: SessionOverviewPanelProps) {
  const status = isFastMode
    ? { label: 'Fast', dotClass: 'bg-amber-400', textClass: 'text-amber-300/90', pulse: true }
    : isRunning
      ? { label: 'Live', dotClass: 'bg-emerald-400', textClass: 'text-emerald-300/90', pulse: true }
      : { label: 'Paused', dotClass: 'bg-slate-500', textClass: 'text-ink/40', pulse: false };

  return (
    <div>
      <div className="flex items-center justify-between">
        <h1 className="text-[13px] font-bold tracking-[0.24em] text-ink">
          NEURO<span className="text-accent">GENESIS</span>
        </h1>
        <span
          className={`flex items-center gap-1.5 rounded-full border border-white/5 bg-surface/50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${status.textClass}`}
        >
          <span className="relative flex h-1.5 w-1.5">
            {status.pulse && (
              <span
                className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${status.dotClass}`}
              />
            )}
            <span className={`relative inline-flex h-1.5 w-1.5 rounded-full ${status.dotClass}`} />
          </span>
          {status.label}
        </span>
      </div>

      <div className="mt-2.5 grid grid-cols-3 gap-1.5">
        <StatTile label="Turn" value={liveMetrics ? liveMetrics.turn.toLocaleString() : '—'} />
        <StatTile
          label="Organisms"
          value={liveMetrics ? String(liveMetrics.metrics.organisms) : '—'}
        />
        <StatTile
          label="Species"
          value={
            liveMetrics ? String(Object.keys(liveMetrics.metrics.species_counts).length) : '—'
          }
        />
      </div>
    </div>
  );
}
