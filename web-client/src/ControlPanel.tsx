import type { World } from './useWorld';

function Sparkline({
  points,
  pick,
  color,
  height = 44,
}: {
  points: { turn: number }[];
  pick: (p: any) => number;
  color: string;
  height?: number;
}) {
  if (points.length < 2) return <div style={{ height }} />;
  const vals = points.map(pick);
  const max = Math.max(1, ...vals);
  const w = 280;
  const step = w / (points.length - 1);
  const path = vals
    .map((v, i) => `${i === 0 ? 'M' : 'L'} ${(i * step).toFixed(1)} ${(height - (v / max) * height).toFixed(1)}`)
    .join(' ');
  return (
    <svg width="100%" viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" style={{ height }}>
      <path d={path} fill="none" stroke={color} strokeWidth={1.5} />
    </svg>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-line bg-void px-3 py-2">
      <div className="font-mono text-[10px] uppercase tracking-wide text-ink/50">{label}</div>
      <div className="mt-0.5 text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}

export function ControlPanel({ world }: { world: World }) {
  const { stats, running, control, history, error } = world;
  const btn =
    'flex-1 rounded-lg border border-line px-3 py-2 text-sm font-medium transition hover:bg-surface';

  return (
    <aside className="flex h-full flex-col gap-3 overflow-y-auto rounded-2xl border border-line bg-panel p-4 shadow-panel scrollbar-none">
      <div>
        <h1 className="text-lg font-bold">NeuroGenesis</h1>
        <p className="text-[12px] text-ink/60">
          indirectly-encoded evolutionary substrate · hex world
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-[12px] text-red-700">
          server unreachable — start it with <code>cargo run -p sim-server</code>
        </div>
      )}

      <div className="flex gap-2">
        <button className={btn} onClick={() => control(running ? 'pause' : 'play')}>
          {running ? '⏸ Pause' : '▶ Play'}
        </button>
        <button className={btn} onClick={() => control('step')} disabled={running}>
          ⏭ Step
        </button>
      </div>

      {stats?.extinct_at != null && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-[12px] text-amber-800">
          ☠ Extinct at turn {stats.extinct_at}. No periodic injection — a dead world stays dead.
        </div>
      )}

      <div className="grid grid-cols-2 gap-2">
        <Metric label="turn" value={stats?.turn ?? '—'} />
        <Metric label="alive" value={stats?.alive ?? '—'} />
        <Metric label="born" value={stats?.total_ever ?? '—'} />
        <Metric label="max gen" value={stats?.max_generation ?? '—'} />
        <Metric label="mean neurons" value={stats ? stats.mean_neurons.toFixed(1) : '—'} />
        <Metric label="mean edges" value={stats ? stats.mean_edges.toFixed(1) : '—'} />
      </div>

      <div>
        <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-ink/50">
          population
        </div>
        <Sparkline points={history} pick={(p) => p.alive} color="#15803d" />
      </div>
      <div>
        <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-ink/50">
          food
        </div>
        <Sparkline points={history} pick={(p) => p.food} color="#3fae4f" height={32} />
      </div>

      <div className="mt-auto text-[11px] text-ink/45">
        Mean energy {stats ? stats.mean_energy.toFixed(0) : '—'} · mean gen{' '}
        {stats ? stats.mean_generation.toFixed(1) : '—'}
      </div>
    </aside>
  );
}
