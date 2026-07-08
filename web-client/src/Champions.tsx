import type { Champions as ChampionsData } from './api';

// The Quality-Diversity champion archive (MAP-Elites): one elite genome per
// behavioral niche. `descriptor` is [brain complexity, generational depth,
// energy fraction]; coverage = occupied niches, qd_score = summed quality.
export function Champions({ data }: { data: ChampionsData | null }) {
  if (!data)
    return <div className="p-4 text-[13px] text-ink/45">Connecting to the champion archive…</div>;

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-4 scrollbar-none">
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded-lg border border-line bg-void px-2 py-1.5 text-center">
          <div className="font-mono text-[10px] uppercase text-ink/50">schema</div>
          <div className="font-semibold">v{data.schema_version}</div>
        </div>
        <div className="rounded-lg border border-line bg-void px-2 py-1.5 text-center">
          <div className="font-mono text-[10px] uppercase text-ink/50">coverage</div>
          <div className="font-semibold">{data.coverage}</div>
        </div>
        <div className="rounded-lg border border-line bg-void px-2 py-1.5 text-center">
          <div className="font-mono text-[10px] uppercase text-ink/50">QD-score</div>
          <div className="font-semibold">{data.qd_score.toFixed(1)}</div>
        </div>
      </div>

      {data.entries.length === 0 ? (
        <div className="text-[13px] text-ink/45">
          No champions yet. Select an organism and “Save to QD champion archive”.
        </div>
      ) : (
        <ul className="flex flex-col gap-1">
          {[...data.entries]
            .sort((a, b) => b.quality - a.quality)
            .slice(0, 40)
            .map((c, i) => (
              <li
                key={i}
                className="flex items-center justify-between rounded-lg border border-line bg-void px-3 py-1.5 text-[12px]"
              >
                <span className="font-mono">
                  niche [{c.descriptor.values.map((v) => v.toFixed(2)).join(', ')}]
                </span>
                <span className="tabular-nums font-semibold text-accent">{c.quality.toFixed(2)}</span>
              </li>
            ))}
        </ul>
      )}
    </div>
  );
}
