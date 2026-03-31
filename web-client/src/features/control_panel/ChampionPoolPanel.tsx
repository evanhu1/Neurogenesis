import type { ChampionPoolEntry } from '../../types';

type ChampionPoolPanelProps = {
  entries: ChampionPoolEntry[];
  onDeleteEntry: (index: number) => void;
  onClearAll: () => void;
};

function formatEnergy(value: number): string {
  return value >= 100 ? Math.round(value).toString() : value.toFixed(1);
}

function formatTimestamp(unixMs: number): string {
  return new Date(unixMs).toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function ChampionPoolPanel({
  entries,
  onDeleteEntry,
  onClearAll,
}: ChampionPoolPanelProps) {
  return (
    <section className="mt-3 rounded-2xl border border-accent/15 bg-slate-100/75 p-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-ink/80">
            Champion Pool
          </h3>
          <p className="mt-1 text-[11px] font-mono text-ink/60">{entries.length} stored genomes</p>
        </div>
        {entries.length > 0 ? (
          <button
            type="button"
            onClick={onClearAll}
            className="rounded-lg border border-rose-300 px-2.5 py-1 font-mono text-[11px] font-medium text-rose-700 transition hover:bg-rose-50"
          >
            Clear All
          </button>
        ) : null}
      </div>

      {entries.length === 0 ? (
        <div className="mt-3 rounded-xl border border-dashed border-accent/15 px-3 py-4 text-center font-mono text-[11px] text-ink/55">
          Pool is empty.
        </div>
      ) : (
        <div className="mt-3 max-h-72 space-y-2 overflow-auto pr-1 scrollbar-none">
          {entries.map((entry, index) => (
            <article
              key={`${entry.source_turn}-${entry.generation}-${entry.reproductions_count}-${index}`}
              className="rounded-xl border border-accent/10 bg-white/80 px-3 py-2"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="font-mono text-xs font-semibold text-ink">
                    #{index + 1} g{entry.generation} e{formatEnergy(entry.energy)}
                  </div>
                  <div className="mt-0.5 font-mono text-[11px] text-ink/55">
                    turn {entry.source_turn} · {formatTimestamp(entry.source_created_at_unix_ms)}
                  </div>
                </div>
                <div className="shrink-0 rounded-full bg-accent/10 px-2 py-0.5 font-mono text-[10px] text-ink/70">
                  {entry.genome.num_neurons}n {entry.genome.num_synapses}s
                </div>
              </div>

              <div className="mt-2 grid grid-cols-3 gap-2 font-mono text-[10px] uppercase tracking-[0.12em] text-ink/58">
                <div>
                  <div className="text-[9px] text-ink/45">Meals</div>
                  <div className="mt-0.5 text-[11px] text-ink">{entry.consumptions_count}</div>
                </div>
                <div>
                  <div className="text-[9px] text-ink/45">Offspring</div>
                  <div className="mt-0.5 text-[11px] text-ink">{entry.reproductions_count}</div>
                </div>
                <div>
                  <div className="text-[9px] text-ink/45">Vision</div>
                  <div className="mt-0.5 text-[11px] text-ink">{entry.genome.vision_distance}</div>
                </div>
              </div>

              <div className="mt-2 flex justify-end">
                <button
                  type="button"
                  onClick={() => onDeleteEntry(index)}
                  className="rounded-md border border-rose-300 px-2 py-1 font-mono text-[10px] font-medium text-rose-700 transition hover:bg-rose-50"
                >
                  Delete
                </button>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
