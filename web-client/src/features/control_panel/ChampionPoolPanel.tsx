import type { ChampionPoolEntry } from '../../types';

type ChampionPoolPanelProps = {
  entries: ChampionPoolEntry[];
  onDeleteEntry: (index: number) => void;
  onClearAll: () => void;
};

function formatEnergy(value: number): string {
  return value >= 100 ? Math.round(value).toString() : value.toFixed(1);
}

export function ChampionPoolPanel({
  entries,
  onDeleteEntry,
  onClearAll,
}: ChampionPoolPanelProps) {
  return (
    <section className="mt-3">
      <div className="flex items-center justify-between">
        <h3 className="text-[10px] font-semibold uppercase tracking-[0.16em] text-ink/40">
          Champions
          <span className="ml-1.5 font-mono text-ink/25">{entries.length}</span>
        </h3>
        {entries.length > 0 && (
          <button
            type="button"
            onClick={onClearAll}
            className="font-mono text-[10px] text-rose-400/60 transition hover:text-rose-400"
          >
            Clear
          </button>
        )}
      </div>

      {entries.length === 0 ? (
        <p className="mt-2 text-center font-mono text-[10px] text-ink/20">
          No champions saved
        </p>
      ) : (
        <div className="mt-1 max-h-44 space-y-px overflow-auto scrollbar-none">
          {entries.map((entry, index) => (
            <div
              key={`${entry.source_turn}-${entry.generation}-${index}`}
              className="group flex items-center gap-2 rounded px-1.5 py-1 transition hover:bg-surface"
            >
              <span className="font-mono text-[10px] text-ink/25">{index + 1}</span>
              <span className="font-mono text-[11px] text-ink/70">g{entry.generation}</span>
              <span className="font-mono text-[11px] text-accent/60">
                {formatEnergy(entry.energy)}e
              </span>
              <span className="font-mono text-[10px] text-ink/25">
                {entry.genome.num_neurons}n/{entry.genome.num_synapses}s
              </span>
              <button
                type="button"
                onClick={() => onDeleteEntry(index)}
                className="ml-auto text-ink/15 opacity-0 transition hover:text-rose-400 group-hover:opacity-100"
                aria-label="Delete champion"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 16 16"
                  fill="currentColor"
                  className="h-3 w-3"
                >
                  <path d="M5.28 4.22a.75.75 0 0 0-1.06 1.06L6.94 8l-2.72 2.72a.75.75 0 1 0 1.06 1.06L8 9.06l2.72 2.72a.75.75 0 1 0 1.06-1.06L9.06 8l2.72-2.72a.75.75 0 0 0-1.06-1.06L8 6.94 5.28 4.22Z" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
