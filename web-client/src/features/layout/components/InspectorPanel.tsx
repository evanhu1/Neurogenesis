import type { BrainState } from '../../../types';
import { BrainCanvas } from '../../inspector/components/BrainCanvas';

type InspectorPanelProps = {
  focusMetaText: string;
  focusedStatsText: string;
  focusedBrain: BrainState | null;
  activeNeuronIds: Set<number> | null;
  onDefocus: () => void;
};

export function InspectorPanel({ focusMetaText, focusedStatsText, focusedBrain, activeNeuronIds, onDefocus }: InspectorPanelProps) {
  return (
    <aside className="h-full overflow-auto rounded-2xl border border-accent/15 bg-panel/95 p-4 shadow-panel">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold tracking-tight">Organism Inspector</h2>
        <button
          onClick={onDefocus}
          className="rounded p-1 text-ink/40 transition hover:bg-slate-200 hover:text-ink/80"
          aria-label="Close inspector"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5">
            <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
          </svg>
        </button>
      </div>
      <div className="mt-2 rounded-xl bg-slate-100/80 p-3 font-mono text-xs">{focusMetaText}</div>
      <section className="mt-3 rounded-xl border border-accent/15 bg-white/80 p-3">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-ink/80">Stats</h3>
        <pre className="mt-2 whitespace-pre-wrap rounded-lg bg-slate-100/80 p-3 font-mono text-xs">
          {focusedStatsText}
        </pre>
      </section>
      <section className="mt-3 rounded-xl border border-accent/15 bg-white/80 p-3">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-ink/80">Brain</h3>
        <BrainCanvas focusedBrain={focusedBrain} activeNeuronIds={activeNeuronIds} />
      </section>
    </aside>
  );
}

