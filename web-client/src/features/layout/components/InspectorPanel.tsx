import type { BrainState } from '../../../types';
import { BrainCanvas } from '../../inspector/components/BrainCanvas';

type InspectorPanelProps = {
  focusMetaText: string;
  focusedStatsText: string;
  focusedBrain: BrainState | null;
};

export function InspectorPanel({ focusMetaText, focusedStatsText, focusedBrain }: InspectorPanelProps) {
  return (
    <aside className="h-full overflow-auto rounded-2xl border border-accent/15 bg-panel/95 p-4 shadow-panel">
      <h2 className="text-xl font-semibold tracking-tight">Organism Inspector</h2>
      <div className="mt-2 rounded-xl bg-slate-100/80 p-3 font-mono text-xs">{focusMetaText}</div>
      <section className="mt-3 rounded-xl border border-accent/15 bg-white/80 p-3">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-ink/80">Stats</h3>
        <pre className="mt-2 whitespace-pre-wrap rounded-lg bg-slate-100/80 p-3 font-mono text-xs">
          {focusedStatsText}
        </pre>
      </section>
      <section className="mt-3 rounded-xl border border-accent/15 bg-white/80 p-3">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-ink/80">Brain</h3>
        <BrainCanvas focusedBrain={focusedBrain} />
      </section>
    </aside>
  );
}

