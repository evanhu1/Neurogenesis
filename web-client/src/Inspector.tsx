import { useEffect, useState } from 'react';
import { api, type OrganismDetail } from './api';
import { BrainView } from './BrainView';

function Bar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = Math.max(0, Math.min(1, value / (max || 1))) * 100;
  return (
    <div>
      <div className="flex justify-between font-mono text-[11px] text-ink/60">
        <span>{label}</span>
        <span>
          {value.toFixed(0)} / {max.toFixed(0)}
        </span>
      </div>
      <div className="mt-0.5 h-2 overflow-hidden rounded bg-void">
        <div className="h-full" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between text-[13px]">
      <span className="text-ink/55">{label}</span>
      <span className="tabular-nums">{value}</span>
    </div>
  );
}

export function Inspector({ id }: { id: number | null }) {
  const [detail, setDetail] = useState<OrganismDetail | null>(null);
  const [saved, setSaved] = useState<null | boolean>(null);

  useEffect(() => {
    setSaved(null);
    if (id === null) {
      setDetail(null);
      return;
    }
    let live = true;
    const load = () =>
      api
        .organism(id)
        .then((d) => live && setDetail(d))
        .catch(() => live && setDetail(null));
    load();
    const h = setInterval(load, 400); // keep energy/health live while selected
    return () => {
      live = false;
      clearInterval(h);
    };
  }, [id]);

  if (id === null)
    return (
      <div className="flex h-full items-center justify-center p-6 text-center text-[13px] text-ink/45">
        Click an organism in the world to inspect its genome and developed brain.
      </div>
    );
  if (!detail)
    return <div className="p-4 text-[13px] text-ink/45">Organism {id} — no longer alive.</div>;

  const g = detail.genome;
  const maxHealth = detail.morphology[4] ?? detail.health; // size dial = max health
  const [rr, gg, bb] = detail.morphology.slice(0, 3);
  const swatch = `rgb(${(rr * 255) | 0},${(gg * 255) | 0},${(bb * 255) | 0})`;

  // Activation histogram over CPPN nodes.
  const acts: Record<string, number> = {};
  for (const n of g.cppn.nodes) acts[n.activation] = (acts[n.activation] ?? 0) + 1;

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-4 scrollbar-none">
      <div className="flex items-center gap-2">
        <span className="inline-block h-4 w-4 rounded-full border border-line" style={{ background: swatch }} />
        <h3 className="text-base font-semibold">Organism #{detail.id}</h3>
      </div>

      <Bar label="energy" value={detail.energy} max={maxHealth} color="#eab308" />
      <Bar label="health" value={detail.health} max={maxHealth} color="#15803d" />

      <div className="rounded-lg border border-line bg-void p-3">
        <Field label="age" value={detail.age_turns} />
        <Field label="generation" value={detail.generation} />
        <Field label="vision distance" value={(detail.morphology[3] ?? 0).toFixed(1)} />
        <Field label="body size" value={(detail.morphology[4] ?? 0).toFixed(0)} />
      </div>

      <div>
        <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-ink/50">
          developed brain
        </div>
        <div className="rounded-lg border border-line bg-void p-2">
          <BrainView brain={detail.brain} />
          <div className="mt-1 flex justify-between px-1 font-mono text-[11px] text-ink/55">
            <span>in {detail.brain.input_count}</span>
            <span>hidden {detail.brain.hidden_count}</span>
            <span>out {detail.brain.output_count}</span>
            <span>edges {detail.brain.edges.length}</span>
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-line bg-void p-3">
        <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-ink/50">
          CPPN genome
        </div>
        <Field label="nodes / connections" value={`${g.cppn.nodes.length} / ${g.cppn.conns.length}`} />
        <div className="mt-1 flex flex-wrap gap-1">
          {Object.entries(acts).map(([a, c]) => (
            <span key={a} className="rounded bg-surface px-1.5 py-0.5 font-mono text-[10px]">
              {a} ×{c}
            </span>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-line bg-void p-3">
        <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-ink/50">
          plasticity (global)
        </div>
        <Field label="hebb η gain" value={(g.header.plasticity.hebb_eta_gain ?? 0).toFixed(3)} />
        <Field label="juvenile scale" value={(g.header.plasticity.juvenile_eta_scale ?? 0).toFixed(2)} />
        <Field label="elig. retention" value={(g.header.plasticity.eligibility_retention ?? 0).toFixed(2)} />
      </div>

      <button
        onClick={() => api.saveChampion(detail.id).then((r: any) => setSaved(!!r?.became_elite))}
        className="rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white transition hover:opacity-90"
      >
        {saved === null
          ? '★ Save to QD champion archive'
          : saved
          ? '★ Saved as a new niche elite'
          : '★ Saved (existing elite kept)'}
      </button>
    </div>
  );
}
