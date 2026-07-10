// The research-cockpit read tabs — each surfaces one sim-cli read (pillars, eco,
// lineage, genome, timeseries, find) as a designed panel. They fetch the live
// server endpoints and reflect the world file's last-persisted state (see
// `useWorldRead`). Styling matches the inspector: mono numerics, tracked labels.

import { useMemo, useState } from 'react';
import { colorForSpeciesId } from '../../speciesColor';
import type {
  PillarIntervalMetric,
  StatsSummary,
} from '../../types';
import { worldClient } from '../sim/api/worldClient';
import { Bar, fmt, fmtInt, SectionLabel, Sparkline, useWorldRead } from './primitives';

export type TabProps = {
  worldName: string | null;
  revision: number;
  active: boolean;
};

function TabFrame({
  state,
  children,
}: {
  state: { loading: boolean; error: string | null; hasData: boolean };
  children: React.ReactNode;
}) {
  if (state.error) {
    return <p className="p-4 text-[11px] text-[#a4492e]">{state.error}</p>;
  }
  if (!state.hasData) {
    return (
      <p className="p-4 text-[11px] text-ink/35">{state.loading ? 'Loading…' : 'No data yet.'}</p>
    );
  }
  return <div className="space-y-4 p-3.5">{children}</div>;
}

// --- Pillars ---------------------------------------------------------------

function seriesOf(intervals: PillarIntervalMetric[], key: keyof PillarIntervalMetric) {
  return intervals.map((m) => m[key] as number | null);
}

function PillarMetric({
  label,
  value,
  digits,
  intervals,
  metricKey,
  highlightFrom,
}: {
  label: string;
  value: number | null;
  digits: number;
  intervals: PillarIntervalMetric[];
  metricKey: keyof PillarIntervalMetric;
  highlightFrom: number;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between">
        <span className="text-[11px] text-ink/45">{label}</span>
        <span className="font-mono text-sm text-ink/85">{fmt(value, digits)}</span>
      </div>
      <div className="text-accent">
        <Sparkline values={seriesOf(intervals, metricKey)} className="text-accent/70" highlightFrom={highlightFrom} />
      </div>
    </div>
  );
}

export function PillarsTab({ worldName, revision, active }: TabProps) {
  const read = useWorldRead(active, worldName, revision, () => worldClient.getPillars(worldName!));
  const p = read.data;
  const intervals = p?.granular.intervals ?? [];
  const highlightFrom = useMemo(() => {
    if (!p || intervals.length === 0) return 1;
    const idx = intervals.findIndex((m) => m.tick > p.window_start_tick);
    return idx < 0 ? 1 : idx / intervals.length;
  }, [p, intervals]);

  return (
    <TabFrame state={{ loading: read.loading, error: read.error, hasData: !!p }}>
      {p && (
        <>
          <div className="flex items-center justify-between">
            <SectionLabel>Competence pillars</SectionLabel>
            <span className="font-mono text-[10px] text-ink/30">
              ({p.window_start_tick.toLocaleString()}, {p.window_end_tick.toLocaleString()}]
            </span>
          </div>
          {(p.partial || p.scaled) && (
            <div className="flex gap-1.5">
              {p.partial && (
                <span className="rounded-full border border-line bg-surface/60 px-2 py-0.5 text-[9px] uppercase tracking-wide text-ink/45">
                  partial window
                </span>
              )}
              {p.scaled && (
                <span className="rounded-full border border-line bg-surface/60 px-2 py-0.5 text-[9px] uppercase tracking-wide text-ink/45">
                  scaled
                </span>
              )}
            </div>
          )}
          <div className="space-y-3.5">
            <PillarMetric label="Foraging · plant rate" value={p.plant_consumption_rate} digits={4} intervals={intervals} metricKey="plant_consumption_rate" highlightFrom={highlightFrom} />
            <PillarMetric label="Predation · prey rate" value={p.prey_consumption_rate} digits={4} intervals={intervals} metricKey="prey_consumption_rate" highlightFrom={highlightFrom} />
            <PillarMetric label="Intelligence · action eff." value={p.action_effectiveness} digits={4} intervals={intervals} metricKey="action_effectiveness" highlightFrom={highlightFrom} />
            <PillarMetric label="Intelligence · MI(s;a)" value={p.mi_sa} digits={4} intervals={intervals} metricKey="mi_sa" highlightFrom={highlightFrom} />
            <PillarMetric label="Learning · slope" value={p.learning_slope} digits={6} intervals={intervals} metricKey="learning_slope" highlightFrom={highlightFrom} />
          </div>
          <p className="text-[10px] leading-4 text-ink/30">
            Shaded region is the scoring window. Values are windowed means over {p.granular.report_every.toLocaleString()}-tick intervals.
          </p>
        </>
      )}
    </TabFrame>
  );
}

// --- Ecology ---------------------------------------------------------------

const DEATH_CAUSES = [
  { key: 'starvation', label: 'Starvation', color: 'bg-[#b07a3a]' },
  { key: 'age', label: 'Age', color: 'bg-[#7d8a5a]' },
  { key: 'predation', label: 'Predation', color: 'bg-[#a4492e]' },
  { key: 'other', label: 'Other', color: 'bg-[#8a8374]' },
] as const;

export function EcologyTab({ worldName, revision, active }: TabProps) {
  const read = useWorldRead(active, worldName, revision, () => worldClient.getEco(worldName!));
  const e = read.data;
  return (
    <TabFrame state={{ loading: read.loading, error: read.error, hasData: !!e }}>
      {e && (
        <>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
            <Stat label="Population" value={fmtInt(e.population)} />
            <Stat label="Descendants" value={fmtInt(e.descendants)} />
            <Stat label="Plants" value={fmtInt(e.food.plants)} />
            <Stat label="Corpses" value={fmtInt(e.food.corpses)} />
          </div>
          {!e.trajectory ? (
            <p className="text-[11px] text-ink/35">{e.note ?? 'Advance the world to record a trajectory.'}</p>
          ) : (
            <>
              <div className="space-y-1">
                <SectionLabel>Population</SectionLabel>
                <div className="text-accent">
                  <Sparkline values={e.trajectory.population_series} className="text-accent/70" />
                </div>
              </div>
              <div className="space-y-1">
                <SectionLabel>Food</SectionLabel>
                <Sparkline values={e.trajectory.food_series} className="text-[#7d8a5a]" />
              </div>
              <div className="space-y-1.5">
                <SectionLabel>Deaths by cause · {fmtInt(e.trajectory.deaths_by_cause.total)} total</SectionLabel>
                {DEATH_CAUSES.map((cause) => {
                  const n = e.trajectory!.deaths_by_cause[cause.key];
                  const frac = e.trajectory!.deaths_by_cause.total > 0 ? n / e.trajectory!.deaths_by_cause.total : 0;
                  return (
                    <div key={cause.key} className="flex items-center gap-2 text-[10px]">
                      <span className="w-16 shrink-0 text-ink/40">{cause.label}</span>
                      <Bar frac={frac} className={cause.color} />
                      <span className="w-16 shrink-0 text-right font-mono text-ink/45">
                        {fmtInt(n)} · {(frac * 100).toFixed(0)}%
                      </span>
                    </div>
                  );
                })}
              </div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
                <Stat label="Births/tick" value={fmt(e.trajectory.births_per_tick, 4)} />
                <Stat label="Deaths/tick" value={fmt(e.trajectory.deaths_per_tick, 4)} />
                <Stat label="Consum/tick" value={fmt(e.trajectory.consumptions_per_tick, 4)} />
                <Stat label="Predat/tick" value={fmt(e.trajectory.predations_per_tick, 4)} />
                <Stat label="Carrying cap." value={fmt(e.trajectory.carrying_capacity_est, 1)} />
              </div>
            </>
          )}
        </>
      )}
    </TabFrame>
  );
}

// --- Lineage ---------------------------------------------------------------

function Histogram({ counts, className = 'bg-accent/50' }: { counts: number[]; className?: string }) {
  const peak = Math.max(1, ...counts);
  return (
    <div className="flex h-12 items-end gap-0.5">
      {counts.map((c, i) => (
        <div key={i} className="flex-1 rounded-sm bg-muted/30" style={{ height: '100%' }}>
          <div className={`w-full rounded-sm ${className}`} style={{ height: `${(c / peak) * 100}%`, marginTop: `${100 - (c / peak) * 100}%` }} />
        </div>
      ))}
    </div>
  );
}

function statLine(s: StatsSummary | null): string {
  if (!s) return 'n/a';
  return `min ${fmt(s.min, 1)} · p50 ${fmt(s.p50, 1)} · mean ${fmt(s.mean, 2)} · max ${fmt(s.max, 1)}`;
}

export function LineageTab({ worldName, revision, active }: TabProps) {
  const read = useWorldRead(active, worldName, revision, () => worldClient.getLineage(worldName!));
  const l = read.data;
  return (
    <TabFrame state={{ loading: read.loading, error: read.error, hasData: !!l }}>
      {l && (
        <>
          <div className="space-y-1.5">
            <SectionLabel>Generation distribution</SectionLabel>
            <Histogram counts={l.generation.histogram} />
            <p className="font-mono text-[10px] text-ink/45">{statLine(l.generation.stats)}</p>
          </div>
          <div className="space-y-1.5">
            <div className="flex items-baseline justify-between">
              <SectionLabel>Founder lineages</SectionLabel>
              <span className="font-mono text-[10px] text-ink/40">{l.lineages.distinct} distinct</span>
            </div>
            {l.lineages.top.map((sp) => (
              <div key={sp.species_id} className="flex items-center gap-2 text-[10px]">
                <span
                  className="h-2 w-2 shrink-0 rounded-full"
                  style={{ backgroundColor: colorForSpeciesId(String(sp.species_id)) }}
                />
                <span className="w-14 shrink-0 truncate font-mono text-ink/45">#{sp.species_id}</span>
                <Bar frac={sp.pct / 100} />
                <span className="w-16 shrink-0 text-right font-mono text-ink/45">
                  {fmtInt(sp.count)} · {sp.pct.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </TabFrame>
  );
}

// --- Genome ----------------------------------------------------------------

export function GenomeTab({ worldName, revision, active }: TabProps) {
  const read = useWorldRead(active, worldName, revision, () => worldClient.getGenome(worldName!));
  const g = read.data;
  const grouped = useMemo(() => {
    const byGroup = new Map<string, { name: string; stats: StatsSummary | null }[]>();
    for (const [name, gene] of Object.entries(g?.genes ?? {})) {
      const list = byGroup.get(gene.group) ?? [];
      list.push({ name, stats: gene.stats });
      byGroup.set(gene.group, list);
    }
    return [...byGroup.entries()];
  }, [g]);

  return (
    <TabFrame state={{ loading: read.loading, error: read.error, hasData: !!g }}>
      {g && (
        <>
          {grouped.map(([group, genes]) => (
            <div key={group} className="space-y-1">
              <SectionLabel>{group}</SectionLabel>
              <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-3 gap-y-0.5 text-[11px]">
                {genes.map((gene) => (
                  <div key={gene.name} className="contents">
                    <dt className="truncate text-ink/40">{gene.name}</dt>
                    <dd className="truncate text-right font-mono text-ink/75">
                      {gene.stats ? `${fmt(gene.stats.mean, 3)} · [${fmt(gene.stats.min, 2)}, ${fmt(gene.stats.max, 2)}]` : 'n/a'}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
          {g.drift_note && <p className="text-[10px] text-ink/30">{g.drift_note}</p>}
        </>
      )}
    </TabFrame>
  );
}

// --- Timeseries ------------------------------------------------------------

const SERIES_COLUMNS = [
  'population',
  'descendants',
  'food',
  'births',
  'deaths',
  'consumptions',
  'predations',
  'reproductions',
  'action_effectiveness',
  'plant_consumption_rate',
  'prey_consumption_rate',
  'mi_sa',
  'learning_slope',
  'pop',
] as const;

const DEFAULT_SERIES = ['population', 'food', 'plant_consumption_rate', 'mi_sa'];

export function SeriesTab({ worldName, revision, active }: TabProps) {
  const [cols, setCols] = useState<string[]>(DEFAULT_SERIES);
  const read = useWorldRead(active, worldName, revision, () =>
    worldClient.getTimeseries(worldName!, cols),
  );
  const data = read.data;

  const toggle = (col: string) => {
    setCols((prev) => (prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]));
  };

  return (
    <TabFrame state={{ loading: read.loading, error: read.error, hasData: !!data }}>
      <div className="flex flex-wrap gap-1">
        {SERIES_COLUMNS.map((col) => {
          const on = cols.includes(col);
          return (
            <button
              key={col}
              onClick={() => toggle(col)}
              className={`rounded-full border px-2 py-0.5 font-mono text-[9px] transition ${
                on
                  ? 'border-accent/40 bg-accent/15 text-ink/70'
                  : 'border-line bg-surface/40 text-ink/35 hover:text-ink/55'
              }`}
            >
              {col}
            </button>
          );
        })}
      </div>
      <div className="space-y-3">
        {data &&
          cols.map((col) => {
            const series = data[col];
            if (!series) return null;
            const last = [...series].reverse().find((v) => v != null && Number.isFinite(v));
            return (
              <div key={col} className="space-y-1">
                <div className="flex items-baseline justify-between">
                  <span className="font-mono text-[10px] text-ink/45">{col}</span>
                  <span className="font-mono text-[11px] text-ink/70">{fmt(last ?? null, 4)}</span>
                </div>
                <Sparkline values={series} className="text-ink/45" />
              </div>
            );
          })}
      </div>
    </TabFrame>
  );
}

// --- Find ------------------------------------------------------------------

export function FindTab({
  worldName,
  onFocusOrganism,
}: TabProps & { onFocusOrganism: (id: number) => void }) {
  const [expr, setExpr] = useState('energy > 100 and age < 50');
  const [limit, setLimit] = useState(20);
  const [result, setResult] = useState<{ matched: number; shown: number; rows: Record<string, number | boolean>[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = () => {
    if (!worldName || !expr.trim()) return;
    setLoading(true);
    setError(null);
    void worldClient
      .find(worldName, expr.trim(), limit)
      .then((r) => setResult(r))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'find failed'))
      .finally(() => setLoading(false));
  };

  const columns = result && result.rows.length > 0 ? Object.keys(result.rows[0]) : [];

  return (
    <div className="space-y-3 p-3.5">
      <div className="space-y-1.5">
        <SectionLabel>Predicate</SectionLabel>
        <textarea
          value={expr}
          onChange={(ev) => setExpr(ev.target.value)}
          onKeyDown={(ev) => {
            if (ev.key === 'Enter' && (ev.metaKey || ev.ctrlKey)) run();
          }}
          rows={2}
          placeholder="energy > 100 and age < 50"
          className="w-full resize-none rounded-lg border border-line bg-surface/40 px-2.5 py-1.5 font-mono text-[11px] text-ink/80 outline-none focus:border-accent/40"
        />
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-[10px] text-ink/40">
            limit
            <input
              type="number"
              min={1}
              value={limit}
              onChange={(ev) => setLimit(Math.max(1, Number(ev.target.value) || 1))}
              className="w-16 rounded-md border border-line bg-surface/40 px-1.5 py-0.5 font-mono text-[10px] text-ink/70 outline-none"
            />
          </label>
          <button
            onClick={run}
            disabled={loading}
            className="rounded-md bg-accent px-3 py-1 text-[11px] font-semibold text-white transition hover:bg-accent/90 disabled:opacity-50"
          >
            {loading ? 'Running…' : 'Run'}
          </button>
        </div>
        <p className="text-[10px] leading-4 text-ink/30">
          Fields: id energy health age generation species consumptions plant prey reproductions
          neurons synapses vision hebb_eta gestating. Operators: &lt; &lt;= &gt; &gt;= == != · join
          with and/or.
        </p>
      </div>

      {error && <p className="text-[11px] text-[#a4492e]">{error}</p>}

      {result && (
        <div className="space-y-1.5">
          <SectionLabel>
            {result.matched} match{result.matched === 1 ? '' : 'es'}
            {result.matched > result.shown ? ` · showing ${result.shown}` : ''}
          </SectionLabel>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[10px]">
              <thead>
                <tr className="text-ink/35">
                  {columns.map((c) => (
                    <th key={c} className="px-1 py-0.5 font-medium">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="font-mono text-ink/70">
                {result.rows.map((row, i) => (
                  <tr
                    key={i}
                    onClick={() => typeof row.id === 'number' && onFocusOrganism(row.id)}
                    className="cursor-pointer border-t border-line/60 transition hover:bg-surface/50"
                  >
                    {columns.map((c) => (
                      <td key={c} className="px-1 py-0.5">
                        {typeof row[c] === 'boolean' ? (row[c] ? 'yes' : 'no') : String(row[c])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// Small shared stat cell used by ecology.
function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-ink/40">{label}</span>
      <span className="font-mono text-ink/75">{value}</span>
    </div>
  );
}
