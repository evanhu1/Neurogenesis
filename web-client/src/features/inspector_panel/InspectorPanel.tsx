import { useMemo, useState } from 'react';
import { unwrapId } from '../../protocol';
import type { BrainState, OrganismState } from '../../types';
import { BrainCanvas } from './BrainCanvas';

type InspectorPanelProps = {
  focusMetaText: string;
  focusedOrganism: OrganismState | null;
  focusedBrain: BrainState | null;
  activeNeuronIds: Set<number> | null;
  onDefocus: () => void;
};

type MutationRateItem = {
  key: keyof OrganismState['genome'];
  label: string;
  value: number;
};

type StatItem = {
  label: string;
  value: string;
};

type SectionKey = 'brain' | 'genome' | 'mutationRates';

function formatFloat(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function statSection(title: string, stats: StatItem[]) {
  return (
    <section className="rounded-xl border border-accent/20 bg-white/85 p-3">
      <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink/75">{title}</h3>
      <div className="mt-2">{statList(stats)}</div>
    </section>
  );
}

function statList(stats: StatItem[]) {
  return (
    <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-3 gap-y-1 text-xs">
      {stats.map((stat) => (
        <div key={stat.label} className="contents">
          <dt className="font-medium text-ink/65">{stat.label}</dt>
          <dd className="truncate text-right font-mono text-ink/95">{stat.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function InspectorPanel({
  focusMetaText,
  focusedOrganism,
  focusedBrain,
  activeNeuronIds,
  onDefocus,
}: InspectorPanelProps) {
  const [expandedSections, setExpandedSections] = useState<Record<SectionKey, boolean>>({
    brain: false,
    genome: false,
    mutationRates: false,
  });

  const setSectionOpen = (key: SectionKey, isOpen: boolean) => {
    setExpandedSections((prev) => {
      if (prev[key] === isOpen) return prev;
      return { ...prev, [key]: isOpen };
    });
  };

  const summary = useMemo(() => {
    if (!focusedOrganism) return null;
    const genome = focusedOrganism.genome;
    const interExcitatoryCount = genome.interneuron_types.filter((v) => v === 'Excitatory').length;
    const interInhibitoryCount = genome.interneuron_types.length - interExcitatoryCount;
    const activeNeuronCount = activeNeuronIds?.size ?? 0;

    const mutationRates: MutationRateItem[] = [
      {
        key: 'mutation_rate_age_of_maturity',
        label: 'Age Maturity',
        value: genome.mutation_rate_age_of_maturity,
      },
      {
        key: 'mutation_rate_vision_distance',
        label: 'Vision Distance',
        value: genome.mutation_rate_vision_distance,
      },
      {
        key: 'mutation_rate_neuron_location',
        label: 'Neuron Location',
        value: genome.mutation_rate_neuron_location,
      },
      {
        key: 'mutation_rate_inter_bias',
        label: 'Inter Bias',
        value: genome.mutation_rate_inter_bias,
      },
      {
        key: 'mutation_rate_inter_update_rate',
        label: 'Inter Update',
        value: genome.mutation_rate_inter_update_rate,
      },
      {
        key: 'mutation_rate_action_bias',
        label: 'Action Bias',
        value: genome.mutation_rate_action_bias,
      },
      {
        key: 'mutation_rate_eligibility_retention',
        label: 'Elig Retention',
        value: genome.mutation_rate_eligibility_retention,
      },
      {
        key: 'mutation_rate_synapse_prune_threshold',
        label: 'Prune Thresh',
        value: genome.mutation_rate_synapse_prune_threshold,
      },
      {
        key: 'mutation_rate_synapse_weight_perturbation',
        label: 'Weight Perturb',
        value: genome.mutation_rate_synapse_weight_perturbation,
      },
      {
        key: 'mutation_rate_add_neuron_split_edge',
        label: 'Split Edge Add',
        value: genome.mutation_rate_add_neuron_split_edge,
      },
    ];

    return {
      coreStats: [
        { label: 'Organism ID', value: String(unwrapId(focusedOrganism.id)) },
        { label: 'Species ID', value: String(unwrapId(focusedOrganism.species_id)) },
        { label: 'Position', value: `(${focusedOrganism.q}, ${focusedOrganism.r})` },
        { label: 'Facing', value: focusedOrganism.facing },
        { label: 'Age (turns)', value: String(focusedOrganism.age_turns) },
        { label: 'Energy', value: formatFloat(focusedOrganism.energy, 2) },
        { label: 'Dopamine', value: formatFloat(focusedOrganism.dopamine, 3) },
        { label: 'Consumptions', value: String(focusedOrganism.consumptions_count) },
        { label: 'Reproductions', value: String(focusedOrganism.reproductions_count) },
      ] satisfies StatItem[],
      brainStats: [
        { label: 'Sensory Neurons', value: String(focusedOrganism.brain.sensory.length) },
        { label: 'Inter Neurons', value: String(focusedOrganism.brain.inter.length) },
        { label: 'Action Neurons', value: String(focusedOrganism.brain.action.length) },
        { label: 'Synapses', value: String(focusedOrganism.brain.synapse_count) },
        { label: 'Active Neurons', value: String(activeNeuronCount) },
      ] satisfies StatItem[],
      genomeStats: [
        { label: 'Genome Neurons', value: String(genome.num_neurons) },
        { label: 'Genome Synapses', value: String(genome.num_synapses) },
        { label: 'Vision Distance', value: String(genome.vision_distance) },
        { label: 'Age Maturity', value: String(genome.age_of_maturity) },
        { label: 'Inter Bias Genes', value: String(genome.inter_biases.length) },
        {
          label: 'Inter Time Constant Genes',
          value: String(genome.inter_log_time_constants.length),
        },
        { label: 'Action Bias Genes', value: String(genome.action_biases.length) },
        { label: 'Inter Excitatory', value: String(interExcitatoryCount) },
        { label: 'Inter Inhibitory', value: String(interInhibitoryCount) },
      ] satisfies StatItem[],
      mutationRates,
    };
  }, [activeNeuronIds, focusedOrganism]);

  return (
    <aside className="h-full overflow-auto rounded-2xl border border-accent/20 bg-gradient-to-b from-white/95 to-slate-50/90 p-4 shadow-panel">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold tracking-tight text-ink">Organism Inspector</h2>
        <button
          onClick={onDefocus}
          className="rounded-md p-1 text-ink/40 transition hover:bg-slate-200 hover:text-ink/80"
          aria-label="Close inspector"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5">
            <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
          </svg>
        </button>
      </div>
      <div className="mt-2 rounded-xl border border-accent/15 bg-slate-100/70 p-3 font-mono text-[11px] text-ink/80">
        {focusMetaText}
      </div>

      {!summary ? (
        <section className="mt-3 rounded-xl border border-dashed border-accent/30 bg-white/80 px-4 py-6 text-sm text-ink/70">
          Select an organism in the world to inspect its stats, genome, mutation rates, and neural activity.
        </section>
      ) : (
        <div className="mt-3 space-y-3">
          {statSection('Core', summary.coreStats)}
          <details
            open={expandedSections.brain}
            onToggle={(event) => setSectionOpen('brain', event.currentTarget.open)}
            className="rounded-xl border border-accent/20 bg-white/85 p-3"
          >
            <summary className="cursor-pointer text-[11px] font-semibold uppercase tracking-[0.14em] text-ink/75">
              Brain
            </summary>
            <div className="mt-2">{statList(summary.brainStats)}</div>
          </details>

          <details
            open={expandedSections.genome}
            onToggle={(event) => setSectionOpen('genome', event.currentTarget.open)}
            className="rounded-xl border border-accent/20 bg-white/85 p-3"
          >
            <summary className="cursor-pointer text-[11px] font-semibold uppercase tracking-[0.14em] text-ink/75">
              Genome
            </summary>
            <div className="mt-2">{statList(summary.genomeStats)}</div>
          </details>

          <details
            open={expandedSections.mutationRates}
            onToggle={(event) => setSectionOpen('mutationRates', event.currentTarget.open)}
            className="rounded-xl border border-accent/20 bg-white/85 p-3"
          >
            <summary className="cursor-pointer text-[11px] font-semibold uppercase tracking-[0.14em] text-ink/75">
              Mutation Rates
            </summary>
            <ul className="mt-2 space-y-2">
              {summary.mutationRates.map((entry) => {
                const clamped = Math.max(0, Math.min(1, entry.value));
                return (
                  <li key={entry.key} className="rounded-lg bg-slate-100/80 p-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-ink/80">{entry.label}</span>
                      <span className="font-mono text-ink/90">{formatPercent(entry.value)}</span>
                    </div>
                    <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-300/80">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-600"
                        style={{ width: `${clamped * 100}%` }}
                      />
                    </div>
                  </li>
                );
              })}
            </ul>
          </details>
        </div>
      )}

          <section className="mt-3 rounded-xl border border-accent/20 bg-white/85 p-3">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-ink/80">Brain</h3>
        <div className="mt-2 h-[440px]">
          <BrainCanvas
            focusedBrain={focusedBrain}
            activeNeuronIds={activeNeuronIds}
            focusOrganismId={focusedOrganism ? unwrapId(focusedOrganism.id) : null}
          />
        </div>
      </section>
    </aside>
  );
}
