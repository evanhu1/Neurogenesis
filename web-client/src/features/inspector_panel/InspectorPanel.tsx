import { useMemo, useState } from 'react';
import type { BrainState, OrganismState } from '../../types';
import { BrainCanvas } from './BrainCanvas';

type InspectorPanelProps = {
  focusedOrganism: OrganismState | null;
  focusedBrain: BrainState | null;
  activeActionNeuronId: number | null;
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
  return `${(value * 100).toFixed(1)}%`;
}

function statGrid(stats: StatItem[]) {
  return (
    <dl className="grid grid-cols-[auto_1fr_auto_1fr] items-baseline gap-x-3 gap-y-0.5 text-[11px]">
      {stats.map((stat) => (
        <div key={stat.label} className="contents">
          <dt className="text-ink/35">{stat.label}</dt>
          <dd className="truncate font-mono text-ink/75">{stat.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function InspectorPanel({
  focusedOrganism,
  focusedBrain,
  activeActionNeuronId,
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
    const activeNeuronCount = activeActionNeuronId === null ? 0 : 1;

    const mutationRates: MutationRateItem[] = [
      { key: 'mutation_rate_age_of_maturity', label: 'Age Mat', value: genome.mutation_rate_age_of_maturity },
      { key: 'mutation_rate_vision_distance', label: 'Vision', value: genome.mutation_rate_vision_distance },
      { key: 'mutation_rate_neuron_location', label: 'Neuron Loc', value: genome.mutation_rate_neuron_location },
      { key: 'mutation_rate_inter_bias', label: 'Bias', value: genome.mutation_rate_inter_bias },
      { key: 'mutation_rate_inter_update_rate', label: 'Update', value: genome.mutation_rate_inter_update_rate },
      { key: 'mutation_rate_eligibility_retention', label: 'Elig Ret', value: genome.mutation_rate_eligibility_retention },
      { key: 'mutation_rate_synapse_prune_threshold', label: 'Prune', value: genome.mutation_rate_synapse_prune_threshold },
      { key: 'mutation_rate_synapse_weight_perturbation', label: 'Wt Perturb', value: genome.mutation_rate_synapse_weight_perturbation },
      { key: 'mutation_rate_add_neuron_split_edge', label: 'Split Edge', value: genome.mutation_rate_add_neuron_split_edge },
    ];

    return {
      coreStats: [
        { label: 'ID', value: String(focusedOrganism.id) },
        { label: 'Species', value: String(focusedOrganism.species_id) },
        { label: 'Gen', value: String(focusedOrganism.generation) },
        { label: 'Pos', value: `(${focusedOrganism.q},${focusedOrganism.r})` },
        { label: 'Facing', value: focusedOrganism.facing },
        { label: 'Age', value: String(focusedOrganism.age_turns) },
        { label: 'Energy', value: formatFloat(focusedOrganism.energy, 2) },
        { label: 'Action', value: focusedOrganism.last_action_taken },
        { label: 'Dopamine', value: formatFloat(focusedOrganism.dopamine, 3) },
        { label: 'Meals', value: String(focusedOrganism.consumptions_count) },
        { label: 'Repr', value: String(focusedOrganism.reproductions_count) },
      ] satisfies StatItem[],
      brainStats: [
        { label: 'Sensory', value: String(focusedOrganism.brain.sensory.length) },
        { label: 'Inter', value: String(focusedOrganism.brain.inter.length) },
        { label: 'Action', value: String(focusedOrganism.brain.action.length) },
        { label: 'Synapses', value: String(focusedOrganism.brain.synapse_count) },
        { label: 'Active', value: String(activeNeuronCount) },
      ] satisfies StatItem[],
      genomeStats: [
        { label: 'Neurons', value: String(genome.num_neurons) },
        { label: 'Synapses', value: String(genome.num_synapses) },
        { label: 'Vision', value: String(genome.vision_distance) },
        { label: 'Maturity', value: String(genome.age_of_maturity) },
        { label: 'Biases', value: String(genome.inter_biases.length) },
        { label: 'Time Cst', value: String(genome.inter_log_time_constants.length) },
      ] satisfies StatItem[],
      mutationRates,
    };
  }, [activeActionNeuronId, focusedOrganism]);

  return (
    <aside className="h-full overflow-auto rounded-xl bg-panel p-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-[0.16em] text-ink/40">Inspector</h2>
        <button
          onClick={onDefocus}
          className="rounded p-0.5 text-ink/20 transition hover:text-ink/50"
          aria-label="Close inspector"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
            <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
          </svg>
        </button>
      </div>

      {!summary ? (
        <p className="mt-4 text-center text-xs text-ink/25">
          Select an organism to inspect
        </p>
      ) : (
        <div className="mt-2 space-y-2">
          <h3 className="text-[10px] font-semibold uppercase tracking-[0.14em] text-ink/35">Core</h3>
          {statGrid(summary.coreStats)}

          <details
            open={expandedSections.brain}
            onToggle={(event) => setSectionOpen('brain', event.currentTarget.open)}
          >
            <summary className="cursor-pointer text-[10px] font-semibold uppercase tracking-[0.14em] text-ink/35 transition hover:text-ink/55">
              Brain
            </summary>
            <div className="mt-1">{statGrid(summary.brainStats)}</div>
          </details>

          <details
            open={expandedSections.genome}
            onToggle={(event) => setSectionOpen('genome', event.currentTarget.open)}
          >
            <summary className="cursor-pointer text-[10px] font-semibold uppercase tracking-[0.14em] text-ink/35 transition hover:text-ink/55">
              Genome
            </summary>
            <div className="mt-1">{statGrid(summary.genomeStats)}</div>
          </details>

          <details
            open={expandedSections.mutationRates}
            onToggle={(event) => setSectionOpen('mutationRates', event.currentTarget.open)}
          >
            <summary className="cursor-pointer text-[10px] font-semibold uppercase tracking-[0.14em] text-ink/35 transition hover:text-ink/55">
              Mutation Rates
            </summary>
            <div className="mt-1 space-y-1">
              {summary.mutationRates.map((entry) => {
                const clamped = Math.max(0, Math.min(1, entry.value));
                return (
                  <div key={entry.key} className="flex items-center gap-2 text-[10px]">
                    <span className="w-16 shrink-0 text-ink/35">{entry.label}</span>
                    <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted/40">
                      <div
                        className="h-full rounded-full bg-accent/40"
                        style={{ width: `${clamped * 100}%` }}
                      />
                    </div>
                    <span className="w-10 shrink-0 text-right font-mono text-ink/40">
                      {formatPercent(entry.value)}
                    </span>
                  </div>
                );
              })}
            </div>
          </details>
        </div>
      )}

      <div className="mt-3">
        <h3 className="text-[10px] font-semibold uppercase tracking-[0.14em] text-ink/35">
          Brain
        </h3>
        <div className="mt-1 h-[500px]">
          <BrainCanvas
            focusedBrain={focusedBrain}
            activeActionNeuronId={activeActionNeuronId}
            focusOrganismId={focusedOrganism?.id ?? null}
          />
        </div>
      </div>
    </aside>
  );
}
