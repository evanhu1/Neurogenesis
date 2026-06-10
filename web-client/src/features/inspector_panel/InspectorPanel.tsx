import { useMemo, useState } from 'react';
import { colorForSpeciesId } from '../../speciesColor';
import type { MutationRateGenes, OrganismState } from '../../types';
import { BrainCanvas } from './BrainCanvas';

type InspectorPanelProps = {
  focusedOrganism: OrganismState | null;
  activeActionNeuronId: number | null;
  onDefocus: () => void;
};

type MutationRateItem = {
  key: keyof MutationRateGenes;
  label: string;
  value: number;
};

type StatItem = {
  label: string;
  value: string;
};

type SectionKey = 'genome' | 'mutationRates';

function formatFloat(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function StatGrid({ stats }: { stats: StatItem[] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr_auto_1fr] items-baseline gap-x-3 gap-y-1 text-[11px]">
      {stats.map((stat) => (
        <div key={stat.label} className="contents">
          <dt className="text-ink/35">{stat.label}</dt>
          <dd className="truncate font-mono text-ink/80">{stat.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function CollapsibleSection({
  title,
  open,
  onToggle,
  children,
}: {
  title: string;
  open: boolean;
  onToggle: (isOpen: boolean) => void;
  children: React.ReactNode;
}) {
  return (
    <details
      open={open}
      onToggle={(event) => onToggle(event.currentTarget.open)}
      className="group rounded-lg border border-white/5 bg-surface/30"
    >
      <summary className="flex cursor-pointer items-center justify-between px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-ink/40 transition hover:text-ink/65 [&::-webkit-details-marker]:hidden">
        {title}
        <svg
          aria-hidden="true"
          viewBox="0 0 16 16"
          className="h-3 w-3 fill-current transition-transform group-open:rotate-180"
        >
          <path d="M3.2 5.7a.75.75 0 0 1 1.06-.04L8 9.05l3.74-3.4a.75.75 0 1 1 1.02 1.1l-4.25 3.87a.75.75 0 0 1-1.02 0L3.24 6.76a.75.75 0 0 1-.04-1.06Z" />
        </svg>
      </summary>
      <div className="px-2.5 pb-2">{children}</div>
    </details>
  );
}

export function InspectorPanel({
  focusedOrganism,
  activeActionNeuronId,
  onDefocus,
}: InspectorPanelProps) {
  const [expandedSections, setExpandedSections] = useState<Record<SectionKey, boolean>>({
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
    const brain = focusedOrganism.brain;

    const mutationRates: MutationRateItem[] = [
      { key: 'age_of_maturity', label: 'Age Mat', value: genome.mutation_rates.age_of_maturity },
      { key: 'gestation_ticks', label: 'Gest', value: genome.mutation_rates.gestation_ticks },
      { key: 'vision_distance', label: 'Vision', value: genome.mutation_rates.vision_distance },
      { key: 'neuron_location', label: 'Neuron Loc', value: genome.mutation_rates.neuron_location },
      { key: 'inter_bias', label: 'Bias', value: genome.mutation_rates.inter_bias },
      { key: 'inter_update_rate', label: 'Update', value: genome.mutation_rates.inter_update_rate },
      {
        key: 'eligibility_retention',
        label: 'Elig Ret',
        value: genome.mutation_rates.eligibility_retention,
      },
      {
        key: 'synapse_prune_threshold',
        label: 'Prune',
        value: genome.mutation_rates.synapse_prune_threshold,
      },
      {
        key: 'synapse_weight_perturbation',
        label: 'Wt Perturb',
        value: genome.mutation_rates.synapse_weight_perturbation,
      },
      {
        key: 'add_neuron_split_edge',
        label: 'Split Edge',
        value: genome.mutation_rates.add_neuron_split_edge,
      },
    ];

    return {
      coreStats: [
        { label: 'Age', value: String(focusedOrganism.age_turns) },
        { label: 'Gen', value: String(focusedOrganism.generation) },
        {
          label: 'Health',
          value: `${formatFloat(focusedOrganism.health, 1)} / ${formatFloat(focusedOrganism.max_health, 1)}`,
        },
        { label: 'Energy', value: formatFloat(focusedOrganism.energy, 2) },
        { label: 'Action', value: focusedOrganism.last_action_taken },
        { label: 'Pregnant', value: focusedOrganism.is_gestating ? 'Yes' : 'No' },
        { label: 'Dopamine', value: formatFloat(focusedOrganism.dopamine, 3) },
        { label: 'V(s)', value: formatFloat(focusedOrganism.value_prev, 3) },
        { label: 'Plants', value: String(focusedOrganism.plant_consumptions_count) },
        { label: 'Prey', value: String(focusedOrganism.prey_consumptions_count) },
        { label: 'Repr', value: String(focusedOrganism.reproductions_count) },
        {
          label: 'Neurons',
          value: `${brain.sensory.length}s · ${brain.inter.length}i · ${brain.action.length}a`,
        },
        { label: 'Synapses', value: String(brain.synapse_count) },
      ] satisfies StatItem[],
      genomeStats: [
        { label: 'Neurons', value: String(genome.topology.num_neurons) },
        { label: 'Synapses', value: String(genome.topology.num_synapses) },
        { label: 'Vision', value: String(genome.topology.vision_distance) },
        { label: 'Maturity', value: String(genome.lifecycle.age_of_maturity) },
        { label: 'Gestation', value: String(genome.lifecycle.gestation_ticks) },
        { label: 'Max Age', value: String(genome.lifecycle.max_organism_age) },
        { label: 'Hebb Gain', value: formatFloat(genome.plasticity.hebb_eta_gain, 3) },
        { label: 'Juvenile Eta', value: formatFloat(genome.plasticity.juvenile_eta_scale, 3) },
        { label: 'Elig Ret', value: formatFloat(genome.plasticity.eligibility_retention, 3) },
        { label: 'Prune Thr', value: formatFloat(genome.plasticity.synapse_prune_threshold, 3) },
        { label: 'Rw Eng Lvl', value: formatFloat(genome.reward_weights[0] ?? 0, 3) },
        { label: 'Rw Eng +', value: formatFloat(genome.reward_weights[1] ?? 0, 3) },
        { label: 'Rw Eng -', value: formatFloat(genome.reward_weights[2] ?? 0, 3) },
        { label: 'Rw HP Lvl', value: formatFloat(genome.reward_weights[3] ?? 0, 3) },
        { label: 'Rw HP +', value: formatFloat(genome.reward_weights[4] ?? 0, 3) },
        { label: 'Rw HP -', value: formatFloat(genome.reward_weights[5] ?? 0, 3) },
      ] satisfies StatItem[],
      mutationRates,
    };
  }, [focusedOrganism]);

  return (
    <aside className="flex h-full flex-col overflow-hidden rounded-2xl border border-white/5 bg-panel/90 shadow-panel backdrop-blur">
      <header className="flex shrink-0 items-center justify-between border-b border-white/5 px-3.5 py-2.5">
        {focusedOrganism ? (
          <div className="flex items-center gap-2">
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: colorForSpeciesId(String(focusedOrganism.species_id)) }}
            />
            <h2 className="text-xs font-semibold text-ink/85">
              Organism <span className="font-mono">#{focusedOrganism.id}</span>
            </h2>
            <span className="rounded-full border border-white/10 bg-surface/60 px-2 py-0.5 font-mono text-[10px] text-ink/45">
              species {focusedOrganism.species_id}
            </span>
          </div>
        ) : (
          <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-ink/40">
            Inspector
          </h2>
        )}
        <button
          onClick={onDefocus}
          className="rounded-md p-1 text-ink/25 transition hover:bg-surface/60 hover:text-ink/60"
          aria-label="Close inspector"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
            className="h-4 w-4"
          >
            <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
          </svg>
        </button>
      </header>

      {!summary ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/5 bg-surface/40">
            <svg aria-hidden="true" viewBox="0 0 24 24" className="h-7 w-7 fill-ink/20">
              <path d="M12 4.5c4.8 0 8.6 3.6 9.7 7.5-1.1 3.9-4.9 7.5-9.7 7.5S3.4 15.9 2.3 12C3.4 8.1 7.2 4.5 12 4.5Zm0 2C8.4 6.5 5.4 9 4.4 12c1 3 4 5.5 7.6 5.5s6.6-2.5 7.6-5.5c-1-3-4-5.5-7.6-5.5Zm0 1.75a3.75 3.75 0 1 1 0 7.5 3.75 3.75 0 0 1 0-7.5Z" />
            </svg>
          </div>
          <p className="text-xs leading-5 text-ink/35">
            Click an organism in the world
            <br />
            to inspect its brain and genome
          </p>
        </div>
      ) : (
        <>
          <div className="max-h-[42%] shrink-0 space-y-2 overflow-auto p-3">
            <StatGrid stats={summary.coreStats} />

            <CollapsibleSection
              title="Genome"
              open={expandedSections.genome}
              onToggle={(isOpen) => setSectionOpen('genome', isOpen)}
            >
              <StatGrid stats={summary.genomeStats} />
            </CollapsibleSection>

            <CollapsibleSection
              title="Mutation Rates"
              open={expandedSections.mutationRates}
              onToggle={(isOpen) => setSectionOpen('mutationRates', isOpen)}
            >
              <div className="space-y-1">
                {summary.mutationRates.map((entry) => {
                  const clamped = Math.max(0, Math.min(1, entry.value));
                  return (
                    <div key={entry.key} className="flex items-center gap-2 text-[10px]">
                      <span className="w-16 shrink-0 text-ink/35">{entry.label}</span>
                      <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted/40">
                        <div
                          className="h-full rounded-full bg-accent/50"
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
            </CollapsibleSection>
          </div>

          <div className="flex min-h-0 flex-1 flex-col border-t border-white/5 p-3">
            <h3 className="shrink-0 text-[10px] font-semibold uppercase tracking-[0.18em] text-ink/35">
              Neural Network
            </h3>
            <div className="mt-1.5 min-h-0 flex-1">
              <BrainCanvas
                focusedBrain={focusedOrganism?.brain ?? null}
                activeActionNeuronId={activeActionNeuronId}
                focusOrganismId={focusedOrganism?.id ?? null}
                actionBiases={focusedOrganism?.genome.brain.action_biases ?? []}
              />
            </div>
          </div>
        </>
      )}
    </aside>
  );
}
