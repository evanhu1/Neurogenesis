import { useMemo, useState } from 'react';
import { colorForSpeciesId } from '../../speciesColor';
import type { OrganismState } from '../../types';
import { BrainCanvas } from './BrainCanvas';

type InspectorContentProps = {
  focusedOrganism: OrganismState | null;
  activeActionNeuronId: number | null;
  onDefocus: () => void;
};

type StatItem = {
  label: string;
  value: string;
};

type SectionKey = 'genome';

function formatFloat(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
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
      className="group rounded-lg border border-line bg-surface/30"
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

// Chrome-less inspector body: identity row + organism stats + neural network.
// Rendered inside the cockpit's "Inspect" tab (the tab strip is the header).
export function InspectorContent({
  focusedOrganism,
  activeActionNeuronId,
  onDefocus,
}: InspectorContentProps) {
  const [expandedSections, setExpandedSections] = useState<Record<SectionKey, boolean>>({
    genome: false,
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
        { label: 'Neurons', value: String(genome.brain.hidden_nodes.length) },
        { label: 'Synapses', value: String(genome.brain.edges.filter((edge) => edge.enabled).length) },
        { label: 'Vision', value: String(genome.topology.vision_distance) },
        { label: 'Maturity', value: String(genome.lifecycle.age_of_maturity) },
        { label: 'Gestation', value: String(genome.lifecycle.gestation_ticks) },
        { label: 'Max Age', value: String(genome.lifecycle.max_organism_age) },
        { label: 'Hebb Gain', value: formatFloat(genome.plasticity.hebb_eta_gain, 3) },
        { label: 'Juvenile Eta', value: formatFloat(genome.plasticity.juvenile_eta_scale, 3) },
        { label: 'Elig Ret', value: formatFloat(genome.plasticity.eligibility_retention, 3) },
        { label: 'Prune Thr', value: formatFloat(genome.plasticity.synapse_prune_threshold, 3) },
      ] satisfies StatItem[],
    };
  }, [focusedOrganism]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      {focusedOrganism && (
        <div className="flex shrink-0 items-center justify-between border-b border-line px-3.5 py-2">
          <div className="flex items-center gap-2">
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: colorForSpeciesId(String(focusedOrganism.species_id)) }}
            />
            <h2 className="text-xs font-semibold text-ink/85">
              Organism <span className="font-mono">#{focusedOrganism.id}</span>
            </h2>
            <span className="rounded-full border border-line bg-surface/60 px-2 py-0.5 font-mono text-[10px] text-ink/45">
              species {focusedOrganism.species_id}
            </span>
          </div>
          <button
            onClick={onDefocus}
            className="rounded-md p-1 text-ink/25 transition hover:bg-surface/60 hover:text-ink/60"
            aria-label="Clear selection"
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
        </div>
      )}

      {!summary ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-line bg-surface/40">
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
          </div>

          <div className="flex min-h-0 flex-1 flex-col border-t border-line p-3">
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
    </div>
  );
}
