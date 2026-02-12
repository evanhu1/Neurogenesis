import { unwrapId } from '../../protocol';
import type { OrganismState, SessionMetadata, WorldSnapshot } from '../../types';

export function formatSessionMeta(session: SessionMetadata | null): string {
  if (!session) return 'No session';
  return `session=${session.id}\ncreated=${new Date(session.created_at_unix_ms).toISOString()}`;
}

export function formatMetrics(snapshot: WorldSnapshot | null): string {
  if (!snapshot) return 'No metrics';
  const speciesAlive = Object.keys(snapshot.metrics.species_counts).length;
  return [
    `turn=${snapshot.turn}`,
    `organisms=${snapshot.metrics.organisms}`,
    `species_alive=${speciesAlive}`,
    `total_species_created=${snapshot.metrics.total_species_created}`,
    `food=${snapshot.foods.length}`,
    `consumptions_last_turn=${snapshot.metrics.consumptions_last_turn}`,
    `predations_last_turn=${snapshot.metrics.predations_last_turn ?? 0}`,
    `total_food_eaten=${snapshot.metrics.total_consumptions}`,
    `reproductions_last_turn=${snapshot.metrics.reproductions_last_turn}`,
    `starvations_last_turn=${snapshot.metrics.starvations_last_turn}`,
    `synapse_ops_last_turn=${snapshot.metrics.synapse_ops_last_turn}`,
    `actions_last_turn=${snapshot.metrics.actions_applied_last_turn}`,
  ].join('\n');
}

export function formatFocusMeta(
  focusedOrganismId: number | null,
  focusedOrganism: OrganismState | null,
): string {
  if (focusedOrganismId === null || !focusedOrganism) {
    return 'Click an organism';
  }
  return `focused organism: ${focusedOrganismId} at (${focusedOrganism.q}, ${focusedOrganism.r})`;
}

export function formatFocusedStats(focusedOrganism: OrganismState | null): string {
  if (!focusedOrganism) return 'No organism selected';
  return [
    `id=${unwrapId(focusedOrganism.id)}`,
    `species_id=${unwrapId(focusedOrganism.species_id)}`,
    `position=(${focusedOrganism.q}, ${focusedOrganism.r})`,
    `facing=${focusedOrganism.facing}`,
    `age_turns=${focusedOrganism.age_turns}`,
    `energy=${focusedOrganism.energy.toFixed(2)}`,
    `consumptions_count=${focusedOrganism.consumptions_count}`,
    `reproductions_count=${focusedOrganism.reproductions_count}`,
    `synapse_count=${focusedOrganism.brain.synapse_count}`,
    `vision_distance=${focusedOrganism.genome.vision_distance}`,
    `genome_neurons=${focusedOrganism.genome.num_neurons}`,
    `genome_edges=${focusedOrganism.genome.edges.length}`,
    `mut_weight=${focusedOrganism.genome.mutation_rate_weight.toFixed(3)}`,
    `mut_split_edge=${focusedOrganism.genome.mutation_rate_split_edge.toFixed(3)}`,
  ].join('\n');
}
