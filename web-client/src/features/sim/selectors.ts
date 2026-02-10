import { unwrapId } from '../../protocol';
import type { OrganismState, SessionMetadata, WorldSnapshot } from '../../types';

export function formatSessionMeta(session: SessionMetadata | null): string {
  if (!session) return 'No session';
  return `session=${session.id}\ncreated=${new Date(session.created_at_unix_ms).toISOString()}`;
}

export function formatMetrics(snapshot: WorldSnapshot | null): string {
  if (!snapshot) return 'No metrics';
  return [
    `turn=${snapshot.turn}`,
    `organisms=${snapshot.metrics.organisms}`,
    `consumptions_last_turn=${snapshot.metrics.consumptions_last_turn}`,
    `starvations_last_turn=${snapshot.metrics.starvations_last_turn}`,
    `births_last_turn=${snapshot.metrics.births_last_turn}`,
    `synapse_ops_last_turn=${snapshot.metrics.synapse_ops_last_turn}`,
    `actions_last_turn=${snapshot.metrics.actions_applied_last_turn}`,
  ].join('\n');
}

export function formatFitnessStats(snapshot: WorldSnapshot | null): string {
  if (!snapshot) return 'No fitness stats';
  const fitness = snapshot.metrics.fitness;
  return [
    `mean_fitness=${fitness.mean_fitness.toFixed(2)}`,
    `median_fitness=${fitness.median_fitness.toFixed(2)}`,
    `max_fitness=${fitness.max_fitness}`,
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
    `position=(${focusedOrganism.q}, ${focusedOrganism.r})`,
    `facing=${focusedOrganism.facing}`,
    `age_turns=${focusedOrganism.age_turns}`,
    `turns_since_last_consumption=${focusedOrganism.turns_since_last_consumption}`,
    `consumptions_count=${focusedOrganism.consumptions_count}`,
    `synapse_count=${focusedOrganism.brain.synapse_count}`,
  ].join('\n');
}
