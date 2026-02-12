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
