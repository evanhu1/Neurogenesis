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
    `meals_last_turn=${snapshot.metrics.meals_last_turn}`,
    `starvations_last_turn=${snapshot.metrics.starvations_last_turn}`,
    `births_last_turn=${snapshot.metrics.births_last_turn}`,
    `synapse_ops_last_turn=${snapshot.metrics.synapse_ops_last_turn}`,
    `actions_last_turn=${snapshot.metrics.actions_applied_last_turn}`,
  ].join('\n');
}

export function formatEvolutionStats(snapshot: WorldSnapshot | null): string {
  if (!snapshot) return 'No evolution stats';
  const evolution = snapshot.metrics.evolution;
  return [
    `mean_age_turns=${evolution.mean_age_turns.toFixed(2)}`,
    `median_age_turns=${evolution.median_age_turns.toFixed(2)}`,
    `max_age_turns=${evolution.max_age_turns}`,
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
    `turns_since_last_meal=${focusedOrganism.turns_since_last_meal}`,
    `meals_eaten=${focusedOrganism.meals_eaten}`,
    `synapse_count=${focusedOrganism.brain.synapse_count}`,
  ].join('\n');
}

