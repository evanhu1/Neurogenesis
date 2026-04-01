import type { LiveMetricsData, OrganismState, SessionMetadata } from '../../types';

export function formatSessionMeta(session: SessionMetadata | null): string {
  if (!session) return 'No session';
  return [
    `session=${session.id}`,
    `created=${new Date(session.created_at_unix_ms).toISOString()}`,
    `running=${session.running}`,
    `ticks_per_second=${session.ticks_per_second}`,
    `stream_mode=${session.stream_mode}`,
  ].join('\n');
}

export function formatRuntimeMetrics(liveMetrics: LiveMetricsData | null): string {
  if (!liveMetrics) return 'No metrics';
  const speciesAlive = Object.keys(liveMetrics.metrics.species_counts).length;
  return [
    `turn=${liveMetrics.turn}`,
    `organisms=${liveMetrics.metrics.organisms}`,
    `species_alive=${speciesAlive}`,
    `total_species_created=${liveMetrics.metrics.total_species_created}`,
    `consumptions_last_turn=${liveMetrics.metrics.consumptions_last_turn}`,
    `predations_last_turn=${liveMetrics.metrics.predations_last_turn ?? 0}`,
    `total_plants_eaten=${liveMetrics.metrics.total_consumptions}`,
    `reproductions_last_turn=${liveMetrics.metrics.reproductions_last_turn}`,
    `starvations_last_turn=${liveMetrics.metrics.starvations_last_turn}`,
    `synapse_ops_last_turn=${liveMetrics.metrics.synapse_ops_last_turn}`,
    `actions_last_turn=${liveMetrics.metrics.actions_applied_last_turn}`,
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
