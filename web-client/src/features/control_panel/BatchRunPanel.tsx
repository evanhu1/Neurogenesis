import { useCallback, useMemo, useState } from 'react';
import type { BatchRunStatusResponse } from '../../types';
import { ControlButton } from './ControlButton';

type BatchRunPanelProps = {
  batchRunStatus: BatchRunStatusResponse | null;
  onStartBatchRun: (worldCount: number, ticksPerWorld: number) => void;
};

export function BatchRunPanel({
  batchRunStatus,
  onStartBatchRun,
}: BatchRunPanelProps) {
  const [worldCountInput, setWorldCountInput] = useState('8');
  const [ticksInput, setTicksInput] = useState('1000');
  const isBatchRunning = batchRunStatus?.status === 'Running';

  const batchProgress = useMemo(() => {
    if (!batchRunStatus) return null;
    const total = Math.max(1, batchRunStatus.total_worlds);
    const completed = Math.max(0, Math.min(total, batchRunStatus.completed_worlds));
    const percent = Math.round((completed / total) * 100);
    return { total, completed, percent };
  }, [batchRunStatus]);

  const runBatch = useCallback(() => {
    if (isBatchRunning) return;
    const parsedWorldCount = Number.parseInt(worldCountInput, 10);
    const parsedTicks = Number.parseInt(ticksInput, 10);
    if (!Number.isFinite(parsedWorldCount) || !Number.isFinite(parsedTicks)) return;
    const worldCount = Math.max(1, parsedWorldCount);
    const ticks = Math.max(1, parsedTicks);
    onStartBatchRun(worldCount, ticks);
    setWorldCountInput(String(worldCount));
    setTicksInput(String(ticks));
  }, [isBatchRunning, onStartBatchRun, ticksInput, worldCountInput]);

  return (
    <>
      <h3 className="mt-3 text-sm font-semibold uppercase tracking-wide text-ink/80">World Batch</h3>
      <div className="mt-2 space-y-2 rounded-lg bg-white/70 px-2 py-2">
        <div className="grid grid-cols-2 gap-2">
          <label className="text-[11px] text-ink/70">
            N
            <input
              type="number"
              min={1}
              step={1}
              value={worldCountInput}
              disabled={isBatchRunning}
              onChange={(evt) => setWorldCountInput(evt.target.value)}
              className="mt-1 w-full rounded-md border border-accent/30 bg-white px-2 py-1 font-mono text-sm text-ink outline-none ring-accent/20 focus:ring-2 disabled:cursor-not-allowed disabled:opacity-50 disabled:grayscale"
            />
          </label>
          <label className="text-[11px] text-ink/70">
            Ticks
            <input
              type="number"
              min={1}
              step={1}
              value={ticksInput}
              disabled={isBatchRunning}
              onChange={(evt) => setTicksInput(evt.target.value)}
              className="mt-1 w-full rounded-md border border-accent/30 bg-white px-2 py-1 font-mono text-sm text-ink outline-none ring-accent/20 focus:ring-2 disabled:cursor-not-allowed disabled:opacity-50 disabled:grayscale"
            />
          </label>
        </div>
        <ControlButton
          label={isBatchRunning ? 'Running...' : 'Run Worlds'}
          onClick={runBatch}
          disabled={isBatchRunning}
        />
        {batchProgress ? (
          <div className="rounded-lg bg-accent/10 px-2 py-2">
            <div className="mb-1 flex items-center justify-between font-mono text-[11px] text-ink/80">
              <span>Status: {batchRunStatus?.status ?? 'Unknown'}</span>
              <span>
                {batchProgress.completed.toLocaleString()} / {batchProgress.total.toLocaleString()} (
                {batchProgress.percent}%)
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-accent/20">
              <div
                className="h-full rounded-full bg-accent transition-[width] duration-200 ease-out"
                style={{ width: `${batchProgress.percent}%` }}
              />
            </div>
          </div>
        ) : null}
      </div>

      {batchRunStatus?.status === 'Completed' && batchRunStatus.aggregate ? (
        <div className="mt-2 rounded-lg border border-accent/25 bg-accent/10 px-3 py-2">
          <div className="text-xs font-semibold uppercase tracking-wide text-ink/80">
            Batch Aggregate
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 font-mono text-xs">
            <div>Total organisms: {batchRunStatus.aggregate.total_organisms_alive.toLocaleString()}</div>
            <div>Total species: {batchRunStatus.aggregate.total_species_alive.toLocaleString()}</div>
            <div>Mean organisms: {batchRunStatus.aggregate.mean_organisms_alive.toFixed(2)}</div>
            <div>Mean species: {batchRunStatus.aggregate.mean_species_alive.toFixed(2)}</div>
          </div>
        </div>
      ) : null}
    </>
  );
}
