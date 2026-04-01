import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react';
import type { StepProgressData, StreamMode, WorldSnapshot } from '../../types';
import { ControlButton } from './ControlButton';

type SimulationControlsPanelProps = {
  snapshot: WorldSnapshot | null;
  isRunning: boolean;
  isStepPending: boolean;
  stepProgress: StepProgressData | null;
  speedLevelIndex: number;
  speedLevelCount: number;
  streamMode: StreamMode;
  isFastMode: boolean;
  onNewSession: (seedInput: string) => void;
  onSaveChampions: () => void;
  onToggleRun: () => void;
  onToggleFastRun: () => void;
  onSpeedLevelChange: (levelIndex: number) => void;
  onStep: (count: number) => void;
};

export function SimulationControlsPanel({
  snapshot,
  isRunning,
  isStepPending,
  stepProgress,
  speedLevelIndex,
  speedLevelCount,
  streamMode,
  isFastMode,
  onNewSession,
  onSaveChampions,
  onToggleRun,
  onToggleFastRun,
  onSpeedLevelChange,
  onStep,
}: SimulationControlsPanelProps) {
  const [skipCountInput, setSkipCountInput] = useState('1000');
  const [seedInput, setSeedInput] = useState('');

  const skipProgress = useMemo(() => {
    if (!stepProgress || stepProgress.requested_count <= 1) return null;
    const requested = Math.max(1, stepProgress.requested_count);
    const completed = Math.min(requested, Math.max(0, stepProgress.completed_count));
    const percent = Math.round((completed / requested) * 100);
    return { requested, completed, percent };
  }, [stepProgress]);

  useEffect(() => {
    if (!snapshot) return;
    setSeedInput(String(snapshot.rng_seed));
  }, [snapshot?.rng_seed]);

  const handleSpeedLevelInput = useCallback(
    (evt: ChangeEvent<HTMLInputElement>) => {
      const rawLevel = Number.parseInt(evt.target.value, 10);
      if (!Number.isFinite(rawLevel)) return;
      onSpeedLevelChange(rawLevel);
    },
    [onSpeedLevelChange],
  );

  const handleSkipCountInput = useCallback((evt: ChangeEvent<HTMLInputElement>) => {
    setSkipCountInput(evt.target.value);
  }, []);

  const runSkip = useCallback(() => {
    if (isStepPending) return;
    const parsed = Number.parseInt(skipCountInput, 10);
    if (!Number.isFinite(parsed)) return;
    const count = Math.max(1, Math.min(parsed, 1_000_000_000));
    onStep(count);
    setSkipCountInput(String(count));
  }, [isStepPending, onStep, skipCountInput]);

  return (
    <>
      <div className="mt-2 flex flex-wrap gap-1.5">
        <ControlButton label="New" onClick={() => onNewSession(seedInput)} />
        <ControlButton label="Save" onClick={onSaveChampions} disabled={!snapshot} />
        <ControlButton
          label={isFastMode ? 'Stop Fast' : 'Fast'}
          onClick={onToggleFastRun}
          disabled={isStepPending}
        />
      </div>

      <div className="mt-1.5 flex items-center gap-2">
        <ControlButton
          label={isRunning && streamMode === 'full' ? 'Stop' : 'Start'}
          onClick={onToggleRun}
          disabled={isStepPending}
        />
        <input
          aria-label="Simulation speed"
          type="range"
          min={0}
          max={speedLevelCount - 1}
          step={1}
          value={speedLevelIndex}
          onChange={handleSpeedLevelInput}
          disabled={isFastMode}
          className="h-1 min-w-0 flex-1 cursor-pointer disabled:cursor-not-allowed disabled:opacity-40"
        />
      </div>

      <div className="mt-1.5 flex items-center gap-2">
        <span className="text-[10px] font-medium uppercase tracking-wide text-ink/35">Seed</span>
        <input
          aria-label="Session seed"
          type="text"
          inputMode="numeric"
          pattern="[0-9]*"
          value={seedInput}
          onChange={(evt) => setSeedInput(evt.target.value)}
          className="min-w-0 flex-1 rounded border border-muted/60 bg-surface px-2 py-0.5 font-mono text-[11px] text-ink/80 outline-none focus:border-accent/40"
        />
      </div>

      <div className="mt-1.5 flex items-center gap-1.5">
        <ControlButton label="×1" onClick={() => onStep(1)} disabled={isStepPending} />
        <ControlButton label="×10" onClick={() => onStep(10)} disabled={isStepPending} />
        <ControlButton label="×100" onClick={() => onStep(100)} disabled={isStepPending} />
        <div className="ml-auto flex items-center gap-1.5">
          <input
            aria-label="Skip step count"
            type="number"
            min={1}
            step={1}
            value={skipCountInput}
            disabled={isStepPending}
            onChange={handleSkipCountInput}
            onKeyDown={(evt) => {
              if (evt.key === 'Enter') runSkip();
            }}
            className="w-20 rounded border border-muted/60 bg-surface px-2 py-0.5 font-mono text-[11px] text-ink/80 outline-none focus:border-accent/40 disabled:cursor-not-allowed disabled:opacity-40"
          />
          <ControlButton label="Skip" onClick={runSkip} disabled={isStepPending} />
        </div>
      </div>

      {skipProgress && (
        <div className="mt-1.5">
          <div className="flex items-center justify-between font-mono text-[10px] text-ink/35">
            <span>
              {skipProgress.completed.toLocaleString()} / {skipProgress.requested.toLocaleString()}
            </span>
            <span>{skipProgress.percent}%</span>
          </div>
          <div className="mt-0.5 h-1 overflow-hidden rounded-full bg-muted/40">
            <div
              className="h-full rounded-full bg-accent/50 transition-[width] duration-200"
              style={{ width: `${skipProgress.percent}%` }}
            />
          </div>
        </div>
      )}
    </>
  );
}
