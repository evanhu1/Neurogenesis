import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react';
import type { StepProgressData, WorldSnapshot } from '../../types';
import { ControlButton } from './ControlButton';

type SimulationControlsPanelProps = {
  snapshot: WorldSnapshot | null;
  isRunning: boolean;
  isStepPending: boolean;
  stepProgress: StepProgressData | null;
  speedLevelIndex: number;
  speedLevelCount: number;
  onNewSession: (seedInput: string) => void;
  onReset: (seedInput: string) => void;
  onToggleRun: () => void;
  onSpeedLevelChange: (levelIndex: number) => void;
  onStep: (count: number) => void;
  onSaveCurrentWorld: () => void;
};

export function SimulationControlsPanel({
  snapshot,
  isRunning,
  isStepPending,
  stepProgress,
  speedLevelIndex,
  speedLevelCount,
  onNewSession,
  onReset,
  onToggleRun,
  onSpeedLevelChange,
  onStep,
  onSaveCurrentWorld,
}: SimulationControlsPanelProps) {
  const [skipCountInput, setSkipCountInput] = useState('1000');
  const [seedInput, setSeedInput] = useState('');

  const skipProgress = useMemo(() => {
    if (!stepProgress || stepProgress.requested_count <= 1) return null;
    const requested = Math.max(1, stepProgress.requested_count);
    const completed = Math.min(requested, Math.max(0, stepProgress.completed_count));
    const percent = Math.round((completed / requested) * 100);
    return {
      requested,
      completed,
      percent,
    };
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
      <div className="mt-3 flex flex-wrap gap-2">
        <ControlButton label="New Session" onClick={() => onNewSession(seedInput)} />
        <ControlButton label="Reset" onClick={() => onReset(seedInput)} />
        <ControlButton label="Save World" onClick={onSaveCurrentWorld} disabled={!snapshot} />
        <div className="flex w-full items-center gap-2 rounded-lg bg-white/70 px-2 py-1">
          <ControlButton
            label={isRunning ? 'Stop' : 'Start'}
            onClick={onToggleRun}
            disabled={isStepPending}
          />
          <span className="text-xs" role="img" aria-label="turtle">
            🐢
          </span>
          <input
            aria-label="Simulation speed"
            type="range"
            min={0}
            max={speedLevelCount - 1}
            step={1}
            value={speedLevelIndex}
            onChange={handleSpeedLevelInput}
            className="h-1.5 min-w-0 flex-1 cursor-pointer accent-accent"
          />
          <span className="text-xs" role="img" aria-label="rabbit">
            🐇
          </span>
        </div>
      </div>

      <div className="mt-2 flex items-center gap-2 rounded-lg bg-white/70 px-2 py-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-ink/75">Seed</span>
        <input
          aria-label="Session seed"
          type="text"
          inputMode="numeric"
          pattern="[0-9]*"
          value={seedInput}
          onChange={(evt) => setSeedInput(evt.target.value)}
          className="w-48 rounded-md border border-accent/30 bg-white px-2 py-1 font-mono text-sm text-ink outline-none ring-accent/20 focus:ring-2"
        />
      </div>

      <div className="mt-2 flex gap-2">
        <ControlButton label="Step 1" onClick={() => onStep(1)} disabled={isStepPending} />
        <ControlButton label="Step 10" onClick={() => onStep(10)} disabled={isStepPending} />
        <ControlButton label="Step 100" onClick={() => onStep(100)} disabled={isStepPending} />
      </div>

      <div className="mt-2 flex items-center gap-2 rounded-lg bg-white/70 px-2 py-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-ink/75">Skip X</span>
        <input
          aria-label="Skip step count"
          type="number"
          min={1}
          step={1}
          value={skipCountInput}
          disabled={isStepPending}
          onChange={handleSkipCountInput}
          onKeyDown={(evt) => {
            if (evt.key === 'Enter') {
              runSkip();
            }
          }}
          className="w-28 rounded-md border border-accent/30 bg-white px-2 py-1 font-mono text-sm text-ink outline-none ring-accent/20 focus:ring-2 disabled:cursor-not-allowed disabled:opacity-50 disabled:grayscale"
        />
        <ControlButton label="Skip" onClick={runSkip} disabled={isStepPending} />
      </div>

      {skipProgress ? (
        <div className="mt-2 rounded-lg bg-white/70 px-2 py-2">
          <div className="mb-1 flex items-center justify-between font-mono text-[11px] text-ink/75">
            <span>Skipping {skipProgress.requested.toLocaleString()} ticks</span>
            <span>
              {skipProgress.completed.toLocaleString()} / {skipProgress.requested.toLocaleString()}{' '}
              ({skipProgress.percent}%)
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-accent/20">
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-200 ease-out"
              style={{ width: `${skipProgress.percent}%` }}
            />
          </div>
        </div>
      ) : null}
    </>
  );
}
