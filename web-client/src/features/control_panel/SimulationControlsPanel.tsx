import { useEffect, useMemo, useState } from 'react';
import type { SimulationSessionState } from '../sim/hooks/useSimulationSession';
import { ControlButton } from './ControlButton';

const MAX_SKIP_COUNT = 1_000_000_000;

const PlayIcon = (
  <svg aria-hidden="true" viewBox="0 0 16 16" className="h-3 w-3 fill-current">
    <path d="M4.5 2.7v10.6c0 .6.65.95 1.15.63l8.3-5.3a.75.75 0 0 0 0-1.26l-8.3-5.3a.75.75 0 0 0-1.15.63Z" />
  </svg>
);

const PauseIcon = (
  <svg aria-hidden="true" viewBox="0 0 16 16" className="h-3 w-3 fill-current">
    <path d="M4 2.5h2.6v11H4v-11Zm5.4 0H12v11H9.4v-11Z" />
  </svg>
);

const BoltIcon = (
  <svg aria-hidden="true" viewBox="0 0 16 16" className="h-3 w-3 fill-current">
    <path d="M9.2 1.2 3.4 8.9h3.3l-1 5.9 5.9-7.7H8.3l.9-5.9Z" />
  </svg>
);

const INPUT_CLASSES =
  'rounded-lg border border-line bg-surface/60 px-2 py-1 font-mono text-[11px] text-ink/85 outline-none transition focus:border-accent/50 disabled:cursor-not-allowed disabled:opacity-40';

function SectionLabel({ children }: { children: string }) {
  return (
    <h3 className="text-[10px] font-semibold uppercase tracking-[0.18em] text-ink/35">
      {children}
    </h3>
  );
}

type SimulationControlsPanelProps = {
  simulation: SimulationSessionState;
};

export function SimulationControlsPanel({ simulation }: SimulationControlsPanelProps) {
  const {
    snapshot,
    isRunning,
    isStepPending,
    stepProgress,
    speedLevelIndex,
    speedLevels,
    streamMode,
    isFastMode,
  } = simulation;
  const [skipCountInput, setSkipCountInput] = useState('1000');
  const [seedInput, setSeedInput] = useState('');

  const skipProgress = useMemo(() => {
    if (!stepProgress || stepProgress.requested_count <= 1) return null;
    const requested = Math.max(1, stepProgress.requested_count);
    const completed = Math.min(requested, Math.max(0, stepProgress.completed_count));
    return { requested, completed, percent: Math.round((completed / requested) * 100) };
  }, [stepProgress]);

  useEffect(() => {
    if (!snapshot) return;
    setSeedInput(String(snapshot.rng_seed));
  }, [snapshot?.rng_seed]);

  const runSkip = () => {
    if (isStepPending) return;
    const parsed = Number.parseInt(skipCountInput, 10);
    if (!Number.isFinite(parsed)) return;
    const count = Math.max(1, Math.min(parsed, MAX_SKIP_COUNT));
    simulation.step(count);
    setSkipCountInput(String(count));
  };

  const isLiveRunning = isRunning && streamMode === 'full';

  return (
    <section className="mt-3">
      <SectionLabel>Simulation</SectionLabel>

      <div className="mt-1.5 flex gap-1.5">
        <ControlButton
          variant="primary"
          className="flex-1"
          icon={isLiveRunning ? PauseIcon : PlayIcon}
          label={isLiveRunning ? 'Pause' : 'Run'}
          onClick={simulation.toggleRun}
          disabled={isStepPending}
        />
        <ControlButton
          className="flex-1"
          active={isFastMode}
          icon={BoltIcon}
          label={isFastMode ? 'Stop Fast' : 'Fast'}
          onClick={simulation.toggleFastRun}
          disabled={isStepPending}
        />
      </div>

      <div className="mt-2 flex items-center gap-2">
        <span className="text-[10px] font-medium uppercase tracking-wide text-ink/35">Speed</span>
        <input
          aria-label="Simulation speed"
          type="range"
          min={0}
          max={speedLevels.length - 1}
          step={1}
          value={speedLevelIndex}
          onChange={(evt) => {
            const level = Number.parseInt(evt.target.value, 10);
            if (Number.isFinite(level)) simulation.setSpeedLevelIndex(level);
          }}
          disabled={isFastMode}
          className="h-1 min-w-0 flex-1 cursor-pointer disabled:cursor-not-allowed disabled:opacity-40"
        />
        <span className="w-12 text-right font-mono text-[10px] text-ink/55">
          {speedLevels[speedLevelIndex]} t/s
        </span>
      </div>

      <div className="mt-2 flex items-center gap-1.5">
        <span className="text-[10px] font-medium uppercase tracking-wide text-ink/35">Step</span>
        <ControlButton label="×1" onClick={() => simulation.step(1)} disabled={isStepPending} />
        <ControlButton label="×10" onClick={() => simulation.step(10)} disabled={isStepPending} />
        <ControlButton label="×100" onClick={() => simulation.step(100)} disabled={isStepPending} />
        <input
          aria-label="Skip step count"
          type="number"
          min={1}
          step={1}
          value={skipCountInput}
          disabled={isStepPending}
          onChange={(evt) => setSkipCountInput(evt.target.value)}
          onKeyDown={(evt) => {
            if (evt.key === 'Enter') runSkip();
          }}
          className={`ml-auto w-[72px] ${INPUT_CLASSES}`}
        />
        <ControlButton label="Skip" onClick={runSkip} disabled={isStepPending} />
      </div>

      {skipProgress && (
        <div className="mt-2">
          <div className="flex items-center justify-between font-mono text-[10px] text-ink/35">
            <span>
              {skipProgress.completed.toLocaleString()} / {skipProgress.requested.toLocaleString()}
            </span>
            <span>{skipProgress.percent}%</span>
          </div>
          <div className="mt-0.5 h-1 overflow-hidden rounded-full bg-muted/40">
            <div
              className="h-full rounded-full bg-accent/60 transition-[width] duration-200"
              style={{ width: `${skipProgress.percent}%` }}
            />
          </div>
        </div>
      )}

      <div className="mt-3">
        <SectionLabel>Session</SectionLabel>
        <div className="mt-1.5 flex items-center gap-1.5">
          <input
            aria-label="Session seed"
            type="text"
            inputMode="numeric"
            pattern="[0-9]*"
            value={seedInput}
            onChange={(evt) => setSeedInput(evt.target.value)}
            placeholder="Seed"
            className={`min-w-0 flex-1 ${INPUT_CLASSES}`}
          />
          <ControlButton label="New" onClick={() => void simulation.createSession(seedInput)} />
          <ControlButton
            label="Save"
            onClick={() => void simulation.saveChampions()}
            disabled={!snapshot}
          />
        </div>
      </div>
    </section>
  );
}
