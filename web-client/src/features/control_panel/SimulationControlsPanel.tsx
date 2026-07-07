import { useEffect, useState } from 'react';
import type { WorldController } from '../sim/hooks/useWorld';
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
  world: WorldController;
};

export function SimulationControlsPanel({ world }: SimulationControlsPanelProps) {
  const { snapshot, isRunning, isStepPending, speedLevelIndex, speedLevels } = world;
  const [skipCountInput, setSkipCountInput] = useState('1000');
  const [seedInput, setSeedInput] = useState('');

  // Stepping mutates the file synchronously and is disabled while running.
  const stepDisabled = isRunning || isStepPending;

  useEffect(() => {
    if (!snapshot) return;
    setSeedInput(String(snapshot.rng_seed));
  }, [snapshot?.rng_seed]);

  const runSkip = () => {
    if (stepDisabled) return;
    const parsed = Number.parseInt(skipCountInput, 10);
    if (!Number.isFinite(parsed)) return;
    const count = Math.max(1, Math.min(parsed, MAX_SKIP_COUNT));
    world.step(count);
    setSkipCountInput(String(count));
  };

  return (
    <section className="mt-3">
      <SectionLabel>Simulation</SectionLabel>

      <div className="mt-1.5 flex gap-1.5">
        <ControlButton
          variant="primary"
          className="flex-1"
          icon={isRunning ? PauseIcon : PlayIcon}
          label={isRunning ? 'Pause' : 'Run'}
          onClick={world.toggleRun}
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
            if (Number.isFinite(level)) world.setSpeedLevelIndex(level);
          }}
          className="h-1 min-w-0 flex-1 cursor-pointer"
        />
        <span className="w-12 text-right font-mono text-[10px] text-ink/55">
          {speedLevels[speedLevelIndex]} t/s
        </span>
      </div>

      <div className="mt-2 flex items-center gap-1.5">
        <span className="text-[10px] font-medium uppercase tracking-wide text-ink/35">Step</span>
        <ControlButton label="×1" onClick={() => world.step(1)} disabled={stepDisabled} />
        <ControlButton label="×10" onClick={() => world.step(10)} disabled={stepDisabled} />
        <ControlButton label="×100" onClick={() => world.step(100)} disabled={stepDisabled} />
        <input
          aria-label="Skip step count"
          type="number"
          min={1}
          step={1}
          value={skipCountInput}
          disabled={stepDisabled}
          onChange={(evt) => setSkipCountInput(evt.target.value)}
          onKeyDown={(evt) => {
            if (evt.key === 'Enter') runSkip();
          }}
          className={`ml-auto w-[72px] ${INPUT_CLASSES}`}
        />
        <ControlButton label="Skip" onClick={runSkip} disabled={stepDisabled} />
      </div>

      {isStepPending && (
        <div className="mt-2 font-mono text-[10px] text-ink/35">stepping…</div>
      )}

      <div className="mt-3">
        <SectionLabel>World</SectionLabel>
        <div className="mt-1.5 flex items-center gap-1.5">
          <input
            aria-label="World seed"
            type="text"
            inputMode="numeric"
            pattern="[0-9]*"
            value={seedInput}
            onChange={(evt) => setSeedInput(evt.target.value)}
            placeholder="Seed"
            className={`min-w-0 flex-1 ${INPUT_CLASSES}`}
          />
          <ControlButton label="New" onClick={() => void world.createWorld(seedInput)} />
          <ControlButton
            label="Save"
            onClick={() => void world.saveChampions()}
            disabled={!snapshot}
          />
        </div>
      </div>
    </section>
  );
}
