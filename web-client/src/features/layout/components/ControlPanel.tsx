import type { ChangeEvent } from 'react';

type ControlPanelProps = {
  sessionMeta: string;
  evolutionStatsText: string;
  metricsText: string;
  errorText: string | null;
  isRunning: boolean;
  speedLevelIndex: number;
  speedLevelCount: number;
  onNewSession: () => void;
  onReset: () => void;
  onToggleRun: () => void;
  onSpeedLevelChange: (levelIndex: number) => void;
  onStep: (count: number) => void;
};

export function ControlPanel({
  sessionMeta,
  evolutionStatsText,
  metricsText,
  errorText,
  isRunning,
  speedLevelIndex,
  speedLevelCount,
  onNewSession,
  onReset,
  onToggleRun,
  onSpeedLevelChange,
  onStep,
}: ControlPanelProps) {
  const handleSpeedLevelInput = (evt: ChangeEvent<HTMLInputElement>) => {
    const rawLevel = Number.parseInt(evt.target.value, 10);
    if (!Number.isFinite(rawLevel)) return;
    onSpeedLevelChange(rawLevel);
  };

  return (
    <aside className="h-full overflow-auto rounded-2xl border border-accent/15 bg-panel/95 p-4 shadow-panel">
      <h1 className="text-2xl font-semibold tracking-tight">NeuroGenesis</h1>
      <pre className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-100/80 p-3 font-mono text-xs">
        {sessionMeta}
      </pre>

      <div className="mt-3 flex flex-wrap gap-2">
        <ControlButton label="New Session" onClick={onNewSession} />
        <ControlButton label="Reset" onClick={onReset} />
        <div className="flex items-center gap-2 rounded-lg border border-accent/20 bg-white/70 px-2 py-1">
          <ControlButton label={isRunning ? 'Pause' : 'Start'} onClick={onToggleRun} />
          <div className="flex items-center gap-2">
            <div className="flex h-12 w-6 items-center justify-center">
              <input
                aria-label="Simulation speed"
                type="range"
                min={0}
                max={speedLevelCount - 1}
                step={1}
                value={speedLevelIndex}
                onChange={handleSpeedLevelInput}
                className="h-2 w-24 -rotate-90 cursor-pointer accent-accent"
              />
            </div>
            <div className="flex h-24 flex-col justify-between">
              <div className="flex items-center gap-1 text-sm leading-none">
                <span className="inline-block h-0.5 w-2 rounded-full bg-slate-500" />
                <span role="img" aria-label="rabbit">
                  🐇
                </span>
              </div>
              <div className="flex items-center gap-1 text-sm leading-none">
                <span className="inline-block h-0.5 w-2 rounded-full bg-slate-500" />
              </div>
              <div className="flex items-center gap-1 text-sm leading-none">
                <span className="inline-block h-0.5 w-2 rounded-full bg-slate-500" />
                <span role="img" aria-label="turtle">
                  🐢
                </span>
              </div>
            </div>
          </div>
        </div>
        <ControlButton label="Step 1" onClick={() => onStep(1)} />
        <ControlButton label="Step 10" onClick={() => onStep(10)} />
        <ControlButton label="Step 100" onClick={() => onStep(100)} />
      </div>

      <h3 className="mt-3 text-sm font-semibold uppercase tracking-wide text-ink/80">Evolution Stats</h3>
      <pre className="mt-2 whitespace-pre-wrap rounded-xl bg-slate-100/80 p-3 font-mono text-xs">
        {evolutionStatsText}
      </pre>

      <h3 className="mt-3 text-sm font-semibold uppercase tracking-wide text-ink/80">Runtime Metrics</h3>
      <pre className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-100/80 p-3 font-mono text-xs">
        {metricsText}
      </pre>

      {errorText ? (
        <div className="mt-3 rounded-xl border border-rose-300 bg-rose-50 px-3 py-2 font-mono text-xs text-rose-700">
          {errorText}
        </div>
      ) : null}
    </aside>
  );
}

type ControlButtonProps = {
  label: string;
  onClick: () => void;
};

function ControlButton({ label, onClick }: ControlButtonProps) {
  return (
    <button
      onClick={onClick}
      className="rounded-lg bg-accent px-3 py-2 text-sm font-semibold text-white transition hover:brightness-110"
    >
      {label}
    </button>
  );
}

