import { useCallback, useState } from 'react';
import type { StepProgressData } from '../../../types';
import { SPEED_LEVELS } from '../constants';

const DEFAULT_SPEED_LEVEL_INDEX = 1;

function nearestSpeedLevelIndex(ticksPerSecond: number): number {
  if (!Number.isFinite(ticksPerSecond) || ticksPerSecond <= 0) return 0;
  let bestIndex = 0;
  let bestDistance = Math.abs(SPEED_LEVELS[0] - ticksPerSecond);
  for (let index = 1; index < SPEED_LEVELS.length; index += 1) {
    const distance = Math.abs(SPEED_LEVELS[index] - ticksPerSecond);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function speedLevelIndexForTicksPerSecond(ticksPerSecond: number): number {
  return ticksPerSecond > 0
    ? nearestSpeedLevelIndex(ticksPerSecond)
    : DEFAULT_SPEED_LEVEL_INDEX;
}

export function useSimulationControls(sendCommand: (command: unknown) => boolean) {
  const [isRunning, setIsRunning] = useState(false);
  const [isStepPending, setIsStepPending] = useState(false);
  const [stepProgress, setStepProgress] = useState<StepProgressData | null>(null);
  const [speedLevelIndex, setSpeedLevelIndex] = useState(DEFAULT_SPEED_LEVEL_INDEX);

  const clearPendingStep = useCallback(() => {
    setIsStepPending(false);
    setStepProgress(null);
  }, []);

  const handleStepProgress = useCallback((progress: StepProgressData) => {
    setStepProgress(progress);
    setIsStepPending(progress.completed_count < progress.requested_count);
  }, []);

  const handleSocketClose = useCallback(() => {
    setIsRunning(false);
    setStepProgress(null);
  }, []);

  const syncSessionState = useCallback(
    (running: boolean, ticksPerSecond: number) => {
      setIsRunning(running);
      clearPendingStep();
      setSpeedLevelIndex(speedLevelIndexForTicksPerSecond(ticksPerSecond));
    },
    [clearPendingStep],
  );

  const setSpeedLevel = useCallback(
    (levelIndex: number) => {
      const nextLevel = Math.max(0, Math.min(SPEED_LEVELS.length - 1, levelIndex));
      setSpeedLevelIndex(nextLevel);
      if (isRunning) {
        sendCommand({ type: 'Start', data: { ticks_per_second: SPEED_LEVELS[nextLevel] } });
      }
    },
    [isRunning, sendCommand],
  );

  const toggleRun = useCallback(() => {
    setIsRunning((currentlyRunning) => {
      const nextCommand = currentlyRunning
        ? { type: 'Pause' }
        : { type: 'Start', data: { ticks_per_second: SPEED_LEVELS[speedLevelIndex] } };
      return sendCommand(nextCommand) ? !currentlyRunning : currentlyRunning;
    });
  }, [sendCommand, speedLevelIndex]);

  const step = useCallback(
    (count: number) => {
      const requestedCount = Math.floor(count);
      if (!Number.isFinite(requestedCount)) return;
      const sent = sendCommand({ type: 'Step', data: { count: Math.max(1, requestedCount) } });
      if (sent) {
        setIsStepPending(true);
        setStepProgress(
          requestedCount > 1
            ? {
                requested_count: requestedCount,
                completed_count: 0,
              }
            : null,
        );
      }
    },
    [sendCommand],
  );

  return {
    isRunning,
    isStepPending,
    stepProgress,
    speedLevels: SPEED_LEVELS,
    speedLevelIndex,
    clearPendingStep,
    handleSocketClose,
    handleStepProgress,
    syncSessionState,
    toggleRun,
    setSpeedLevelIndex: setSpeedLevel,
    step,
  };
}
