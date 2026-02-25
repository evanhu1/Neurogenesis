type PerfBucket = {
  count: number;
  totalMs: number;
  maxMs: number;
  lastMs: number;
  emaMs: number;
};

type PerfState = {
  worldDraw: PerfBucket;
  tickDeltaApply: PerfBucket;
  reactCommit: PerfBucket;
};

type PerfSnapshot = {
  timestampMs: number;
  worldDraw: PerfBucket;
  tickDeltaApply: PerfBucket;
  reactCommit: PerfBucket;
};

const EMA_ALPHA = 0.2;

const perfState: PerfState = {
  worldDraw: createBucket(),
  tickDeltaApply: createBucket(),
  reactCommit: createBucket(),
};

function createBucket(): PerfBucket {
  return {
    count: 0,
    totalMs: 0,
    maxMs: 0,
    lastMs: 0,
    emaMs: 0,
  };
}

function updateBucket(bucket: PerfBucket, durationMs: number) {
  if (!Number.isFinite(durationMs) || durationMs < 0) return;
  bucket.count += 1;
  bucket.totalMs += durationMs;
  bucket.lastMs = durationMs;
  bucket.maxMs = Math.max(bucket.maxMs, durationMs);
  bucket.emaMs = bucket.count === 1 ? durationMs : bucket.emaMs + (durationMs - bucket.emaMs) * EMA_ALPHA;
}

function cloneBucket(bucket: PerfBucket): PerfBucket {
  return {
    count: bucket.count,
    totalMs: bucket.totalMs,
    maxMs: bucket.maxMs,
    lastMs: bucket.lastMs,
    emaMs: bucket.emaMs,
  };
}

function ensureWindowApi() {
  if (typeof window === 'undefined') return;
  const win = window as typeof window & {
    __neuroPerf?: {
      snapshot: () => PerfSnapshot;
      reset: () => void;
    };
  };
  if (win.__neuroPerf) return;
  win.__neuroPerf = {
    snapshot: getPerfSnapshot,
    reset: resetPerf,
  };
}

export function recordWorldDraw(durationMs: number) {
  updateBucket(perfState.worldDraw, durationMs);
  ensureWindowApi();
}

export function recordTickDeltaApply(durationMs: number) {
  updateBucket(perfState.tickDeltaApply, durationMs);
  ensureWindowApi();
}

export function recordReactCommit(durationMs: number) {
  updateBucket(perfState.reactCommit, durationMs);
  ensureWindowApi();
}

export function getPerfSnapshot(): PerfSnapshot {
  return {
    timestampMs: Date.now(),
    worldDraw: cloneBucket(perfState.worldDraw),
    tickDeltaApply: cloneBucket(perfState.tickDeltaApply),
    reactCommit: cloneBucket(perfState.reactCommit),
  };
}

export function resetPerf() {
  perfState.worldDraw = createBucket();
  perfState.tickDeltaApply = createBucket();
  perfState.reactCommit = createBucket();
}
