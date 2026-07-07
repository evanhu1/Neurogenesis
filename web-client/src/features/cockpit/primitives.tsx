// Shared building blocks for the research-cockpit tabs: an inline-SVG sparkline,
// a fetch-on-active/refetch-on-revision hook, and small formatting helpers.
// Styling mirrors the existing panels (JetBrains-mono numerics, uppercase
// tracked section labels, earthy tokens).

import { useCallback, useEffect, useRef, useState } from 'react';

export function fmt(value: number | null | undefined, digits = 3): string {
  return value == null || !Number.isFinite(value) ? 'n/a' : value.toFixed(digits);
}

export function fmtInt(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? 'n/a' : Math.round(value).toLocaleString();
}

export function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-[10px] font-semibold uppercase tracking-[0.18em] text-ink/35">{children}</h3>
  );
}

/** Compact inline sparkline. Nulls/NaNs are treated as gaps. `highlightFrom`
 *  (fraction 0..1 of the series) shades the tail window — used by Pillars to
 *  mark the scoring window behind the windowed mean. */
export function Sparkline({
  values,
  className = 'text-ink/45',
  height = 30,
  highlightFrom,
}: {
  values: (number | null)[];
  className?: string;
  height?: number;
  highlightFrom?: number;
}) {
  const width = 120;
  const finite = values.filter((v): v is number => v != null && Number.isFinite(v));
  if (finite.length === 0) {
    return <div className="text-[10px] text-ink/25">no data</div>;
  }
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || 1;
  const n = values.length;
  const x = (i: number) => (n <= 1 ? 0 : (i / (n - 1)) * width);
  const y = (v: number) => height - 2 - ((v - min) / span) * (height - 4);

  // Build a path, breaking at gaps.
  let d = '';
  let penDown = false;
  values.forEach((v, i) => {
    if (v == null || !Number.isFinite(v)) {
      penDown = false;
      return;
    }
    d += `${penDown ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)} `;
    penDown = true;
  });

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className={`h-[30px] w-full ${className}`}
      aria-hidden="true"
    >
      {highlightFrom != null && highlightFrom < 1 && (
        <rect
          x={(highlightFrom * width).toFixed(1)}
          y={0}
          width={((1 - highlightFrom) * width).toFixed(1)}
          height={height}
          className="fill-accent/10"
        />
      )}
      <path d={d.trim()} fill="none" stroke="currentColor" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** A horizontal proportion bar (0..1). */
export function Bar({ frac, className = 'bg-accent/50' }: { frac: number; className?: string }) {
  const clamped = Math.max(0, Math.min(1, frac));
  return (
    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted/40">
      <div className={`h-full rounded-full ${className}`} style={{ width: `${clamped * 100}%` }} />
    </div>
  );
}

type ReadState<T> = { data: T | null; loading: boolean; error: string | null };

/** Fetch `fetcher()` when the tab is active and the world has a name, and
 *  refetch whenever `revision` changes (i.e. the world file was saved) while
 *  active. Cockpit reads reflect the last persisted state by design. */
export function useWorldRead<T>(
  active: boolean,
  worldName: string | null,
  revision: number,
  fetcher: () => Promise<T>,
): ReadState<T> & { reload: () => void } {
  const [state, setState] = useState<ReadState<T>>({ data: null, loading: false, error: null });
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const reqIdRef = useRef(0);

  const load = useCallback(() => {
    if (!worldName) return;
    const reqId = ++reqIdRef.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    void fetcherRef.current()
      .then((data) => {
        if (reqIdRef.current === reqId) setState({ data, loading: false, error: null });
      })
      .catch((err: unknown) => {
        if (reqIdRef.current === reqId) {
          setState((s) => ({ ...s, loading: false, error: err instanceof Error ? err.message : 'read failed' }));
        }
      });
  }, [worldName]);

  useEffect(() => {
    if (active && worldName) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, worldName, revision]);

  return { ...state, reload: load };
}
