import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type MouseEvent,
  type RefObject,
} from 'react';
import {
  buildHexLayout,
  createWorldRenderCache,
  hexCenter,
  pickOrganismAtCanvasPoint,
  renderWorld,
} from '../worldCanvas';
import { unwrapId } from '../../../protocol';
import type { WorldOrganismState, WorldSnapshot } from '../../../types';
import { useWorldViewport } from '../hooks/useWorldViewport';

type WorldCanvasProps = {
  snapshot: WorldSnapshot | null;
  focusedOrganismId: number | null;
  showFastOverlay?: boolean;
  onOrganismSelect: (organism: WorldOrganismState) => void;
  panToHexRef?: RefObject<((q: number, r: number) => void) | null>;
};

export function WorldCanvas({
  snapshot,
  focusedOrganismId,
  showFastOverlay = false,
  onOrganismSelect,
  panToHexRef,
}: WorldCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameRequestRef = useRef<number | null>(null);
  const needsRenderRef = useRef(true);
  const requestRenderCallbackRef = useRef<() => void>(() => {});
  const displaySizeRef = useRef({ width: 0, height: 0, dpr: 0 });
  const renderCacheRef = useRef(createWorldRenderCache());
  const snapshotRef = useRef<WorldSnapshot | null>(snapshot);
  const focusedOrganismIdRef = useRef<number | null>(focusedOrganismId);
  const onOrganismSelectRef = useRef(onOrganismSelect);
  const hasAutoFitRef = useRef(false);
  const [showOrganisms, setShowOrganisms] = useState(true);
  const [showPlants, setShowPlants] = useState(true);
  const showOrganismsRef = useRef(showOrganisms);
  const showPlantsRef = useRef(showPlants);
  const onViewportChange = useCallback(() => {
    requestRenderCallbackRef.current();
  }, []);
  const {
    viewportRef,
    isSpacePressed,
    cursorClass,
    zoomAtPointer,
    fitWorldToCanvas,
    panToWorldPoint,
    onCanvasMouseDown,
    onCanvasMouseMove,
    onCanvasMouseUp,
    consumeSuppressedClick,
  } = useWorldViewport({ onViewportChange });
  const syncCanvasDisplaySize = useCallback((canvas: HTMLCanvasElement) => {
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const displayHeight = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    const previous = displaySizeRef.current;
    const changed =
      previous.width !== displayWidth || previous.height !== displayHeight || previous.dpr !== dpr;

    if (!changed) return false;

    displaySizeRef.current = { width: displayWidth, height: displayHeight, dpr };
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }
    return true;
  }, []);
  const drawCurrentFrame = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !needsRenderRef.current) return;

    needsRenderRef.current = false;
    syncCanvasDisplaySize(canvas);

    const context = canvas.getContext('2d');
    if (!context) return;

    renderWorld(context, canvas, snapshotRef.current, focusedOrganismIdRef.current, viewportRef.current, {
      organisms: showOrganismsRef.current,
      plants: showPlantsRef.current,
    }, renderCacheRef.current);
  }, [syncCanvasDisplaySize, viewportRef]);
  const requestRender = useCallback(() => {
    needsRenderRef.current = true;
    if (frameRequestRef.current != null || document.visibilityState === 'hidden') return;

    frameRequestRef.current = requestAnimationFrame(() => {
      frameRequestRef.current = null;
      drawCurrentFrame();
      if (needsRenderRef.current) {
        requestRender();
      }
    });
  }, [drawCurrentFrame]);
  requestRenderCallbackRef.current = requestRender;

  // Keep refs synchronized during render so the RAF draw loop sees latest props immediately.
  snapshotRef.current = snapshot;
  focusedOrganismIdRef.current = focusedOrganismId;
  onOrganismSelectRef.current = onOrganismSelect;
  showOrganismsRef.current = showOrganisms;
  showPlantsRef.current = showPlants;

  useEffect(() => {
    hasAutoFitRef.current = false;
  }, [snapshot?.config.world_width]);

  useEffect(() => {
    requestRender();
  }, [requestRender, snapshot, focusedOrganismId, showOrganisms, showPlants]);

  useEffect(() => {
    if (!snapshot || hasAutoFitRef.current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    syncCanvasDisplaySize(canvas);

    fitWorldToCanvas(canvas, snapshot.config.world_width);
    hasAutoFitRef.current = true;
  }, [fitWorldToCanvas, snapshot, syncCanvasDisplaySize]);

  useEffect(() => {
    if (!panToHexRef) return;
    panToHexRef.current = (q: number, r: number) => {
      const canvas = canvasRef.current;
      const worldWidth = snapshotRef.current?.config.world_width;
      if (!canvas || !worldWidth) return;
      const layout = buildHexLayout(canvas.width, canvas.height, worldWidth);
      const center = hexCenter(layout, q, r);
      panToWorldPoint(center.x, center.y, canvas.width, canvas.height);
    };
    return () => {
      panToHexRef.current = null;
    };
  }, [panToHexRef, panToWorldPoint]);

  // Track focused organism across ticks
  useEffect(() => {
    if (!snapshot || focusedOrganismId == null) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const org = snapshot.organisms.find((o) => unwrapId(o.id) === focusedOrganismId);
    if (!org) return;
    const layout = buildHexLayout(canvas.width, canvas.height, snapshot.config.world_width);
    const center = hexCenter(layout, org.q, org.r);
    panToWorldPoint(center.x, center.y, canvas.width, canvas.height);
  }, [snapshot, focusedOrganismId, panToWorldPoint]);

  const onCanvasClick = useCallback(
    (evt: MouseEvent<HTMLCanvasElement>) => {
      if (consumeSuppressedClick()) return;
      if (isSpacePressed) return;
      if (showFastOverlay) return;
      if (!showOrganismsRef.current) return;

      const activeSnapshot = snapshotRef.current;
      if (!activeSnapshot) return;

      const canvas = evt.currentTarget;
      const rect = canvas.getBoundingClientRect();
      const xPx = ((evt.clientX - rect.left) / rect.width) * canvas.width;
      const yPx = ((evt.clientY - rect.top) / rect.height) * canvas.height;

      const picked = pickOrganismAtCanvasPoint(
        activeSnapshot,
        canvas.width,
        canvas.height,
        xPx,
        yPx,
        viewportRef.current,
      );
      if (!picked) return;
      onOrganismSelectRef.current(picked.organism);
    },
    [consumeSuppressedClick, isSpacePressed, showFastOverlay, viewportRef],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (evt: WheelEvent) => {
      evt.preventDefault();
      zoomAtPointer(
        canvas,
        evt.clientX,
        evt.clientY,
        evt.deltaY,
        snapshotRef.current?.config.world_width ?? null,
      );
    };

    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      canvas.removeEventListener('wheel', onWheel);
    };
  }, [zoomAtPointer]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observedElement = canvas.parentElement ?? canvas;

    const resizeObserver = new ResizeObserver(() => {
      if (!syncCanvasDisplaySize(canvas)) return;
      requestRender();
    });
    resizeObserver.observe(observedElement);

    const onVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        requestRender();
      }
    };
    document.addEventListener('visibilitychange', onVisibilityChange);

    requestRender();

    return () => {
      resizeObserver.disconnect();
      document.removeEventListener('visibilitychange', onVisibilityChange);
      if (frameRequestRef.current != null) {
        cancelAnimationFrame(frameRequestRef.current);
        frameRequestRef.current = null;
      }
    };
  }, [requestRender, syncCanvasDisplaySize]);

  return (
    <div className="relative h-full w-full">
      <canvas
        ref={canvasRef}
        onClick={onCanvasClick}
        onMouseDown={onCanvasMouseDown}
        onMouseMove={onCanvasMouseMove}
        onMouseUp={onCanvasMouseUp}
        onMouseLeave={onCanvasMouseUp}
        id="world-canvas"
        width={900}
        height={900}
        className={`block h-full w-full max-h-full max-w-full shrink-0 select-none bg-surface ${cursorClass}`}
      />

      <div className="absolute bottom-3 left-3 z-10 rounded bg-panel/85 px-2.5 py-1.5 shadow-panel backdrop-blur-sm">
        <div className="text-[10px] font-semibold uppercase tracking-wide text-ink/50">Visibility</div>
        <label className="mt-1 flex cursor-pointer items-center gap-2 text-xs text-ink">
          <input
            type="checkbox"
            checked={showOrganisms}
            onChange={(evt) => setShowOrganisms(evt.target.checked)}
            className="h-3.5 w-3.5 rounded border-accent/40 text-accent focus:ring-accent/30"
          />
          <span>Organisms</span>
        </label>
        <label className="mt-1 flex cursor-pointer items-center gap-2 text-xs text-ink">
          <input
            type="checkbox"
            checked={showPlants}
            onChange={(evt) => setShowPlants(evt.target.checked)}
            className="h-3.5 w-3.5 rounded border-accent/40 text-accent focus:ring-accent/30"
          />
          <span>Plants</span>
        </label>
      </div>

      {showFastOverlay ? (
        <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center bg-slate-950/42 backdrop-blur-[1px]">
          <div className="flex flex-col items-center gap-3 rounded-2xl border border-white/15 bg-slate-950/45 px-6 py-5 text-white shadow-2xl">
            <svg
              aria-hidden="true"
              viewBox="0 0 24 24"
              className="h-12 w-12 fill-current opacity-95"
            >
              <path d="M4 5.5v13l8-6.5-8-6.5Z" />
              <path d="M12 5.5v13l8-6.5-8-6.5Z" />
            </svg>
            <div className="text-xs font-semibold uppercase tracking-[0.28em] text-white/80">
              Fast Run
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
