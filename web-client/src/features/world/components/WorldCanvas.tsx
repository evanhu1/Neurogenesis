import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type MouseEvent,
  type RefObject,
} from 'react';
import { buildHexLayout, hexCenter, pickOrganismAtCanvasPoint, renderWorld } from '../worldCanvas';
import { unwrapId } from '../../../protocol';
import type { WorldOrganismState, WorldSnapshot } from '../../../types';
import { useWorldViewport } from '../hooks/useWorldViewport';

type WorldCanvasProps = {
  snapshot: WorldSnapshot | null;
  focusedOrganismId: number | null;
  onOrganismSelect: (organism: WorldOrganismState) => void;
  panToHexRef?: RefObject<((q: number, r: number) => void) | null>;
};

export function WorldCanvas({
  snapshot,
  focusedOrganismId,
  onOrganismSelect,
  panToHexRef,
}: WorldCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameRequestRef = useRef<number | null>(null);
  const needsRenderRef = useRef(true);
  const requestRenderCallbackRef = useRef<() => void>(() => {});
  const snapshotRef = useRef<WorldSnapshot | null>(snapshot);
  const focusedOrganismIdRef = useRef<number | null>(focusedOrganismId);
  const onOrganismSelectRef = useRef(onOrganismSelect);
  const hasAutoFitRef = useRef(false);
  const [showOrganisms, setShowOrganisms] = useState(true);
  const [showPlants, setShowPlants] = useState(true);
  const showOrganismsRef = useRef(showOrganisms);
  const showPlantsRef = useRef(showPlants);
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
  } = useWorldViewport({ onViewportChange: () => requestRenderCallbackRef.current() });
  const drawCurrentFrame = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !needsRenderRef.current) return;

    needsRenderRef.current = false;

    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const displayHeight = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }

    const context = canvas.getContext('2d');
    if (!context) return;

    renderWorld(context, canvas, snapshotRef.current, focusedOrganismIdRef.current, viewportRef.current, {
      organisms: showOrganismsRef.current,
      plants: showPlantsRef.current,
    });
  }, [viewportRef]);
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

    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const displayHeight = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }

    fitWorldToCanvas(canvas, snapshot.config.world_width);
    hasAutoFitRef.current = true;
  }, [fitWorldToCanvas, snapshot]);

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
    [consumeSuppressedClick, isSpacePressed, viewportRef],
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

    const resizeObserver = new ResizeObserver(() => {
      requestRender();
    });
    resizeObserver.observe(canvas);

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
  }, [requestRender]);

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
        className={`block h-full w-full max-h-full max-w-full shrink-0 select-none rounded-xl border border-accent/20 bg-white ${cursorClass}`}
      />

      <div className="absolute bottom-3 left-3 z-10 rounded-lg border border-accent/25 bg-panel/90 px-3 py-2 shadow-panel backdrop-blur-sm">
        <div className="text-[10px] font-semibold uppercase tracking-wide text-ink/70">Visibility</div>
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
    </div>
  );
}
