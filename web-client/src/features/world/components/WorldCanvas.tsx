import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
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
  deadFlashCells: Array<{ q: number; r: number }> | null;
  bornFlashCells: Array<{ q: number; r: number }> | null;
  onOrganismSelect: (organism: WorldOrganismState) => void;
  panToHexRef?: RefObject<((q: number, r: number) => void) | null>;
};

export function WorldCanvas({
  snapshot,
  focusedOrganismId,
  deadFlashCells,
  bornFlashCells,
  onOrganismSelect,
  panToHexRef,
}: WorldCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const snapshotRef = useRef<WorldSnapshot | null>(snapshot);
  const focusedOrganismIdRef = useRef<number | null>(focusedOrganismId);
  const deadFlashCellsRef = useRef<Array<{ q: number; r: number }> | null>(deadFlashCells);
  const bornFlashCellsRef = useRef<Array<{ q: number; r: number }> | null>(bornFlashCells);
  const onOrganismSelectRef = useRef(onOrganismSelect);
  const hasAutoFitRef = useRef(false);
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
  } = useWorldViewport();

  // Keep refs synchronized during render so the RAF draw loop sees latest props immediately.
  snapshotRef.current = snapshot;
  focusedOrganismIdRef.current = focusedOrganismId;
  deadFlashCellsRef.current = deadFlashCells;
  bornFlashCellsRef.current = bornFlashCells;
  onOrganismSelectRef.current = onOrganismSelect;

  useEffect(() => {
    hasAutoFitRef.current = false;
  }, [snapshot?.config.world_width]);

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
  useLayoutEffect(() => {
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
    let frameId = 0;
    const draw = () => {
      const canvas = canvasRef.current;
      if (canvas) {
        const dpr = window.devicePixelRatio || 1;
        const displayWidth = Math.max(1, Math.floor(canvas.clientWidth * dpr));
        const displayHeight = Math.max(1, Math.floor(canvas.clientHeight * dpr));
        if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
          canvas.width = displayWidth;
          canvas.height = displayHeight;
        }
        const context = canvas.getContext('2d');
        if (context) {
          renderWorld(
            context,
            canvas,
            snapshotRef.current,
            focusedOrganismIdRef.current,
            viewportRef.current,
            deadFlashCellsRef.current,
            bornFlashCellsRef.current,
          );
        }
      }
      frameId = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(frameId);
  }, [viewportRef]);

  return (
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
  );
}
