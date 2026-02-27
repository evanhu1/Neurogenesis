import { useCallback, useEffect, useRef, useState, type MouseEvent } from 'react';
import { computeBaseHexSize, computeWorldFitZoom, type WorldViewport } from '../worldCanvas';

const BASE_MIN_WORLD_ZOOM = 0.65;
const ABSOLUTE_MIN_WORLD_ZOOM = 0.2;
const BASE_MAX_WORLD_ZOOM = 4;
const ABSOLUTE_MAX_WORLD_ZOOM = 48;
const START_FIT_MIN_WORLD_ZOOM = BASE_MIN_WORLD_ZOOM;
const TARGET_HEX_RADIUS_PX = 48;
const WORLD_ZOOM_STEP = 1.12;
const FIT_WORLD_MARGIN = 0.95;
const DEFAULT_WORLD_VIEWPORT: WorldViewport = { zoom: 1, panX: 0, panY: 0 };

function viewportEqual(a: WorldViewport, b: WorldViewport): boolean {
  return a.zoom === b.zoom && a.panX === b.panX && a.panY === b.panY;
}

function computeMaxWorldZoom(canvas: HTMLCanvasElement, worldWidth: number | null | undefined): number {
  if (!worldWidth || worldWidth <= 0) return BASE_MAX_WORLD_ZOOM;
  const baseHexSize = computeBaseHexSize(canvas.width, canvas.height, worldWidth);
  if (!Number.isFinite(baseHexSize) || baseHexSize <= 0) return BASE_MAX_WORLD_ZOOM;
  const zoomForTargetHexSize = TARGET_HEX_RADIUS_PX / baseHexSize;
  return Math.min(ABSOLUTE_MAX_WORLD_ZOOM, Math.max(BASE_MAX_WORLD_ZOOM, zoomForTargetHexSize));
}

function computeMinWorldZoom(canvas: HTMLCanvasElement, worldWidth: number | null | undefined): number {
  if (!worldWidth || worldWidth <= 0) return BASE_MIN_WORLD_ZOOM;
  const fitZoom = computeWorldFitZoom(canvas.width, canvas.height, worldWidth);
  if (!Number.isFinite(fitZoom) || fitZoom <= 0) return BASE_MIN_WORLD_ZOOM;
  const minFromFit = fitZoom * FIT_WORLD_MARGIN;
  return Math.max(ABSOLUTE_MIN_WORLD_ZOOM, Math.min(BASE_MIN_WORLD_ZOOM, minFromFit));
}

function isInteractiveTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tagName = target.tagName;
  return (
    target.isContentEditable ||
    tagName === 'INPUT' ||
    tagName === 'TEXTAREA' ||
    tagName === 'SELECT' ||
    tagName === 'BUTTON'
  );
}

type PanState = {
  startClientX: number;
  startClientY: number;
  startPanX: number;
  startPanY: number;
  dragged: boolean;
};

type UseWorldViewportOptions = {
  onViewportChange?: () => void;
};

export function useWorldViewport({ onViewportChange }: UseWorldViewportOptions = {}) {
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const [isPanningWorld, setIsPanningWorld] = useState(false);
  const viewportRef = useRef<WorldViewport>(DEFAULT_WORLD_VIEWPORT);
  const panRef = useRef<PanState | null>(null);
  const suppressNextClickRef = useRef(false);

  const updateViewport = useCallback((updater: (prev: WorldViewport) => WorldViewport) => {
    const prev = viewportRef.current;
    const next = updater(prev);
    if (viewportEqual(prev, next)) return;
    viewportRef.current = next;
    onViewportChange?.();
  }, [onViewportChange]);

  const zoomAtPointer = useCallback(
    (
      canvas: HTMLCanvasElement,
      clientX: number,
      clientY: number,
      deltaY: number,
      worldWidth: number | null | undefined,
    ) => {
      const rect = canvas.getBoundingClientRect();
      const xPx = ((clientX - rect.left) / rect.width) * canvas.width;
      const yPx = ((clientY - rect.top) / rect.height) * canvas.height;

      updateViewport((prev) => {
        const zoomFactor = deltaY < 0 ? WORLD_ZOOM_STEP : 1 / WORLD_ZOOM_STEP;
        const minWorldZoom = computeMinWorldZoom(canvas, worldWidth);
        const maxWorldZoom = computeMaxWorldZoom(canvas, worldWidth);
        const nextZoom = Math.max(minWorldZoom, Math.min(maxWorldZoom, prev.zoom * zoomFactor));
        if (nextZoom === prev.zoom) return prev;

        const worldX = (xPx - canvas.width / 2 - prev.panX) / prev.zoom + canvas.width / 2;
        const worldY = (yPx - canvas.height / 2 - prev.panY) / prev.zoom + canvas.height / 2;
        const panX = xPx - canvas.width / 2 - (worldX - canvas.width / 2) * nextZoom;
        const panY = yPx - canvas.height / 2 - (worldY - canvas.height / 2) * nextZoom;

        return { zoom: nextZoom, panX, panY };
      });
    },
    [updateViewport],
  );

  const onCanvasMouseDown = useCallback(
    (evt: MouseEvent<HTMLCanvasElement>) => {
      if (!isSpacePressed) return;
      evt.preventDefault();
      panRef.current = {
        startClientX: evt.clientX,
        startClientY: evt.clientY,
        startPanX: viewportRef.current.panX,
        startPanY: viewportRef.current.panY,
        dragged: false,
      };
      setIsPanningWorld(true);
    },
    [isSpacePressed],
  );

  const onCanvasMouseMove = useCallback(
    (evt: MouseEvent<HTMLCanvasElement>) => {
      if (!isPanningWorld) return;
      const panState = panRef.current;
      if (!panState) return;
      const dx = evt.clientX - panState.startClientX;
      const dy = evt.clientY - panState.startClientY;
      const rect = evt.currentTarget.getBoundingClientRect();
      const scaleX = rect.width > 0 ? evt.currentTarget.width / rect.width : 1;
      const scaleY = rect.height > 0 ? evt.currentTarget.height / rect.height : 1;
      const dxCanvasPx = dx * scaleX;
      const dyCanvasPx = dy * scaleY;
      if (!panState.dragged && (Math.abs(dx) > 1 || Math.abs(dy) > 1)) {
        panState.dragged = true;
      }
      updateViewport((prev) => ({
        ...prev,
        panX: panState.startPanX + dxCanvasPx,
        panY: panState.startPanY + dyCanvasPx,
      }));
    },
    [isPanningWorld, updateViewport],
  );

  const onCanvasMouseUp = useCallback(() => {
    if (panRef.current?.dragged) {
      suppressNextClickRef.current = true;
    }
    panRef.current = null;
    setIsPanningWorld(false);
  }, []);

  useEffect(() => {
    const onKeyDown = (evt: KeyboardEvent) => {
      if (evt.code !== 'Space') return;
      if (isInteractiveTarget(evt.target)) return;
      evt.preventDefault();
      setIsSpacePressed(true);
    };

    const onKeyUp = (evt: KeyboardEvent) => {
      if (evt.code !== 'Space') return;
      setIsSpacePressed(false);
      panRef.current = null;
      setIsPanningWorld(false);
    };

    const onBlur = () => {
      setIsSpacePressed(false);
      panRef.current = null;
      setIsPanningWorld(false);
    };

    window.addEventListener('keydown', onKeyDown, { passive: false });
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    };
  }, []);

  const consumeSuppressedClick = useCallback(() => {
    if (!suppressNextClickRef.current) return false;
    suppressNextClickRef.current = false;
    return true;
  }, []);

  const panToWorldPoint = useCallback(
    (worldX: number, worldY: number, canvasWidth: number, canvasHeight: number) => {
      updateViewport((prev) => {
        const panX = -(worldX - canvasWidth / 2) * prev.zoom;
        const panY = -(worldY - canvasHeight / 2) * prev.zoom;
        if (panX === prev.panX && panY === prev.panY) return prev;
        return { ...prev, panX, panY };
      });
    },
    [updateViewport],
  );

  const fitWorldToCanvas = useCallback(
    (canvas: HTMLCanvasElement, worldWidth: number | null | undefined) => {
      if (!worldWidth || worldWidth <= 0) return;
      const fitZoom = computeWorldFitZoom(canvas.width, canvas.height, worldWidth);
      if (!Number.isFinite(fitZoom) || fitZoom <= 0) return;
      const maxWorldZoom = computeMaxWorldZoom(canvas, worldWidth);
      // Prevent startup auto-fit from zooming too far out just to include the entire world.
      const nextZoom = Math.max(START_FIT_MIN_WORLD_ZOOM, Math.min(maxWorldZoom, fitZoom));

      updateViewport((prev) => {
        if (prev.zoom === nextZoom && prev.panX === 0 && prev.panY === 0) return prev;
        return { zoom: nextZoom, panX: 0, panY: 0 };
      });
    },
    [updateViewport],
  );

  const cursorClass = isSpacePressed || isPanningWorld ? 'cursor-pointer' : 'cursor-default';

  return {
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
  };
}
