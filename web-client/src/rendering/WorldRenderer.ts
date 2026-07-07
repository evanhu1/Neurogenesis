//! `WorldRenderer` — the standalone, framework-agnostic world visualization
//! engine. It owns the in-memory `WorldSnapshot`, applies per-tick deltas, drives
//! the `requestAnimationFrame` loop, and manages viewport (zoom/pan) state plus
//! all pointer/keyboard input. Construct it with a `<canvas>` element; feed it
//! full snapshots (`setWorld`) or live deltas (`applyDelta`); it draws.
//!
//! Statefulness that used to live across React refs/hooks/effects is quarantined
//! here on purpose: this module exists solely to produce visualizations and
//! animations, so the app frontend can stay a thin command/read client.

import {
  buildHexLayout,
  computeBaseHexSize,
  computeWorldFitZoom,
  createWorldRenderCache,
  hexCenter,
  pickOrganismAtCanvasPoint,
  renderWorld,
  type RenderVisibility,
  type WorldRenderCache,
  type WorldViewport,
} from './worldCanvas';
import { applyTickDelta, findOrganism } from './delta';
import type { TickDelta, WorldOrganismState, WorldSnapshot } from '../types';

const BASE_MIN_WORLD_ZOOM = 0.65;
const ABSOLUTE_MIN_WORLD_ZOOM = 0.2;
const BASE_MAX_WORLD_ZOOM = 4;
const ABSOLUTE_MAX_WORLD_ZOOM = 48;
const START_FIT_MIN_WORLD_ZOOM = BASE_MIN_WORLD_ZOOM;
const TARGET_HEX_RADIUS_PX = 48;
const WORLD_ZOOM_STEP = 1.12;
const FIT_WORLD_MARGIN = 0.95;

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

export type WorldRendererOptions = {
  onPick?: (organism: WorldOrganismState | null) => void;
  onViewportChange?: () => void;
};

export class WorldRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly onPick?: (organism: WorldOrganismState | null) => void;
  private readonly onViewportChange?: () => void;

  private readonly cache: WorldRenderCache = createWorldRenderCache();
  private snapshot: WorldSnapshot | null = null;
  private focusedOrganismId: number | null = null;
  private visibility: RenderVisibility = { organisms: true, plants: true };
  private viewport: WorldViewport = { zoom: 1, panX: 0, panY: 0 };

  private frameRequest: number | null = null;
  private needsRender = true;
  private readonly displaySize = { width: 0, height: 0, dpr: 0 };
  private hasAutoFit = false;
  private lastWorldWidth: number | undefined = undefined;
  private lastTurn = -1;

  private isSpacePressed = false;
  private isPanning = false;
  private panState: PanState | null = null;
  private suppressNextClick = false;

  private readonly resizeObserver: ResizeObserver;
  private disposed = false;

  // Bound handlers retained so they can be detached in `dispose`.
  private readonly onWheel = (evt: WheelEvent) => {
    evt.preventDefault();
    this.zoomAtPointer(evt.clientX, evt.clientY, evt.deltaY);
  };
  private readonly onMouseDown = (evt: MouseEvent) => {
    if (!this.isSpacePressed) return;
    evt.preventDefault();
    this.panState = {
      startClientX: evt.clientX,
      startClientY: evt.clientY,
      startPanX: this.viewport.panX,
      startPanY: this.viewport.panY,
      dragged: false,
    };
    this.isPanning = true;
    this.updateCursor();
  };
  private readonly onMouseMove = (evt: MouseEvent) => {
    if (!this.isPanning) return;
    const panState = this.panState;
    if (!panState) return;
    const dx = evt.clientX - panState.startClientX;
    const dy = evt.clientY - panState.startClientY;
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = rect.width > 0 ? this.canvas.width / rect.width : 1;
    const scaleY = rect.height > 0 ? this.canvas.height / rect.height : 1;
    const dxCanvasPx = dx * scaleX;
    const dyCanvasPx = dy * scaleY;
    if (!panState.dragged && (Math.abs(dx) > 1 || Math.abs(dy) > 1)) {
      panState.dragged = true;
    }
    this.updateViewport((prev) => ({
      ...prev,
      panX: panState.startPanX + dxCanvasPx,
      panY: panState.startPanY + dyCanvasPx,
    }));
  };
  private readonly onMouseUp = () => {
    if (this.panState?.dragged) {
      this.suppressNextClick = true;
    }
    this.panState = null;
    this.isPanning = false;
    this.updateCursor();
  };
  private readonly onClick = (evt: MouseEvent) => {
    if (this.suppressNextClick) {
      this.suppressNextClick = false;
      return;
    }
    if (this.isSpacePressed) return;
    if (!this.visibility.organisms) return;
    const snapshot = this.snapshot;
    if (!snapshot) return;
    const rect = this.canvas.getBoundingClientRect();
    const xPx = ((evt.clientX - rect.left) / rect.width) * this.canvas.width;
    const yPx = ((evt.clientY - rect.top) / rect.height) * this.canvas.height;
    const picked = pickOrganismAtCanvasPoint(
      snapshot,
      this.canvas.width,
      this.canvas.height,
      xPx,
      yPx,
      this.viewport,
    );
    if (!picked) return;
    this.onPick?.(picked);
  };
  private readonly onKeyDown = (evt: KeyboardEvent) => {
    if (evt.code !== 'Space') return;
    if (isInteractiveTarget(evt.target)) return;
    evt.preventDefault();
    this.isSpacePressed = true;
    this.updateCursor();
  };
  private readonly onKeyUp = (evt: KeyboardEvent) => {
    if (evt.code !== 'Space') return;
    this.isSpacePressed = false;
    this.panState = null;
    this.isPanning = false;
    this.updateCursor();
  };
  private readonly onBlur = () => {
    this.isSpacePressed = false;
    this.panState = null;
    this.isPanning = false;
    this.updateCursor();
  };
  private readonly onVisibilityChange = () => {
    if (document.visibilityState === 'visible') {
      this.requestRender();
    }
  };

  constructor(canvas: HTMLCanvasElement, options: WorldRendererOptions = {}) {
    this.canvas = canvas;
    this.onPick = options.onPick;
    this.onViewportChange = options.onViewportChange;

    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    canvas.addEventListener('mousedown', this.onMouseDown);
    canvas.addEventListener('mousemove', this.onMouseMove);
    canvas.addEventListener('mouseup', this.onMouseUp);
    canvas.addEventListener('mouseleave', this.onMouseUp);
    canvas.addEventListener('click', this.onClick);
    window.addEventListener('keydown', this.onKeyDown, { passive: false });
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('blur', this.onBlur);
    document.addEventListener('visibilitychange', this.onVisibilityChange);

    this.resizeObserver = new ResizeObserver(() => {
      if (!this.syncCanvasDisplaySize()) return;
      this.requestRender();
    });
    this.resizeObserver.observe(canvas.parentElement ?? canvas);

    this.updateCursor();
    this.requestRender();
  }

  // --- Public API -----------------------------------------------------------

  /// Replace the animated world with a full snapshot (initial load, static
  /// feed, or fast-run refresh). Auto-fits the first world of a given width and
  /// keeps a focused organism centered.
  setWorld(snapshot: WorldSnapshot): void {
    const worldWidth = snapshot.config.world_width;
    if (worldWidth !== this.lastWorldWidth) {
      this.hasAutoFit = false;
      this.lastWorldWidth = worldWidth;
    }
    this.snapshot = snapshot;
    this.lastTurn = snapshot.turn;
    if (!this.hasAutoFit) {
      this.syncCanvasDisplaySize();
      this.fitWorldToCanvas();
      this.hasAutoFit = true;
    }
    this.followFocusedOrganism();
    this.requestRender();
  }

  /// Fold a live per-tick delta into the in-memory world. Stale/out-of-order
  /// frames (turn <= last applied) are dropped.
  applyDelta(delta: TickDelta): void {
    const snapshot = this.snapshot;
    if (!snapshot) return;
    if (delta.turn <= this.lastTurn) return;
    this.snapshot = applyTickDelta(snapshot, delta);
    this.lastTurn = delta.turn;
    this.followFocusedOrganism();
    this.requestRender();
  }

  setFocusedOrganismId(id: number | null): void {
    this.focusedOrganismId = id;
    this.followFocusedOrganism();
    this.requestRender();
  }

  setVisibility(visibility: RenderVisibility): void {
    this.visibility = { ...visibility };
    this.requestRender();
  }

  getSnapshot(): WorldSnapshot | null {
    return this.snapshot;
  }

  /// Fit the current world to the canvas, clamped so startup auto-fit never
  /// zooms too far out.
  fitWorldToCanvas(): void {
    const worldWidth = this.snapshot?.config.world_width;
    if (!worldWidth || worldWidth <= 0) return;
    const fitZoom = computeWorldFitZoom(this.canvas.width, this.canvas.height, worldWidth);
    if (!Number.isFinite(fitZoom) || fitZoom <= 0) return;
    const maxWorldZoom = computeMaxWorldZoom(this.canvas, worldWidth);
    const nextZoom = Math.max(START_FIT_MIN_WORLD_ZOOM, Math.min(maxWorldZoom, fitZoom));
    this.updateViewport((prev) => {
      if (prev.zoom === nextZoom && prev.panX === 0 && prev.panY === 0) return prev;
      return { zoom: nextZoom, panX: 0, panY: 0 };
    });
  }

  /// Center the viewport on an organism's current cell.
  panToOrganism(id: number): void {
    const snapshot = this.snapshot;
    if (!snapshot) return;
    const organism = findOrganism(snapshot, id);
    if (!organism) return;
    this.panToHex(organism.q, organism.r);
  }

  /// Center the viewport on a hex coordinate.
  panToHex(q: number, r: number): void {
    const worldWidth = this.snapshot?.config.world_width;
    if (!worldWidth) return;
    const layout = buildHexLayout(this.canvas.width, this.canvas.height, worldWidth);
    const center = hexCenter(layout, q, r);
    this.panToWorldPoint(center.x, center.y);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.canvas.removeEventListener('wheel', this.onWheel);
    this.canvas.removeEventListener('mousedown', this.onMouseDown);
    this.canvas.removeEventListener('mousemove', this.onMouseMove);
    this.canvas.removeEventListener('mouseup', this.onMouseUp);
    this.canvas.removeEventListener('mouseleave', this.onMouseUp);
    this.canvas.removeEventListener('click', this.onClick);
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    window.removeEventListener('blur', this.onBlur);
    document.removeEventListener('visibilitychange', this.onVisibilityChange);
    this.resizeObserver.disconnect();
    if (this.frameRequest != null) {
      cancelAnimationFrame(this.frameRequest);
      this.frameRequest = null;
    }
  }

  // --- Internals ------------------------------------------------------------

  private followFocusedOrganism(): void {
    if (this.focusedOrganismId == null) return;
    const snapshot = this.snapshot;
    if (!snapshot) return;
    const organism = findOrganism(snapshot, this.focusedOrganismId);
    if (!organism) return;
    this.panToHex(organism.q, organism.r);
  }

  private panToWorldPoint(worldX: number, worldY: number): void {
    this.updateViewport((prev) => {
      const panX = -(worldX - this.canvas.width / 2) * prev.zoom;
      const panY = -(worldY - this.canvas.height / 2) * prev.zoom;
      if (panX === prev.panX && panY === prev.panY) return prev;
      return { ...prev, panX, panY };
    });
  }

  private zoomAtPointer(clientX: number, clientY: number, deltaY: number): void {
    const canvas = this.canvas;
    const worldWidth = this.snapshot?.config.world_width ?? null;
    const rect = canvas.getBoundingClientRect();
    const xPx = ((clientX - rect.left) / rect.width) * canvas.width;
    const yPx = ((clientY - rect.top) / rect.height) * canvas.height;

    this.updateViewport((prev) => {
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
  }

  private updateViewport(updater: (prev: WorldViewport) => WorldViewport): void {
    const prev = this.viewport;
    const next = updater(prev);
    if (viewportEqual(prev, next)) return;
    this.viewport = next;
    this.onViewportChange?.();
    this.requestRender();
  }

  private updateCursor(): void {
    this.canvas.style.cursor = this.isSpacePressed || this.isPanning ? 'pointer' : 'default';
  }

  private syncCanvasDisplaySize(): boolean {
    const canvas = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const displayHeight = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    const previous = this.displaySize;
    const changed =
      previous.width !== displayWidth || previous.height !== displayHeight || previous.dpr !== dpr;
    if (!changed) return false;
    this.displaySize.width = displayWidth;
    this.displaySize.height = displayHeight;
    this.displaySize.dpr = dpr;
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }
    return true;
  }

  private drawCurrentFrame(): void {
    const canvas = this.canvas;
    if (!this.needsRender) return;
    this.needsRender = false;
    this.syncCanvasDisplaySize();
    const context = canvas.getContext('2d');
    if (!context) return;
    renderWorld(
      context,
      canvas,
      this.snapshot,
      this.focusedOrganismId,
      this.viewport,
      this.visibility,
      this.cache,
    );
  }

  private requestRender(): void {
    this.needsRender = true;
    if (this.disposed) return;
    if (this.frameRequest != null || document.visibilityState === 'hidden') return;
    this.frameRequest = requestAnimationFrame(() => {
      this.frameRequest = null;
      this.drawCurrentFrame();
      if (this.needsRender) {
        this.requestRender();
      }
    });
  }
}
