//! `BrainRenderer` — standalone neural-network visualization engine for the
//! inspector. Construct it with a `<canvas>`; call `setBrain` when the focused
//! organism or its brain changes. Owns the pan/zoom transform, DPR sizing, and
//! wheel/drag input, refitting only when the organism or its layer counts
//! change (not on every per-tick brain update).

import {
  computeBrainLayout,
  renderBrain,
  zoomToFitBrain,
  type BrainTransform,
} from './brainCanvas';
import type { BrainState } from '../types';

export class BrainRenderer {
  private readonly canvas: HTMLCanvasElement;
  private transform: BrainTransform = { x: 0, y: 0, scale: 1 };
  private dragState: { sx: number; sy: number; tx: number; ty: number } | null = null;
  private fitKey = '';

  private brain: BrainState | null = null;
  private activeActionNeuronId: number | null = null;
  private actionBiases: number[] = [];

  private readonly resizeObserver: ResizeObserver;
  private disposed = false;

  private readonly onWheel = (evt: WheelEvent) => {
    evt.preventDefault();
    const t = this.transform;
    const factor = evt.deltaY < 0 ? 1.12 : 1 / 1.12;
    const newScale = Math.max(0.08, Math.min(12, t.scale * factor));
    const rect = this.canvas.getBoundingClientRect();
    const mx = (evt.clientX - rect.left) * (this.canvas.width / rect.width);
    const my = (evt.clientY - rect.top) * (this.canvas.height / rect.height);
    this.transform = {
      x: mx - (mx - t.x) * (newScale / t.scale),
      y: my - (my - t.y) * (newScale / t.scale),
      scale: newScale,
    };
    this.draw();
  };
  private readonly onMouseDown = (evt: MouseEvent) => {
    this.dragState = {
      sx: evt.clientX,
      sy: evt.clientY,
      tx: this.transform.x,
      ty: this.transform.y,
    };
  };
  private readonly onMouseMove = (evt: MouseEvent) => {
    const drag = this.dragState;
    if (!drag) return;
    const rect = this.canvas.getBoundingClientRect();
    const dx = (evt.clientX - drag.sx) * (this.canvas.width / rect.width);
    const dy = (evt.clientY - drag.sy) * (this.canvas.height / rect.height);
    this.transform = { ...this.transform, x: drag.tx + dx, y: drag.ty + dy };
    this.draw();
  };
  private readonly onMouseUp = () => {
    this.dragState = null;
  };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    canvas.addEventListener('mousedown', this.onMouseDown);
    canvas.addEventListener('mousemove', this.onMouseMove);
    canvas.addEventListener('mouseup', this.onMouseUp);
    canvas.addEventListener('mouseleave', this.onMouseUp);
    this.resizeObserver = new ResizeObserver(() => this.syncSize());
    this.resizeObserver.observe(canvas);
    this.syncSize();
  }

  /// Set the brain to render. `focusOrganismId` participates in the fit key so
  /// switching organisms refits, while per-tick updates to the same organism do
  /// not reset the user's pan/zoom.
  setBrain(
    brain: BrainState | null,
    activeActionNeuronId: number | null,
    actionBiases: number[] = [],
    focusOrganismId: number | null = null,
  ): void {
    this.brain = brain;
    this.activeActionNeuronId = activeActionNeuronId;
    this.actionBiases = actionBiases;

    if (!brain) {
      this.transform = { x: 0, y: 0, scale: 1 };
      this.fitKey = '';
      this.draw();
      return;
    }
    const fitKey = [
      focusOrganismId ?? 'none',
      brain.sensory.length,
      brain.inter.length,
      brain.action.length,
    ].join(':');
    if (this.fitKey !== fitKey) {
      this.fitToCanvas();
      this.fitKey = fitKey;
    }
    this.draw();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.canvas.removeEventListener('wheel', this.onWheel);
    this.canvas.removeEventListener('mousedown', this.onMouseDown);
    this.canvas.removeEventListener('mousemove', this.onMouseMove);
    this.canvas.removeEventListener('mouseup', this.onMouseUp);
    this.canvas.removeEventListener('mouseleave', this.onMouseUp);
    this.resizeObserver.disconnect();
  }

  private fitToCanvas(): void {
    if (!this.brain) return;
    const layout = computeBrainLayout(this.brain, this.activeActionNeuronId, this.actionBiases);
    this.transform = zoomToFitBrain(layout, this.canvas.width, this.canvas.height);
  }

  private draw(): void {
    const ctx = this.canvas.getContext('2d');
    if (!ctx) return;
    renderBrain(
      ctx,
      this.canvas,
      this.brain,
      this.activeActionNeuronId,
      this.transform,
      this.actionBiases,
    );
  }

  private syncSize(): void {
    const canvas = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width === width && canvas.height === height) return;
    canvas.width = width;
    canvas.height = height;
    this.fitToCanvas();
    this.draw();
  }
}
