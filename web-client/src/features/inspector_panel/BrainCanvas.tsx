import { useEffect, useRef, useCallback } from 'react';
import {
  renderBrain,
  computeBrainLayout,
  zoomToFitBrain,
  type BrainTransform,
} from './brainRenderer';
import type { BrainState } from '../../types';
import React from 'react';

type BrainCanvasProps = {
  focusedBrain: BrainState | null;
  activeActionNeuronId: number | null;
  focusOrganismId: number | null;
};

export function BrainCanvas({
  focusedBrain,
  activeActionNeuronId,
  focusOrganismId,
}: BrainCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const transformRef = useRef<BrainTransform>({ x: 0, y: 0, scale: 1 });
  const dragRef = useRef<{ sx: number; sy: number; tx: number; ty: number } | null>(null);
  const fitKeyRef = useRef<string>('');

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    renderBrain(ctx, canvas, focusedBrain, activeActionNeuronId, transformRef.current);
  }, [focusedBrain, activeActionNeuronId]);

  // Only auto-fit on focus/layout changes, not every tick update.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      draw();
      return;
    }
    if (!focusedBrain) {
      transformRef.current = { x: 0, y: 0, scale: 1 };
      fitKeyRef.current = '';
      draw();
      return;
    }
    const fitKey = [
      focusOrganismId ?? 'none',
      focusedBrain.sensory.length,
      focusedBrain.inter.length,
      focusedBrain.action.length,
    ].join(':');
    if (fitKeyRef.current !== fitKey) {
      const layout = computeBrainLayout(focusedBrain, activeActionNeuronId);
      transformRef.current = zoomToFitBrain(layout, canvas.width, canvas.height);
      fitKeyRef.current = fitKey;
    }
    draw();
  }, [focusOrganismId, focusedBrain, activeActionNeuronId, draw]);

  // Wheel zoom (centered on cursor)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const t = transformRef.current;
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      const newScale = Math.max(0.08, Math.min(12, t.scale * factor));
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
      const my = (e.clientY - rect.top) * (canvas.height / rect.height);
      transformRef.current = {
        x: mx - (mx - t.x) * (newScale / t.scale),
        y: my - (my - t.y) * (newScale / t.scale),
        scale: newScale,
      };
      draw();
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [draw]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      dragRef.current = {
        sx: e.clientX,
        sy: e.clientY,
        tx: transformRef.current.x,
        ty: transformRef.current.y,
      };
    },
    [],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const d = dragRef.current;
      if (!d) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const dx = (e.clientX - d.sx) * (canvas.width / rect.width);
      const dy = (e.clientY - d.sy) * (canvas.height / rect.height);
      transformRef.current = { ...transformRef.current, x: d.tx + dx, y: d.ty + dy };
      draw();
    },
    [draw],
  );

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  return (
    <canvas
      ref={canvasRef}
      id="brain-canvas"
      width={420}
      height={460}
      className="mt-2 w-full  h-full rounded-xl border border-accent/20 bg-white cursor-grab active:cursor-grabbing"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    />
  );
}
