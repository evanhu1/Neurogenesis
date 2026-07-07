import { useEffect, useRef } from 'react';
import { BrainRenderer } from '../../rendering/BrainRenderer';
import type { BrainState } from '../../types';

type BrainCanvasProps = {
  focusedBrain: BrainState | null;
  activeActionNeuronId: number | null;
  focusOrganismId: number | null;
  actionBiases: number[];
};

export function BrainCanvas({
  focusedBrain,
  activeActionNeuronId,
  focusOrganismId,
  actionBiases,
}: BrainCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<BrainRenderer | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const renderer = new BrainRenderer(canvas);
    rendererRef.current = renderer;
    return () => {
      renderer.dispose();
      rendererRef.current = null;
    };
  }, []);

  useEffect(() => {
    rendererRef.current?.setBrain(focusedBrain, activeActionNeuronId, actionBiases, focusOrganismId);
  }, [focusedBrain, activeActionNeuronId, actionBiases, focusOrganismId]);

  return (
    <canvas
      ref={canvasRef}
      id="brain-canvas"
      className="h-full w-full cursor-grab rounded-lg border border-line bg-void active:cursor-grabbing"
    />
  );
}
