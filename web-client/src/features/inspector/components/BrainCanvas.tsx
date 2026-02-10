import { useEffect, useRef } from 'react';
import { renderBrain } from '../../../canvas';
import type { BrainState } from '../../../types';

type BrainCanvasProps = {
  focusedBrain: BrainState | null;
};

export function BrainCanvas({ focusedBrain }: BrainCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    renderBrain(context, canvas, focusedBrain);
  }, [focusedBrain]);

  return (
    <canvas
      ref={canvasRef}
      id="brain-canvas"
      width={420}
      height={460}
      className="mt-2 w-full rounded-xl border border-accent/20 bg-white"
    />
  );
}

