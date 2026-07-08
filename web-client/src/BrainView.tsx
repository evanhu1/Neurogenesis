import { useEffect, useRef } from 'react';
import type { BrainNet } from './api';

// Draws the developed BrainNet: input neurons in a left column, hidden in the
// middle, outputs on the right; edges blue (excitatory) / red (inhibitory),
// width ∝ |weight|. Phenotype view (the CPPN that generated it is summarized in
// the inspector).
export function BrainView({ brain }: { brain: BrainNet }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 320;
    const height = 200;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const inputs = brain.input_count;
    const hidden = brain.hidden_count;
    const outputs = brain.output_count;
    const colX = (k: number) => 26 + k * ((width - 52) / 2);
    const lay = (i: number, count: number) =>
      count <= 1 ? height / 2 : 18 + (i / (count - 1)) * (height - 36);
    const pos = (i: number): [number, number] => {
      if (i < inputs) return [colX(0), lay(i, inputs)];
      if (i < inputs + hidden) return [colX(1), lay(i - inputs, hidden)];
      return [colX(2), lay(i - inputs - hidden, outputs)];
    };

    for (const e of brain.edges) {
      const [x1, y1] = pos(e.from);
      const [x2, y2] = pos(e.to);
      ctx.strokeStyle = e.weight >= 0 ? 'rgba(21,128,61,0.5)' : 'rgba(190,60,55,0.5)';
      ctx.lineWidth = Math.min(2.5, 0.4 + Math.abs(e.weight));
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    brain.neurons.forEach((n, i) => {
      const [x, y] = pos(i);
      ctx.fillStyle = n.kind === 'Input' ? '#3fae4f' : n.kind === 'Output' ? '#eab308' : '#7b8794';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [brain]);
  return <canvas ref={ref} style={{ width: '100%', height: 200, display: 'block' }} />;
}
