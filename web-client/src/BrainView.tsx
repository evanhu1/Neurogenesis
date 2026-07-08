import { useEffect, useRef } from 'react';
import type { BrainNet } from './api';

// Draws the developed BrainNet: input neurons in a left column, hidden in the
// middle, outputs on the right; edges blue (excitatory) / red (inhibitory),
// width ∝ |weight|. Purely a phenotype view (the CPPN that generated it is
// summarized separately in the inspector).
export function BrainView({ brain }: { brain: BrainNet }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const width = 300;
    const height = 220;
    canvas.width = width;
    canvas.height = height;
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    const inputs = brain.input_count;
    const hidden = brain.hidden_count;
    const outputs = brain.output_count;
    const col = (kind: number) => 30 + kind * ((width - 60) / 2);
    const pos = (i: number): [number, number] => {
      if (i < inputs) return [col(0), lay(i, inputs, height)];
      if (i < inputs + hidden) return [col(1), lay(i - inputs, hidden, height)];
      return [col(2), lay(i - inputs - hidden, outputs, height)];
    };

    for (const e of brain.edges) {
      const [x1, y1] = pos(e.from);
      const [x2, y2] = pos(e.to);
      ctx.strokeStyle = e.weight >= 0 ? 'rgba(88,166,255,0.6)' : 'rgba(248,81,73,0.6)';
      ctx.lineWidth = Math.min(3, 0.4 + Math.abs(e.weight));
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    brain.neurons.forEach((n, i) => {
      const [x, y] = pos(i);
      ctx.fillStyle =
        n.kind === 'Input' ? '#3fb950' : n.kind === 'Output' ? '#e3b341' : '#8b949e';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [brain]);
  return <canvas ref={ref} style={{ width: 300, height: 220, borderRadius: 6 }} />;
}

function lay(i: number, count: number, height: number): number {
  if (count <= 1) return height / 2;
  return 20 + (i / (count - 1)) * (height - 40);
}
