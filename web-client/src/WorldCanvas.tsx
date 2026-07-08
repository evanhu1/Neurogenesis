import { useEffect, useRef } from 'react';
import type { RenderSnapshot } from './api';

// Renders the hex world snapshot to a canvas. Cells are drawn on an axial grid
// with a simple offset so hex adjacency reads correctly; organisms are colored
// discs, food green/orange, walls grey, spikes red. Clicking selects the
// nearest organism.
export function WorldCanvas({
  snapshot,
  onSelect,
  selectedId,
}: {
  snapshot: RenderSnapshot | null;
  onSelect: (id: number) => void;
  selectedId: number | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !snapshot) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const size = 560;
    canvas.width = size;
    canvas.height = size;
    const w = snapshot.width;
    const cell = size / (w + 0.5);
    const px = (q: number, r: number): [number, number] => [
      (q + (r % 2) * 0.5) * cell + cell / 2,
      r * cell * 0.88 + cell / 2,
    ];

    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, size, size);

    for (const [q, r] of snapshot.walls) {
      const [x, y] = px(q, r);
      ctx.fillStyle = '#30363d';
      ctx.fillRect(x - cell / 2, y - cell / 2, cell, cell);
    }
    for (const [q, r] of snapshot.spikes) {
      const [x, y] = px(q, r);
      ctx.fillStyle = 'rgba(220,60,60,0.5)';
      ctx.fillRect(x - cell / 2, y - cell / 2, cell, cell);
    }
    for (const [q, r] of snapshot.food) {
      const [x, y] = px(q, r);
      ctx.fillStyle = '#3fb950';
      ctx.beginPath();
      ctx.arc(x, y, cell * 0.28, 0, Math.PI * 2);
      ctx.fill();
    }
    for (const o of snapshot.organisms) {
      const [x, y] = px(o.q, o.r);
      const [rr, gg, bb] = o.color;
      ctx.fillStyle = `rgb(${(rr * 255) | 0},${(gg * 255) | 0},${(bb * 255) | 0})`;
      ctx.beginPath();
      ctx.arc(x, y, cell * 0.42, 0, Math.PI * 2);
      ctx.fill();
      if (o.id === selectedId) {
        ctx.strokeStyle = '#e3b341';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }, [snapshot, selectedId]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!snapshot) return;
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const my = ((e.clientY - rect.top) / rect.height) * canvas.height;
    const w = snapshot.width;
    const cell = canvas.width / (w + 0.5);
    let best: number | null = null;
    let bestDist = Infinity;
    for (const o of snapshot.organisms) {
      const x = (o.q + (o.r % 2) * 0.5) * cell + cell / 2;
      const y = o.r * cell * 0.88 + cell / 2;
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < bestDist) {
        bestDist = d;
        best = o.id;
      }
    }
    if (best !== null && bestDist < (cell * 1.5) ** 2) onSelect(best);
  };

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      style={{ width: 560, height: 560, borderRadius: 8, cursor: 'crosshair' }}
    />
  );
}
