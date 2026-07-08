import { useEffect, useRef, useState } from 'react';
import type { RenderSnapshot } from './api';

// Real pointy-top hexagon renderer for the toroidal hex world, with pan (drag)
// and zoom (wheel). Axial (q,r) → pixel uses the standard pointy-top layout so
// six-neighbor adjacency reads correctly. Fed by /api/snapshot (full frames,
// polling — no server-side delta stream).

const SQRT3 = Math.sqrt(3);

// Odd-r offset layout (pointy-top): a rectangular hex field (odd rows shifted
// half a hex), rather than the sheared rhombus a raw axial→pixel would give.
function hexToPixel(q: number, r: number, size: number): [number, number] {
  return [size * SQRT3 * (q + 0.5 * (r & 1)), size * 1.5 * r];
}

function hexPath(ctx: CanvasRenderingContext2D, cx: number, cy: number, size: number) {
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const a = (Math.PI / 180) * (60 * i - 30);
    const x = cx + size * Math.cos(a);
    const y = cy + size * Math.sin(a);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

export function HexWorld({
  snapshot,
  selectedId,
  onSelect,
}: {
  snapshot: RenderSnapshot | null;
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const moved = useRef(false);

  // Base hex size that fits the rectangular hex field to the viewport, then
  // center it (before user pan/zoom).
  const layout = (canvas: HTMLCanvasElement, snap: RenderSnapshot) => {
    const w = snap.width;
    // Field extent in "size" units: width ≈ sqrt3·(w+0.5), height ≈ 1.5·w+0.5.
    const base = Math.min(
      canvas.width / (SQRT3 * (w + 0.5)),
      canvas.height / (1.5 * w + 0.5)
    );
    const size = base * zoom;
    const gridW = size * SQRT3 * (w + 0.5);
    const gridH = size * (1.5 * (w - 1) + 2);
    const ox = canvas.width / 2 - gridW / 2 + pan.x;
    const oy = canvas.height / 2 - gridH / 2 + size + pan.y;
    return { size, ox, oy };
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap || !snapshot) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = wrap.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cw = rect.width;
    const ch = rect.height;
    canvas.style.width = `${cw}px`;
    canvas.style.height = `${ch}px`;

    const { size, ox, oy } = layout({ width: cw, height: ch } as HTMLCanvasElement, snapshot);
    const at = (q: number, r: number): [number, number] => {
      const [x, y] = hexToPixel(q, r, size);
      return [x + ox, y + oy];
    };

    // Background.
    ctx.fillStyle = '#aacddc';
    ctx.fillRect(0, 0, cw, ch);

    // Faint base grid (only when cells aren't tiny).
    if (size > 3 && snapshot.width <= 96) {
      ctx.strokeStyle = 'rgba(40,60,70,0.10)';
      ctx.lineWidth = 1;
      for (let r = 0; r < snapshot.width; r++) {
        for (let q = 0; q < snapshot.width; q++) {
          const [x, y] = at(q, r);
          if (x < -size || x > cw + size || y < -size || y > ch + size) continue;
          hexPath(ctx, x, y, size);
          ctx.stroke();
        }
      }
    }

    // Terrain: walls, then spikes.
    ctx.fillStyle = '#5b6470';
    for (const [q, r] of snapshot.walls) {
      const [x, y] = at(q, r);
      hexPath(ctx, x, y, size);
      ctx.fill();
    }
    ctx.fillStyle = 'rgba(190,60,55,0.55)';
    for (const [q, r] of snapshot.spikes) {
      const [x, y] = at(q, r);
      hexPath(ctx, x, y, size * 0.9);
      ctx.fill();
    }

    // Food.
    ctx.fillStyle = '#3fae4f';
    for (const [q, r] of snapshot.food) {
      const [x, y] = at(q, r);
      ctx.beginPath();
      ctx.arc(x, y, size * 0.32, 0, Math.PI * 2);
      ctx.fill();
    }

    // Organisms: body-color-filled hexes.
    for (const o of snapshot.organisms) {
      const [x, y] = at(o.q, o.r);
      const [rr, gg, bb] = o.color;
      ctx.fillStyle = `rgb(${(rr * 255) | 0},${(gg * 255) | 0},${(bb * 255) | 0})`;
      hexPath(ctx, x, y, size * 0.86);
      ctx.fill();
      ctx.strokeStyle = 'rgba(20,25,30,0.35)';
      ctx.lineWidth = 1;
      ctx.stroke();
      if (o.id === selectedId) {
        ctx.strokeStyle = '#15803d';
        ctx.lineWidth = 3;
        hexPath(ctx, x, y, size * 1.0);
        ctx.stroke();
      }
    }
  }, [snapshot, selectedId, zoom, pan]);

  // Interaction.
  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setZoom((z) => Math.min(8, Math.max(0.4, z * (e.deltaY < 0 ? 1.12 : 0.89))));
  };
  const onDown = (e: React.MouseEvent) => {
    drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
    moved.current = false;
  };
  const onMove = (e: React.MouseEvent) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) moved.current = true;
    setPan({ x: drag.current.px + dx, y: drag.current.py + dy });
  };
  const onUp = (e: React.MouseEvent) => {
    const wasDrag = moved.current;
    drag.current = null;
    if (wasDrag || !snapshot) return;
    // Click → select nearest organism in pixel space.
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const { size, ox, oy } = layout({ width: rect.width, height: rect.height } as HTMLCanvasElement, snapshot);
    let best: number | null = null;
    let bestD = Infinity;
    for (const o of snapshot.organisms) {
      const [px, py] = hexToPixel(o.q, o.r, size);
      const d = (px + ox - mx) ** 2 + (py + oy - my) ** 2;
      if (d < bestD) {
        bestD = d;
        best = o.id;
      }
    }
    if (best !== null && bestD < (size * 1.4) ** 2) onSelect(best);
  };

  return (
    <div ref={wrapRef} className="relative h-full w-full">
      <canvas
        ref={canvasRef}
        onWheel={onWheel}
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={() => (drag.current = null)}
        style={{ cursor: drag.current ? 'grabbing' : 'grab', display: 'block' }}
      />
      <div className="pointer-events-none absolute bottom-2 right-3 font-mono text-[11px] text-ink/50">
        drag to pan · scroll to zoom · click an organism
      </div>
    </div>
  );
}
