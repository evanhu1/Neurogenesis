import type { BrainState, FacingDirection, OrganismState, WorldSnapshot } from './types';
import { unwrapId } from './protocol';

const SQRT_3 = Math.sqrt(3);
const ORGANISM_COLORS = [
  '#ef4444',
  '#f97316',
  '#eab308',
  '#84cc16',
  '#22c55e',
  '#14b8a6',
  '#06b6d4',
  '#3b82f6',
  '#6366f1',
  '#ec4899',
] as const;

type HexLayout = {
  size: number;
  originX: number;
  originY: number;
  worldWidth: number;
};

export type WorldViewport = {
  zoom: number;
  panX: number;
  panY: number;
};

function toWorldSpace(
  xPx: number,
  yPx: number,
  canvasWidth: number,
  canvasHeight: number,
  viewport: WorldViewport,
) {
  return {
    x: (xPx - canvasWidth / 2 - viewport.panX) / viewport.zoom + canvasWidth / 2,
    y: (yPx - canvasHeight / 2 - viewport.panY) / viewport.zoom + canvasHeight / 2,
  };
}

function buildHexLayout(canvasWidth: number, canvasHeight: number, worldWidth: number): HexLayout {
  const widthFactor = SQRT_3 * (1.5 * Math.max(0, worldWidth - 1) + 1);
  const heightFactor = 1.5 * Math.max(0, worldWidth - 1) + 2;
  const size = Math.min(canvasWidth / widthFactor, canvasHeight / heightFactor);

  const worldPixelWidth = widthFactor * size;
  const worldPixelHeight = heightFactor * size;

  return {
    size,
    worldWidth,
    originX: (canvasWidth - worldPixelWidth) / 2 + (SQRT_3 / 2) * size,
    originY: (canvasHeight - worldPixelHeight) / 2 + size,
  };
}

function hexCenter(layout: HexLayout, q: number, r: number) {
  return {
    x: layout.originX + layout.size * SQRT_3 * (q + r / 2),
    y: layout.originY + layout.size * 1.5 * r,
  };
}

function traceHex(ctx: CanvasRenderingContext2D, cx: number, cy: number, size: number) {
  ctx.beginPath();
  for (let i = 0; i < 6; i += 1) {
    const angle = (Math.PI / 180) * (60 * i - 30);
    const x = cx + size * Math.cos(angle);
    const y = cy + size * Math.sin(angle);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.closePath();
}

function facingDelta(direction: FacingDirection): [number, number] {
  switch (direction) {
    case 'East':
      return [1, 0];
    case 'NorthEast':
      return [1, -1];
    case 'NorthWest':
      return [0, -1];
    case 'West':
      return [-1, 0];
    case 'SouthWest':
      return [-1, 1];
    case 'SouthEast':
      return [0, 1];
  }
}

export function renderWorld(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  snapshot: WorldSnapshot | null,
  focusedOrganismId: number | null,
  viewport: WorldViewport,
  deadFlashCells: Array<{ q: number; r: number }> | null,
) {
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  if (!snapshot) {
    ctx.fillStyle = '#1b2638';
    ctx.font = '20px Space Grotesk';
    ctx.fillText('Create a session to begin', 24, 40);
    return;
  }

  ctx.save();
  ctx.translate(width / 2 + viewport.panX, height / 2 + viewport.panY);
  ctx.scale(viewport.zoom, viewport.zoom);
  ctx.translate(-width / 2, -height / 2);

  const worldWidth = snapshot.config.world_width;
  const layout = buildHexLayout(width, height, worldWidth);

  for (let r = 0; r < worldWidth; r += 1) {
    for (let q = 0; q < worldWidth; q += 1) {
      const center = hexCenter(layout, q, r);
      traceHex(ctx, center.x, center.y, layout.size - 0.5);
      ctx.fillStyle = (q + r) % 2 === 0 ? '#d7dde8' : '#e3e8f0';
      ctx.fill();
      ctx.strokeStyle = '#c7d0df';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  if (deadFlashCells) {
    for (const cell of deadFlashCells) {
      const center = hexCenter(layout, cell.q, cell.r);
      traceHex(ctx, center.x, center.y, layout.size - 0.5);
      ctx.fillStyle = 'rgba(248, 113, 113, 0.42)';
      ctx.fill();
    }
  }

  for (const org of snapshot.organisms) {
    const id = unwrapId(org.id);
    const center = hexCenter(layout, org.q, org.r);

    ctx.beginPath();
    ctx.arc(center.x, center.y, Math.max(3, layout.size * 0.5), 0, Math.PI * 2);
    ctx.fillStyle = ORGANISM_COLORS[id % ORGANISM_COLORS.length];
    ctx.fill();
    if (id === focusedOrganismId) {
      ctx.lineWidth = Math.max(1.2, layout.size * 0.12);
      ctx.strokeStyle = '#0b1730';
      ctx.stroke();
    }

    const [dq, dr] = facingDelta(org.facing);
    const neighbor = hexCenter(layout, org.q + dq, org.r + dr);
    const vx = neighbor.x - center.x;
    const vy = neighbor.y - center.y;
    const norm = Math.hypot(vx, vy) || 1;
    const ux = vx / norm;
    const uy = vy / norm;

    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = Math.max(1.2, layout.size * 0.08);
    ctx.beginPath();
    ctx.moveTo(center.x, center.y);
    ctx.lineTo(center.x + ux * layout.size * 0.45, center.y + uy * layout.size * 0.45);
    ctx.stroke();
  }

  ctx.restore();

}

export function renderBrain(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  focusedBrain: BrainState | null,
) {
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  if (!focusedBrain) {
    ctx.fillStyle = '#1b2638';
    ctx.font = '15px Space Grotesk';
    ctx.fillText('No focused organism', 16, 30);
    return;
  }

  const sensory = focusedBrain.sensory;
  const inter = focusedBrain.inter;
  const action = focusedBrain.action;

  const layers: Array<
    Array<{ id: number; label: string; type: string; activation: number; isActive: boolean }>
  > = [];
  layers.push(
    sensory.map((neuron) => ({
      id: unwrapId(neuron.neuron.neuron_id),
      label: neuron.receptor_type,
      type: 'sensory',
      activation: neuron.neuron.activation,
      isActive: neuron.neuron.is_active,
    })),
  );

  const interColumns = Math.max(1, Math.ceil(inter.length / 8));
  for (let column = 0; column < interColumns; column += 1) {
    const start = column * 8;
    const slice = inter.slice(start, start + 8);
    layers.push(
      slice.map((neuron) => ({
        id: unwrapId(neuron.neuron.neuron_id),
        label: `I${unwrapId(neuron.neuron.neuron_id)}`,
        type: 'inter',
        activation: neuron.neuron.activation,
        isActive: neuron.neuron.is_active,
      })),
    );
  }

  layers.push(
    action.map((neuron) => ({
      id: unwrapId(neuron.neuron.neuron_id),
      label: neuron.action_type,
      type: 'action',
      activation: neuron.neuron.activation,
      isActive: neuron.neuron.is_active,
    })),
  );

  const xGap = width / Math.max(2, layers.length);
  const positions = new Map<
    number,
    { x: number; y: number; type: string; activation: number; label: string; isActive: boolean }
  >();

  layers.forEach((layer, layerIndex) => {
    const yGap = height / (layer.length + 1);
    layer.forEach((node, idx) => {
      const x = 20 + layerIndex * xGap;
      const y = (idx + 1) * yGap;
      positions.set(node.id, {
        x,
        y,
        type: node.type,
        activation: node.activation,
        label: node.label,
        isActive: node.isActive,
      });
    });
  });

  const drawSynapseWeightLabel = (
    pre: { x: number; y: number },
    post: { x: number; y: number },
    weight: number,
  ) => {
    const label = weight.toFixed(2);
    const vx = post.x - pre.x;
    const vy = post.y - pre.y;
    const length = Math.hypot(vx, vy) || 1;
    const nx = -vy / length;
    const ny = vx / length;
    const midX = (pre.x + post.x) / 2 + nx * 7;
    const midY = (pre.y + post.y) / 2 + ny * 7;

    ctx.font = '10px Space Grotesk';
    const textWidth = ctx.measureText(label).width;
    const textHeight = 10;
    ctx.fillStyle = 'rgba(248, 250, 252, 0.88)';
    ctx.fillRect(
      midX - textWidth / 2 - 2,
      midY - textHeight / 2 - 1,
      textWidth + 4,
      textHeight + 2,
    );

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = weight >= 0 ? '#0f4f86' : '#8a1634';
    ctx.fillText(label, midX, midY);
    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
  };

  ctx.lineWidth = 1;
  for (const neuron of sensory) {
    const pre = positions.get(unwrapId(neuron.neuron.neuron_id));
    if (!pre) continue;
    for (const synapse of neuron.synapses) {
      const post = positions.get(unwrapId(synapse.post_neuron_id));
      if (!post) continue;
      ctx.strokeStyle = synapse.weight >= 0 ? 'rgba(17,103,177,0.6)' : 'rgba(177,28,59,0.7)';
      ctx.lineWidth = Math.max(0.5, (Math.abs(synapse.weight) / 8) * 2);
      ctx.beginPath();
      ctx.moveTo(pre.x, pre.y);
      ctx.lineTo(post.x, post.y);
      ctx.stroke();
      drawSynapseWeightLabel(pre, post, synapse.weight);
    }
  }

  for (const neuron of inter) {
    const pre = positions.get(unwrapId(neuron.neuron.neuron_id));
    if (!pre) continue;
    for (const synapse of neuron.synapses) {
      const post = positions.get(unwrapId(synapse.post_neuron_id));
      if (!post) continue;
      ctx.strokeStyle = synapse.weight >= 0 ? 'rgba(17,103,177,0.55)' : 'rgba(177,28,59,0.65)';
      ctx.lineWidth = Math.max(0.5, (Math.abs(synapse.weight) / 8) * 2);
      ctx.beginPath();
      ctx.moveTo(pre.x, pre.y);
      ctx.lineTo(post.x, post.y);
      ctx.stroke();
      drawSynapseWeightLabel(pre, post, synapse.weight);
    }
  }

  for (const [, node] of positions) {
    ctx.beginPath();
    ctx.fillStyle = node.type === 'sensory' ? '#1167b1' : node.type === 'inter' ? '#1f9aa8' : '#16a34a';
    ctx.arc(node.x, node.y, 10, 0, Math.PI * 2);
    ctx.fill();
    if (node.isActive) {
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    ctx.fillStyle = '#10233f';
    ctx.font = '11px Space Grotesk';
    ctx.fillText(node.label, node.x + 12, node.y + 4);
    ctx.fillStyle = '#43556f';
    ctx.fillText(node.activation.toFixed(2), node.x + 12, node.y + 16);
  }
}

export function pickOrganismAtCanvasPoint(
  snapshot: WorldSnapshot,
  canvasWidth: number,
  canvasHeight: number,
  xPx: number,
  yPx: number,
  viewport: WorldViewport,
) {
  const worldWidth = snapshot.config.world_width;
  const layout = buildHexLayout(canvasWidth, canvasHeight, worldWidth);
  const point = toWorldSpace(xPx, yPx, canvasWidth, canvasHeight, viewport);

  let best: { organism: OrganismState; distance: number } | null = null;
  for (const organism of snapshot.organisms) {
    const center = hexCenter(layout, organism.q, organism.r);
    const distance = Math.hypot(center.x - point.x, center.y - point.y);
    if (distance > layout.size * 0.42) continue;
    if (!best || distance < best.distance) {
      best = { organism, distance };
    }
  }

  if (!best) return null;
  return {
    organism: best.organism,
    gridQ: best.organism.q,
    gridR: best.organism.r,
  };
}
