import type { BrainState, FacingDirection, WorldOrganismState, WorldSnapshot } from './types';
import { unwrapId } from './protocol';
import { colorForSpeciesId } from './speciesColor';

const SQRT_3 = Math.sqrt(3);
const FOOD_COLOR = '#16a34a';
const GRID_LAYER_MAX_DIMENSION = 8192;

type HexLayout = {
  size: number;
  originX: number;
  originY: number;
  worldWidth: number;
};

type GridLayerCache = {
  key: string;
  canvas: HTMLCanvasElement;
  layout: HexLayout;
};

let gridLayerCache: GridLayerCache | null = null;

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

export function computeBaseHexSize(
  canvasWidth: number,
  canvasHeight: number,
  worldWidth: number,
): number {
  const widthFactor = SQRT_3 * (1.5 * Math.max(0, worldWidth - 1) + 1);
  const heightFactor = 1.5 * Math.max(0, worldWidth - 1) + 2;
  return Math.min(canvasWidth / widthFactor, canvasHeight / heightFactor);
}

export function buildHexLayout(canvasWidth: number, canvasHeight: number, worldWidth: number): HexLayout {
  const widthFactor = SQRT_3 * (1.5 * Math.max(0, worldWidth - 1) + 1);
  const heightFactor = 1.5 * Math.max(0, worldWidth - 1) + 2;
  const size = computeBaseHexSize(canvasWidth, canvasHeight, worldWidth);

  const worldPixelWidth = widthFactor * size;
  const worldPixelHeight = heightFactor * size;

  return {
    size,
    worldWidth,
    originX: (canvasWidth - worldPixelWidth) / 2 + (SQRT_3 / 2) * size,
    originY: (canvasHeight - worldPixelHeight) / 2 + size,
  };
}

export function hexCenter(layout: HexLayout, q: number, r: number) {
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

function computeGridRenderScale(canvasWidth: number, canvasHeight: number, zoom: number): number {
  const safeZoom = Number.isFinite(zoom) && zoom > 0 ? zoom : 1;
  if (safeZoom <= 1) return 1;
  const maxCanvasDimension = Math.max(canvasWidth, canvasHeight);
  if (maxCanvasDimension <= 0) return 1;
  const maxScaleByDimension = Math.max(1, Math.floor(GRID_LAYER_MAX_DIMENSION / maxCanvasDimension));
  return Math.min(safeZoom, maxScaleByDimension);
}

function getGridLayer(
  canvasWidth: number,
  canvasHeight: number,
  worldWidth: number,
  zoom: number,
): GridLayerCache {
  const renderScale = computeGridRenderScale(canvasWidth, canvasHeight, zoom);
  const layerWidth = Math.max(1, Math.round(canvasWidth * renderScale));
  const layerHeight = Math.max(1, Math.round(canvasHeight * renderScale));
  const key = `${canvasWidth}x${canvasHeight}:${worldWidth}:${layerWidth}x${layerHeight}`;
  if (gridLayerCache && gridLayerCache.key === key) {
    return gridLayerCache;
  }

  const layer = document.createElement('canvas');
  layer.width = layerWidth;
  layer.height = layerHeight;

  const layout = buildHexLayout(canvasWidth, canvasHeight, worldWidth);
  const layerCtx = layer.getContext('2d');
  if (layerCtx) {
    const scaleX = layerWidth / canvasWidth;
    const scaleY = layerHeight / canvasHeight;
    layerCtx.save();
    layerCtx.scale(scaleX, scaleY);
    for (let r = 0; r < worldWidth; r += 1) {
      for (let q = 0; q < worldWidth; q += 1) {
        const center = hexCenter(layout, q, r);
        traceHex(layerCtx, center.x, center.y, layout.size);
        layerCtx.fillStyle = (q + r) % 2 === 0 ? '#cfd6e2' : '#e3e8f0';
        layerCtx.fill();
        layerCtx.strokeStyle = '#8a94a8';
        layerCtx.lineWidth = 0.4;
        layerCtx.stroke();
      }
    }
    layerCtx.restore();
  }

  const cacheEntry = { key, canvas: layer, layout };
  gridLayerCache = cacheEntry;
  return cacheEntry;
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
  bornFlashCells: Array<{ q: number; r: number }> | null,
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
  const gridLayer = getGridLayer(width, height, worldWidth, viewport.zoom);
  const layout = gridLayer.layout;
  ctx.drawImage(gridLayer.canvas, 0, 0, width, height);

  if (deadFlashCells) {
    for (const cell of deadFlashCells) {
      const center = hexCenter(layout, cell.q, cell.r);
      traceHex(ctx, center.x, center.y, layout.size - 0.5);
      ctx.fillStyle = 'rgba(248, 113, 113, 0.42)';
      ctx.fill();
    }
  }

  if (bornFlashCells) {
    for (const cell of bornFlashCells) {
      const center = hexCenter(layout, cell.q, cell.r);
      traceHex(ctx, center.x, center.y, layout.size - 0.5);
      ctx.fillStyle = 'rgba(134, 239, 172, 0.45)';
      ctx.fill();
    }
  }

  const foods = Array.isArray(snapshot.foods) ? snapshot.foods : [];
  for (const food of foods) {
    const center = hexCenter(layout, food.q, food.r);
    ctx.beginPath();
    ctx.arc(center.x, center.y, Math.max(2.5, layout.size * 0.24), 0, Math.PI * 2);
    ctx.fillStyle = FOOD_COLOR;
    ctx.fill();
    ctx.strokeStyle = 'rgba(15, 23, 42, 0.35)';
    ctx.lineWidth = Math.max(0.5, layout.size * 0.03);
    ctx.stroke();
  }

  for (const org of snapshot.organisms) {
    const id = unwrapId(org.id);
    const speciesId = unwrapId(org.species_id);
    const center = hexCenter(layout, org.q, org.r);

    ctx.beginPath();
    ctx.arc(center.x, center.y, Math.max(3, layout.size * 0.5), 0, Math.PI * 2);
    ctx.fillStyle = colorForSpeciesId(String(speciesId));
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
  activeNeuronIds: Set<number> | null,
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
    Array<{
      id: number;
      type: string;
      label?: string;
      activation: number;
      bias: number;
      isActive: boolean;
    }>
  > = [];
  layers.push(
    sensory.map((neuron) => {
      const nid = unwrapId(neuron.neuron.neuron_id);
      return {
        id: nid,
        type: 'sensory',
        label:
          neuron.receptor_type === 'Look'
            ? `Vision: ${neuron.look_target ?? 'Look'}`
            : neuron.receptor_type,
        activation: neuron.neuron.activation,
        bias: neuron.neuron.bias,
        isActive: activeNeuronIds?.has(nid) ?? false,
      };
    }),
  );

  const interColumns = Math.max(1, Math.ceil(inter.length / 8));
  for (let column = 0; column < interColumns; column += 1) {
    const start = column * 8;
    const slice = inter.slice(start, start + 8);
    layers.push(
      slice.map((neuron) => {
        const nid = unwrapId(neuron.neuron.neuron_id);
        return {
          id: nid,
          type: 'inter',
          activation: neuron.neuron.activation,
          bias: neuron.neuron.bias,
          isActive: activeNeuronIds?.has(nid) ?? false,
        };
      }),
    );
  }

  layers.push(
    action.map((neuron) => {
      const nid = unwrapId(neuron.neuron.neuron_id);
      return {
        id: nid,
        type: 'action',
        label: neuron.action_type,
        activation: neuron.neuron.activation,
        bias: neuron.neuron.bias,
        isActive: activeNeuronIds?.has(nid) ?? false,
      };
    }),
  );

  const xGap = width / Math.max(2, layers.length);
  const positions = new Map<
    number,
    {
      x: number;
      y: number;
      type: string;
      label?: string;
      activation: number;
      bias: number;
      isActive: boolean;
    }
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
        label: node.label,
        activation: node.activation,
        bias: node.bias,
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

  const drawDirectedSynapse = (
    pre: { x: number; y: number },
    post: { x: number; y: number },
    color: string,
    lineWidth: number,
  ) => {
    const vx = post.x - pre.x;
    const vy = post.y - pre.y;
    const length = Math.hypot(vx, vy);
    if (length < 1) return;

    const ux = vx / length;
    const uy = vy / length;
    const nx = -uy;
    const ny = ux;

    const nodeRadius = 10;
    const arrowLength = Math.max(6, lineWidth * 4);
    const arrowHalfWidth = Math.max(3, lineWidth * 2.2);

    const tipX = post.x - ux * (nodeRadius + 1);
    const tipY = post.y - uy * (nodeRadius + 1);
    const baseX = tipX - ux * arrowLength;
    const baseY = tipY - uy * arrowLength;

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(pre.x, pre.y);
    ctx.lineTo(baseX, baseY);
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(baseX + nx * arrowHalfWidth, baseY + ny * arrowHalfWidth);
    ctx.lineTo(baseX - nx * arrowHalfWidth, baseY - ny * arrowHalfWidth);
    ctx.closePath();
    ctx.fill();
  };

  ctx.lineWidth = 1;
  for (const neuron of sensory) {
    const pre = positions.get(unwrapId(neuron.neuron.neuron_id));
    if (!pre) continue;
    for (const synapse of neuron.synapses) {
      const post = positions.get(unwrapId(synapse.post_neuron_id));
      if (!post) continue;
      const strokeColor = synapse.weight >= 0 ? 'rgba(17,103,177,0.6)' : 'rgba(177,28,59,0.7)';
      const strokeWidth = Math.max(0.5, (Math.abs(synapse.weight) / 8) * 2);
      drawDirectedSynapse(pre, post, strokeColor, strokeWidth);
      drawSynapseWeightLabel(pre, post, synapse.weight);
    }
  }

  for (const neuron of inter) {
    const pre = positions.get(unwrapId(neuron.neuron.neuron_id));
    if (!pre) continue;
    for (const synapse of neuron.synapses) {
      const post = positions.get(unwrapId(synapse.post_neuron_id));
      if (!post) continue;
      const strokeColor = synapse.weight >= 0 ? 'rgba(17,103,177,0.55)' : 'rgba(177,28,59,0.65)';
      const strokeWidth = Math.max(0.5, (Math.abs(synapse.weight) / 8) * 2);
      drawDirectedSynapse(pre, post, strokeColor, strokeWidth);
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
    const hasLabel = typeof node.label === 'string' && node.label.length > 0;
    if (hasLabel) {
      ctx.fillText(node.label as string, node.x + 12, node.y + 4);
    }
    ctx.fillStyle = '#43556f';
    const metricsY = hasLabel ? node.y + 16 : node.y + 4;
    ctx.fillText(`h=${node.activation.toFixed(2)}`, node.x + 12, metricsY);
    if (node.type === 'inter') {
      ctx.fillText(`b=${node.bias.toFixed(2)}`, node.x + 12, metricsY + 12);
    }
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

  let best: { organism: WorldOrganismState; distance: number } | null = null;
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
