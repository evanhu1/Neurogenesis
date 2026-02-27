import type { FacingDirection, WorldOrganismState, WorldSnapshot } from '../../types';
import { unwrapId } from '../../protocol';
import { colorForSpeciesId } from '../../speciesColor';

const SQRT_3 = Math.sqrt(3);
const PLANT_COLOR = '#16a34a';
const WALL_COLOR = '#5f6572';
const BASE_HEX_SIZE_AT_900PX = 8;
const BASE_HEX_MIN_SIZE_PX = 6;
const BASE_HEX_REFERENCE_CANVAS_PX = 900;
const EARTH_COLOR = '#d4c4a8';
const HEX_VERTEX_OFFSETS = Array.from({ length: 6 }, (_, index) => {
  const angle = (Math.PI / 180) * (60 * index - 30);
  return {
    x: Math.cos(angle),
    y: Math.sin(angle),
  };
});

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

export type RenderVisibility = {
  organisms: boolean;
  plants: boolean;
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
  _worldWidth: number,
): number {
  const minCanvasDimension = Math.max(1, Math.min(canvasWidth, canvasHeight));
  const scaledSize = (minCanvasDimension / BASE_HEX_REFERENCE_CANVAS_PX) * BASE_HEX_SIZE_AT_900PX;
  return Math.max(BASE_HEX_MIN_SIZE_PX, scaledSize);
}

function computeWorldDimensionFactors(worldWidth: number) {
  return {
    widthFactor: SQRT_3 * (1.5 * Math.max(0, worldWidth - 1) + 1),
    heightFactor: 1.5 * Math.max(0, worldWidth - 1) + 2,
  };
}

export function computeWorldFitZoom(
  canvasWidth: number,
  canvasHeight: number,
  worldWidth: number,
): number {
  if (!Number.isFinite(worldWidth) || worldWidth <= 0) return 1;
  const size = computeBaseHexSize(canvasWidth, canvasHeight, worldWidth);
  if (!Number.isFinite(size) || size <= 0) return 1;
  const { widthFactor, heightFactor } = computeWorldDimensionFactors(worldWidth);
  const worldPixelWidth = widthFactor * size;
  const worldPixelHeight = heightFactor * size;
  if (worldPixelWidth <= 0 || worldPixelHeight <= 0) return 1;
  const fitZoom = Math.min(canvasWidth / worldPixelWidth, canvasHeight / worldPixelHeight);
  if (!Number.isFinite(fitZoom) || fitZoom <= 0) return 1;
  return Math.min(1, fitZoom);
}

export function buildHexLayout(canvasWidth: number, canvasHeight: number, worldWidth: number): HexLayout {
  const { widthFactor, heightFactor } = computeWorldDimensionFactors(worldWidth);
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
  for (let index = 0; index < HEX_VERTEX_OFFSETS.length; index += 1) {
    const vertex = HEX_VERTEX_OFFSETS[index];
    const x = cx + size * vertex.x;
    const y = cy + size * vertex.y;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.closePath();
}

function drawVisibleGrid(
  ctx: CanvasRenderingContext2D,
  layout: HexLayout,
  minX: number,
  maxX: number,
  minY: number,
  maxY: number,
) {
  const size = layout.size;
  const worldWidth = layout.worldWidth;
  if (size <= 0 || worldWidth <= 0) return;

  const rMinEstimate = (minY - layout.originY) / (1.5 * size) - 1;
  const rMaxEstimate = (maxY - layout.originY) / (1.5 * size) + 1;
  const rStart = Math.max(0, Math.floor(rMinEstimate));
  const rEnd = Math.min(worldWidth - 1, Math.ceil(rMaxEstimate));

  ctx.beginPath();
  for (let r = rStart; r <= rEnd; r += 1) {
    const qMinEstimate = (minX - layout.originX) / (SQRT_3 * size) - r / 2 - 1;
    const qMaxEstimate = (maxX - layout.originX) / (SQRT_3 * size) - r / 2 + 1;
    const qStart = Math.max(0, Math.floor(qMinEstimate));
    const qEnd = Math.min(worldWidth - 1, Math.ceil(qMaxEstimate));

    for (let q = qStart; q <= qEnd; q += 1) {
      const center = hexCenter(layout, q, r);
      traceHex(ctx, center.x, center.y, size);
    }
  }
  ctx.fillStyle = EARTH_COLOR;
  ctx.fill();
  ctx.strokeStyle = '#8a94a8';
  ctx.lineWidth = 0.4;
  ctx.stroke();
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
  visibility: RenderVisibility = { organisms: true, plants: true },
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
  const topLeft = toWorldSpace(0, 0, width, height, viewport);
  const bottomRight = toWorldSpace(width, height, width, height, viewport);
  const minX = Math.min(topLeft.x, bottomRight.x) - layout.size * 2;
  const maxX = Math.max(topLeft.x, bottomRight.x) + layout.size * 2;
  const minY = Math.min(topLeft.y, bottomRight.y) - layout.size * 2;
  const maxY = Math.max(topLeft.y, bottomRight.y) + layout.size * 2;
  drawVisibleGrid(ctx, layout, minX, maxX, minY, maxY);

  const occupancy = Array.isArray(snapshot.occupancy) ? snapshot.occupancy : [];
  ctx.beginPath();
  for (const cell of occupancy) {
    if (cell.occupant.type !== 'Wall') continue;
    const center = hexCenter(layout, cell.q, cell.r);
    traceHex(ctx, center.x, center.y, layout.size);
  }
  ctx.fillStyle = WALL_COLOR;
  ctx.fill();
  ctx.strokeStyle = '#4d5360';
  ctx.lineWidth = 0.4;
  ctx.stroke();

  if (visibility.plants) {
    const plants = Array.isArray(snapshot.foods) ? snapshot.foods : [];
    ctx.beginPath();
    for (const plant of plants) {
      const center = hexCenter(layout, plant.q, plant.r);
      traceHex(ctx, center.x, center.y, layout.size);
    }
    ctx.fillStyle = PLANT_COLOR;
    ctx.fill();
    ctx.strokeStyle = '#8a94a8';
    ctx.lineWidth = 0.4;
    ctx.stroke();
  }

  if (visibility.organisms) {
    for (const org of snapshot.organisms) {
      const id = unwrapId(org.id);
      const speciesId = unwrapId(org.species_id);
      const center = hexCenter(layout, org.q, org.r);

      const radius = Math.max(3, layout.size * 0.6);
      const [dq, dr] = facingDelta(org.facing);
      const neighbor = hexCenter(layout, org.q + dq, org.r + dr);
      const vx = neighbor.x - center.x;
      const vy = neighbor.y - center.y;
      const norm = Math.hypot(vx, vy) || 1;
      const ux = vx / norm;
      const uy = vy / norm;

      // Triangle pointing in facing direction
      const tipX = center.x + ux * radius;
      const tipY = center.y + uy * radius;
      const backX = center.x - ux * radius * 0.7;
      const backY = center.y - uy * radius * 0.7;
      const perpX = -uy * radius * 0.65;
      const perpY = ux * radius * 0.65;

      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(backX + perpX, backY + perpY);
      ctx.lineTo(backX - perpX, backY - perpY);
      ctx.closePath();
      ctx.fillStyle = colorForSpeciesId(String(speciesId));
      ctx.fill();
      ctx.lineWidth = id === focusedOrganismId ? Math.max(1.5, layout.size * 0.14) : Math.max(0.8, layout.size * 0.06);
      ctx.strokeStyle = id === focusedOrganismId ? '#0b1730' : 'rgba(15, 23, 42, 0.6)';
      ctx.stroke();
    }
  }

  ctx.restore();

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
