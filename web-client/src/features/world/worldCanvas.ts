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
const GRID_STROKE_COLOR = '#8a94a8';
const WALL_STROKE_COLOR = '#4d5360';
const GRID_STROKE_WIDTH = 0.4;
const HEX_VERTEX_OFFSETS = Array.from({ length: 6 }, (_, index) => {
  const angle = (Math.PI / 180) * (60 * index - 30);
  return {
    x: Math.cos(angle),
    y: Math.sin(angle),
  };
});
const HEX_DIAGONAL_UNIT_X = 0.5;
const HEX_DIAGONAL_UNIT_Y = SQRT_3 / 2;
const FACING_UNIT_VECTORS: Record<FacingDirection, { x: number; y: number }> = {
  East: { x: 1, y: 0 },
  NorthEast: { x: HEX_DIAGONAL_UNIT_X, y: -HEX_DIAGONAL_UNIT_Y },
  NorthWest: { x: -HEX_DIAGONAL_UNIT_X, y: -HEX_DIAGONAL_UNIT_Y },
  West: { x: -1, y: 0 },
  SouthWest: { x: -HEX_DIAGONAL_UNIT_X, y: HEX_DIAGONAL_UNIT_Y },
  SouthEast: { x: HEX_DIAGONAL_UNIT_X, y: HEX_DIAGONAL_UNIT_Y },
};
const ORGANISM_RADIUS_SCALE = 0.6;
const ORGANISM_PICK_RADIUS_SCALE = 0.42;
const ORGANISM_TAIL_LENGTH_SCALE = 0.7;
const ORGANISM_SIDE_SPAN_SCALE = 0.65;
const FOCUSED_ORGANISM_STROKE_SCALE = 0.14;
const FOCUSED_ORGANISM_MIN_STROKE_PX = 1.5;
const ORGANISM_STROKE_SCALE = 0.06;
const ORGANISM_MIN_STROKE_PX = 0.8;
const FOCUSED_ORGANISM_STROKE_COLOR = '#0b1730';
const ORGANISM_STROKE_COLOR = 'rgba(15, 23, 42, 0.6)';

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
  worldWidth: number,
): number {
  const minCanvasDimension = Math.max(1, Math.min(canvasWidth, canvasHeight));
  const scaledSize = (minCanvasDimension / BASE_HEX_REFERENCE_CANVAS_PX) * BASE_HEX_SIZE_AT_900PX;
  if (!Number.isFinite(worldWidth) || worldWidth <= 0) {
    return Math.max(BASE_HEX_MIN_SIZE_PX, scaledSize);
  }

  const { widthFactor, heightFactor } = computeWorldDimensionFactors(worldWidth);
  const fitLimitedSize = Math.min(canvasWidth / widthFactor, canvasHeight / heightFactor);
  if (!Number.isFinite(fitLimitedSize) || fitLimitedSize <= 0) {
    return Math.max(BASE_HEX_MIN_SIZE_PX, scaledSize);
  }

  const minimumSize = Math.min(BASE_HEX_MIN_SIZE_PX, fitLimitedSize);
  return Math.max(minimumSize, Math.min(scaledSize, fitLimitedSize));
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
  ctx.strokeStyle = GRID_STROKE_COLOR;
  ctx.lineWidth = GRID_STROKE_WIDTH;
  ctx.stroke();
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
  ctx.strokeStyle = WALL_STROKE_COLOR;
  ctx.lineWidth = GRID_STROKE_WIDTH;
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
    ctx.strokeStyle = GRID_STROKE_COLOR;
    ctx.lineWidth = GRID_STROKE_WIDTH;
    ctx.stroke();
  }

  if (visibility.organisms) {
    for (const org of snapshot.organisms) {
      const id = unwrapId(org.id);
      const speciesId = unwrapId(org.species_id);
      const center = hexCenter(layout, org.q, org.r);

      const radius = Math.max(3, layout.size * ORGANISM_RADIUS_SCALE);
      const { x: ux, y: uy } = FACING_UNIT_VECTORS[org.facing];

      // Triangle pointing in facing direction
      const tipX = center.x + ux * radius;
      const tipY = center.y + uy * radius;
      const backX = center.x - ux * radius * ORGANISM_TAIL_LENGTH_SCALE;
      const backY = center.y - uy * radius * ORGANISM_TAIL_LENGTH_SCALE;
      const perpX = -uy * radius * ORGANISM_SIDE_SPAN_SCALE;
      const perpY = ux * radius * ORGANISM_SIDE_SPAN_SCALE;

      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(backX + perpX, backY + perpY);
      ctx.lineTo(backX - perpX, backY - perpY);
      ctx.closePath();
      ctx.fillStyle = colorForSpeciesId(String(speciesId));
      ctx.fill();
      ctx.lineWidth =
        id === focusedOrganismId
          ? Math.max(FOCUSED_ORGANISM_MIN_STROKE_PX, layout.size * FOCUSED_ORGANISM_STROKE_SCALE)
          : Math.max(ORGANISM_MIN_STROKE_PX, layout.size * ORGANISM_STROKE_SCALE);
      ctx.strokeStyle =
        id === focusedOrganismId ? FOCUSED_ORGANISM_STROKE_COLOR : ORGANISM_STROKE_COLOR;
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
    if (distance > layout.size * ORGANISM_PICK_RADIUS_SCALE) continue;
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
