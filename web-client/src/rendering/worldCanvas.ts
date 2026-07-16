import type { FacingDirection, WorldOrganismState, WorldSnapshot } from '../types';

const SQRT_3 = Math.sqrt(3);
// Natural-world palette: grassland earth with subtle per-cell tonal variation
// and gray rock mountains.
const EARTH_VARIANTS = ['#b9c98f', '#b1c287', '#c1cf98', '#aabd80'];
const MOUNTAIN_VARIANTS = ['#a39d8f', '#998f7f', '#aaa496'];
const BASE_HEX_SIZE_AT_900PX = 8;
const BASE_HEX_MIN_SIZE_PX = 6;
const BASE_HEX_REFERENCE_CANVAS_PX = 900;
// Offscreen layers are rendered at higher resolution so they stay crisp when zoomed.
const LAYER_RESOLUTION_SCALE = 4;
const GRID_STROKE_COLOR = 'rgba(86, 98, 55, 0.22)';
const WALL_STROKE_COLOR = 'rgba(74, 68, 58, 0.4)';
const GRID_STROKE_WIDTH = 0.18;
const HIGH_DETAIL_MIN_SCREEN_HEX_PX = 15;
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
const ORGANISM_RADIUS_SCALE = 0.26;
const ORGANISM_MIN_RADIUS_PX = 1.2;
const ORGANISM_PICK_RADIUS_SCALE = 0.42;
const ORGANISM_FORWARD_SCALE = 1.5;
const ORGANISM_TAIL_LENGTH_SCALE = 0.95;
const ORGANISM_SIDE_SPAN_SCALE = 1.0;
const ORGANISM_CENTER_OFFSET_SCALE = 0.12;
const FOCUSED_ORGANISM_STROKE_SCALE = 0.05;
const FOCUSED_ORGANISM_MIN_STROKE_PX = 0.5;
const ORGANISM_STROKE_SCALE = 0.012;
const ORGANISM_MIN_STROKE_PX = 0.18;
// Dark strokes keep pale genome colors legible against the light terrain.
const FOCUSED_ORGANISM_STROKE_COLOR = '#16190f';
const ORGANISM_STROKE_COLOR = 'rgba(35, 41, 24, 0.55)';

const TERRAIN_NONE = 0;
const TERRAIN_WALL = 1;

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
};

type LayerCache = {
  surface: HTMLCanvasElement | null;
  width: number;
  height: number;
  worldWidth: number;
};

type TerrainLayerCache = LayerCache & { terrainSeed: number };
type OrganismLayerCache = LayerCache & {
  organisms: WorldSnapshot['organisms'] | null;
  focusedOrganismId: number | null;
};

type TerrainMaskCache = {
  worldWidth: number;
  terrainSeed: number;
  mask: Uint8Array;
};

type LayoutGeometryCache = {
  width: number;
  height: number;
  worldWidth: number;
  layout: HexLayout;
  centerXs: Float32Array;
  centerYs: Float32Array;
};

type HexSprite = {
  surface: HTMLCanvasElement;
  anchorX: number;
  anchorY: number;
};

export type WorldRenderCache = {
  terrain: TerrainLayerCache;
  organisms: OrganismLayerCache;
  terrainMask: TerrainMaskCache | null;
  geometry: LayoutGeometryCache | null;
  hexSprites: Map<string, HexSprite>;
};

export function createWorldRenderCache(): WorldRenderCache {
  return {
    terrain: { surface: null, width: 0, height: 0, worldWidth: 0, terrainSeed: Number.NaN },
    organisms: {
      surface: null,
      width: 0,
      height: 0,
      worldWidth: 0,
      organisms: null,
      focusedOrganismId: null,
    },
    terrainMask: null,
    geometry: null,
    hexSprites: new Map(),
  };
}

function clampByte(value: number) {
  return Math.round(Math.max(0, Math.min(1, value)) * 255);
}

function visualToCss(visual: { r: number; g: number; b: number }, alpha = 1) {
  const rgb = `${clampByte(visual.r)}, ${clampByte(visual.g)}, ${clampByte(visual.b)}`;
  return alpha >= 1 ? `rgb(${rgb})` : `rgba(${rgb}, ${Math.max(0, alpha)})`;
}

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

function computeWorldDimensionFactors(worldWidth: number) {
  return {
    widthFactor: SQRT_3 * (1.5 * Math.max(0, worldWidth - 1) + 1),
    heightFactor: 1.5 * Math.max(0, worldWidth - 1) + 2,
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

export function buildHexLayout(
  canvasWidth: number,
  canvasHeight: number,
  worldWidth: number,
): HexLayout {
  const { widthFactor, heightFactor } = computeWorldDimensionFactors(worldWidth);
  const size = computeBaseHexSize(canvasWidth, canvasHeight, worldWidth);

  return {
    size,
    worldWidth,
    originX: (canvasWidth - widthFactor * size) / 2 + (SQRT_3 / 2) * size,
    originY: (canvasHeight - heightFactor * size) / 2 + size,
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

function cellIndex(worldWidth: number, q: number, r: number) {
  return r * worldWidth + q;
}

// Deterministic per-cell tone pick; gives the terrain an organic, mottled look
// while keeping the variant count small enough to batch fills and cache sprites.
// Hashes q and r independently — hashing the linear index correlates along rows
// and produces visible diagonal striping.
function cellVariant(index: number, worldWidth: number, variantCount: number) {
  const q = index % worldWidth;
  const r = (index / worldWidth) | 0;
  let hash = (Math.imul(q + 1, 0x9e3779b1) ^ Math.imul(r + 1, 0x85ebca77)) >>> 0;
  hash = Math.imul(hash ^ (hash >>> 15), 0x27d4eb2f) >>> 0;
  return ((hash ^ (hash >>> 13)) >>> 0) % variantCount;
}

function getLayoutGeometry(
  cache: WorldRenderCache,
  width: number,
  height: number,
  worldWidth: number,
): LayoutGeometryCache {
  const cached = cache.geometry;
  if (cached && cached.width === width && cached.height === height && cached.worldWidth === worldWidth) {
    return cached;
  }

  const layout = buildHexLayout(width, height, worldWidth);
  const cellCount = worldWidth * worldWidth;
  const centerXs = new Float32Array(cellCount);
  const centerYs = new Float32Array(cellCount);
  const rowStepX = layout.size * SQRT_3;
  const rowStepY = layout.size * 1.5;

  let index = 0;
  for (let r = 0; r < worldWidth; r += 1) {
    const rowX = layout.originX + rowStepX * (r / 2);
    const rowY = layout.originY + rowStepY * r;
    for (let q = 0; q < worldWidth; q += 1) {
      centerXs[index] = rowX + rowStepX * q;
      centerYs[index] = rowY;
      index += 1;
    }
  }

  const geometry = { width, height, worldWidth, layout, centerXs, centerYs };
  cache.geometry = geometry;
  return geometry;
}

function getTerrainMask(cache: WorldRenderCache, snapshot: WorldSnapshot): Uint8Array {
  const worldWidth = snapshot.config.world_width;
  const terrainSeed = snapshot.rng_seed;
  const cached = cache.terrainMask;
  if (cached && cached.worldWidth === worldWidth && cached.terrainSeed === terrainSeed) {
    return cached.mask;
  }

  const mask = new Uint8Array(worldWidth * worldWidth);
  for (const cell of snapshot.terrain) {
    mask[cellIndex(worldWidth, cell.q, cell.r)] = TERRAIN_WALL;
  }
  cache.terrainMask = { worldWidth, terrainSeed, mask };
  return mask;
}

function getHexSprite(
  cache: WorldRenderCache,
  canvas: HTMLCanvasElement,
  size: number,
  fillColor: string,
  strokeColor: string,
  lineWidth: number,
): HexSprite | null {
  const key = `${size}|${fillColor}|${strokeColor}|${lineWidth}`;
  const cachedSprite = cache.hexSprites.get(key);
  if (cachedSprite) {
    return cachedSprite;
  }

  const padding = Math.max(2, Math.ceil(lineWidth * 2));
  const surface = canvas.ownerDocument.createElement('canvas');
  surface.width = Math.max(1, Math.ceil(SQRT_3 * size + padding * 2));
  surface.height = Math.max(1, Math.ceil(size * 2 + padding * 2));
  const context = surface.getContext('2d');
  if (!context) return null;

  const anchorX = surface.width / 2;
  const anchorY = surface.height / 2;
  context.beginPath();
  traceHex(context, anchorX, anchorY, size);
  context.fillStyle = fillColor;
  context.fill();
  if (lineWidth > 0) {
    context.strokeStyle = strokeColor;
    context.lineWidth = lineWidth;
    context.stroke();
  }

  const sprite = { surface, anchorX, anchorY };
  cache.hexSprites.set(key, sprite);
  return sprite;
}

function drawHexSpriteAt(
  ctx: CanvasRenderingContext2D,
  sprite: HexSprite,
  centerX: number,
  centerY: number,
) {
  ctx.drawImage(sprite.surface, centerX - sprite.anchorX, centerY - sprite.anchorY);
}

function drawOrganism(
  ctx: CanvasRenderingContext2D,
  org: WorldOrganismState,
  centerX: number,
  centerY: number,
  hexSize: number,
  isFocused: boolean,
) {
  const radius = Math.max(ORGANISM_MIN_RADIUS_PX, hexSize * ORGANISM_RADIUS_SCALE);
  const { x: ux, y: uy } = FACING_UNIT_VECTORS[org.facing];
  const anchorX = centerX - ux * radius * ORGANISM_CENTER_OFFSET_SCALE;
  const anchorY = centerY - uy * radius * ORGANISM_CENTER_OFFSET_SCALE;

  // Triangle pointing in facing direction.
  const tipX = anchorX + ux * radius * ORGANISM_FORWARD_SCALE;
  const tipY = anchorY + uy * radius * ORGANISM_FORWARD_SCALE;
  const backX = anchorX - ux * radius * ORGANISM_TAIL_LENGTH_SCALE;
  const backY = anchorY - uy * radius * ORGANISM_TAIL_LENGTH_SCALE;
  const perpX = -uy * radius * ORGANISM_SIDE_SPAN_SCALE;
  const perpY = ux * radius * ORGANISM_SIDE_SPAN_SCALE;

  const traceBody = () => {
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(backX + perpX, backY + perpY);
    ctx.lineTo(backX - perpX, backY - perpY);
    ctx.closePath();
  };

  traceBody();
  ctx.fillStyle = visualToCss(org.visual);
  ctx.fill();
  ctx.lineWidth = isFocused
    ? Math.max(FOCUSED_ORGANISM_MIN_STROKE_PX, hexSize * FOCUSED_ORGANISM_STROKE_SCALE)
    : Math.max(ORGANISM_MIN_STROKE_PX, hexSize * ORGANISM_STROKE_SCALE);
  ctx.strokeStyle = isFocused ? FOCUSED_ORGANISM_STROKE_COLOR : ORGANISM_STROKE_COLOR;
  ctx.stroke();
}

type VisibleHexRange = {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  rStart: number;
  rEnd: number;
};

function computeVisibleHexRange(
  layout: HexLayout,
  canvasWidth: number,
  canvasHeight: number,
  viewport: WorldViewport,
): VisibleHexRange {
  const topLeft = toWorldSpace(0, 0, canvasWidth, canvasHeight, viewport);
  const bottomRight = toWorldSpace(canvasWidth, canvasHeight, canvasWidth, canvasHeight, viewport);
  const minX = Math.min(topLeft.x, bottomRight.x) - layout.size * 2;
  const maxX = Math.max(topLeft.x, bottomRight.x) + layout.size * 2;
  const minY = Math.min(topLeft.y, bottomRight.y) - layout.size * 2;
  const maxY = Math.max(topLeft.y, bottomRight.y) + layout.size * 2;
  return {
    minX,
    maxX,
    minY,
    maxY,
    rStart: Math.max(0, Math.floor((minY - layout.originY) / (1.5 * layout.size) - 1)),
    rEnd: Math.min(
      layout.worldWidth - 1,
      Math.ceil((maxY - layout.originY) / (1.5 * layout.size) + 1),
    ),
  };
}

function forEachVisibleCell(
  geometry: LayoutGeometryCache,
  range: VisibleHexRange,
  callback: (index: number, centerX: number, centerY: number) => void,
) {
  const { layout } = geometry;
  const size = layout.size;
  const worldWidth = layout.worldWidth;
  for (let r = range.rStart; r <= range.rEnd; r += 1) {
    const qMinEstimate = (range.minX - layout.originX) / (SQRT_3 * size) - r / 2 - 1;
    const qMaxEstimate = (range.maxX - layout.originX) / (SQRT_3 * size) - r / 2 + 1;
    const qStart = Math.max(0, Math.floor(qMinEstimate));
    const qEnd = Math.min(worldWidth - 1, Math.ceil(qMaxEstimate));
    for (let q = qStart; q <= qEnd; q += 1) {
      const index = cellIndex(worldWidth, q, r);
      callback(index, geometry.centerXs[index], geometry.centerYs[index]);
    }
  }
}

function getLayerDimensions(canvas: HTMLCanvasElement) {
  return {
    width: Math.max(1, Math.round(canvas.width * LAYER_RESOLUTION_SCALE)),
    height: Math.max(1, Math.round(canvas.height * LAYER_RESOLUTION_SCALE)),
  };
}

function ensureLayerSurface(
  existingSurface: HTMLCanvasElement | null,
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
) {
  const surface = existingSurface ?? canvas.ownerDocument.createElement('canvas');
  if (surface.width !== width) {
    surface.width = width;
  }
  if (surface.height !== height) {
    surface.height = height;
  }
  return surface;
}

function renderTerrainLayer(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  cache: WorldRenderCache,
  width: number,
  height: number,
  snapshot: WorldSnapshot,
) {
  ctx.clearRect(0, 0, width, height);
  const geometry = getLayoutGeometry(cache, width, height, snapshot.config.world_width);
  const size = geometry.layout.size;
  // No cell strokes in the zoomed-out layer: sub-pixel strokes alias into
  // visible moiré banding when the supersampled layer is scaled down.
  const earthHexes = EARTH_VARIANTS.map((color) =>
    getHexSprite(cache, canvas, size, color, GRID_STROKE_COLOR, 0),
  );
  const wallHexes = MOUNTAIN_VARIANTS.map((color) =>
    getHexSprite(cache, canvas, size, color, WALL_STROKE_COLOR, 0),
  );
  if (earthHexes.some((sprite) => !sprite) || wallHexes.some((sprite) => !sprite)) {
    return;
  }

  const terrainMask = getTerrainMask(cache, snapshot);
  for (let index = 0; index < geometry.centerXs.length; index += 1) {
    const sprite = earthHexes[cellVariant(index, geometry.worldWidth, EARTH_VARIANTS.length)] as HexSprite;
    drawHexSpriteAt(ctx, sprite, geometry.centerXs[index], geometry.centerYs[index]);
  }
  for (let index = 0; index < geometry.centerXs.length; index += 1) {
    if (terrainMask[index] === TERRAIN_NONE) continue;
    const sprite = wallHexes[
      cellVariant(index, geometry.worldWidth, MOUNTAIN_VARIANTS.length)
    ] as HexSprite;
    drawHexSpriteAt(ctx, sprite, geometry.centerXs[index], geometry.centerYs[index]);
  }
}

function renderOrganismLayer(
  ctx: CanvasRenderingContext2D,
  cache: WorldRenderCache,
  width: number,
  height: number,
  snapshot: WorldSnapshot,
  focusedOrganismId: number | null,
) {
  ctx.clearRect(0, 0, width, height);
  const geometry = getLayoutGeometry(cache, width, height, snapshot.config.world_width);
  for (const org of snapshot.organisms) {
    const index = cellIndex(geometry.worldWidth, org.q, org.r);
    drawOrganism(
      ctx,
      org,
      geometry.centerXs[index],
      geometry.centerYs[index],
      geometry.layout.size,
      org.id === focusedOrganismId,
    );
  }
}

function getTerrainLayer(
  cache: WorldRenderCache,
  canvas: HTMLCanvasElement,
  snapshot: WorldSnapshot,
) {
  const { width, height } = getLayerDimensions(canvas);
  const worldWidth = snapshot.config.world_width;
  const terrainSeed = snapshot.rng_seed;
  const cached = cache.terrain;
  if (
    cached.surface != null &&
    cached.width === width &&
    cached.height === height &&
    cached.worldWidth === worldWidth &&
    cached.terrainSeed === terrainSeed
  ) {
    return cached.surface;
  }

  const surface = ensureLayerSurface(cached.surface, canvas, width, height);
  const context = surface.getContext('2d');
  if (!context) return null;

  renderTerrainLayer(context, canvas, cache, width, height, snapshot);
  cache.terrain = { surface, width, height, worldWidth, terrainSeed };
  return surface;
}

function getOrganismLayer(
  cache: WorldRenderCache,
  canvas: HTMLCanvasElement,
  snapshot: WorldSnapshot,
  focusedOrganismId: number | null,
) {
  const { width, height } = getLayerDimensions(canvas);
  const worldWidth = snapshot.config.world_width;
  const cached = cache.organisms;
  if (
    cached.surface != null &&
    cached.width === width &&
    cached.height === height &&
    cached.worldWidth === worldWidth &&
    cached.organisms === snapshot.organisms &&
    cached.focusedOrganismId === focusedOrganismId
  ) {
    return cached.surface;
  }

  const surface = ensureLayerSurface(cached.surface, canvas, width, height);
  const context = surface.getContext('2d');
  if (!context) return null;

  renderOrganismLayer(context, cache, width, height, snapshot, focusedOrganismId);
  cache.organisms = {
    surface,
    width,
    height,
    worldWidth,
    organisms: snapshot.organisms,
    focusedOrganismId,
  };
  return surface;
}

function renderVisibleTerrain(
  ctx: CanvasRenderingContext2D,
  cache: WorldRenderCache,
  snapshot: WorldSnapshot,
  width: number,
  height: number,
  viewport: WorldViewport,
) {
  const geometry = getLayoutGeometry(cache, width, height, snapshot.config.world_width);
  const range = computeVisibleHexRange(geometry.layout, width, height, viewport);
  const terrainMask = getTerrainMask(cache, snapshot);
  const size = geometry.layout.size;

  for (let variant = 0; variant < EARTH_VARIANTS.length; variant += 1) {
    ctx.beginPath();
    forEachVisibleCell(geometry, range, (index, centerX, centerY) => {
      if (cellVariant(index, geometry.worldWidth, EARTH_VARIANTS.length) !== variant) return;
      traceHex(ctx, centerX, centerY, size);
    });
    ctx.fillStyle = EARTH_VARIANTS[variant];
    ctx.fill();
  }

  ctx.beginPath();
  forEachVisibleCell(geometry, range, (_index, centerX, centerY) => {
    traceHex(ctx, centerX, centerY, size);
  });
  ctx.strokeStyle = GRID_STROKE_COLOR;
  ctx.lineWidth = GRID_STROKE_WIDTH;
  ctx.stroke();

  for (let variant = 0; variant < MOUNTAIN_VARIANTS.length; variant += 1) {
    ctx.beginPath();
    forEachVisibleCell(geometry, range, (index, centerX, centerY) => {
      if (terrainMask[index] !== TERRAIN_WALL) return;
      if (cellVariant(index, geometry.worldWidth, MOUNTAIN_VARIANTS.length) !== variant) return;
      traceHex(ctx, centerX, centerY, size);
    });
    ctx.fillStyle = MOUNTAIN_VARIANTS[variant];
    ctx.fill();
    ctx.strokeStyle = WALL_STROKE_COLOR;
    ctx.lineWidth = GRID_STROKE_WIDTH;
    ctx.stroke();
  }
}

function renderVisibleOrganisms(
  ctx: CanvasRenderingContext2D,
  cache: WorldRenderCache,
  snapshot: WorldSnapshot,
  focusedOrganismId: number | null,
  width: number,
  height: number,
  viewport: WorldViewport,
) {
  const geometry = getLayoutGeometry(cache, width, height, snapshot.config.world_width);
  const range = computeVisibleHexRange(geometry.layout, width, height, viewport);
  const pad = geometry.layout.size;

  for (const org of snapshot.organisms) {
    const index = cellIndex(geometry.worldWidth, org.q, org.r);
    const centerX = geometry.centerXs[index];
    const centerY = geometry.centerYs[index];
    if (
      centerX < range.minX - pad ||
      centerX > range.maxX + pad ||
      centerY < range.minY - pad ||
      centerY > range.maxY + pad
    ) {
      continue;
    }
    drawOrganism(ctx, org, centerX, centerY, geometry.layout.size, org.id === focusedOrganismId);
  }
}

export function renderWorld(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  snapshot: WorldSnapshot | null,
  focusedOrganismId: number | null,
  viewport: WorldViewport,
  visibility: RenderVisibility,
  cache: WorldRenderCache,
) {
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  if (!snapshot) {
    return;
  }

  const geometry = getLayoutGeometry(cache, width, height, snapshot.config.world_width);
  const renderHighDetail = geometry.layout.size * viewport.zoom >= HIGH_DETAIL_MIN_SCREEN_HEX_PX;

  ctx.save();
  ctx.translate(width / 2 + viewport.panX, height / 2 + viewport.panY);
  ctx.scale(viewport.zoom, viewport.zoom);
  ctx.translate(-width / 2, -height / 2);
  if (renderHighDetail) {
    renderVisibleTerrain(ctx, cache, snapshot, width, height, viewport);
    if (visibility.organisms) {
      renderVisibleOrganisms(ctx, cache, snapshot, focusedOrganismId, width, height, viewport);
    }
  } else {
    const terrainLayer = getTerrainLayer(cache, canvas, snapshot);
    if (terrainLayer) {
      ctx.drawImage(terrainLayer, 0, 0, width, height);
      if (visibility.organisms) {
        const organismLayer = getOrganismLayer(cache, canvas, snapshot, focusedOrganismId);
        if (organismLayer) ctx.drawImage(organismLayer, 0, 0, width, height);
      }
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
): WorldOrganismState | null {
  const layout = buildHexLayout(canvasWidth, canvasHeight, snapshot.config.world_width);
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

  return best?.organism ?? null;
}
