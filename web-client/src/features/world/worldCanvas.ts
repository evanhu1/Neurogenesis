import type { FacingDirection, WorldOrganismState, WorldSnapshot } from '../../types';
import { unwrapId } from '../../protocol';
import { colorForSpeciesId } from '../../speciesColor';

const SQRT_3 = Math.sqrt(3);
const PLANT_COLOR = '#16a34a';
const WALL_COLOR = '#5f6572';
const BASE_HEX_SIZE_AT_900PX = 8;
const BASE_HEX_MIN_SIZE_PX = 6;
const BASE_HEX_REFERENCE_CANVAS_PX = 900;
const LOW_DETAIL_LAYER_RESOLUTION_SCALE = 2;
const EARTH_COLOR = '#d4c4a8';
const GRID_STROKE_COLOR = '#8a94a8';
const WALL_STROKE_COLOR = '#4d5360';
const GRID_STROKE_WIDTH = 0.18;
const HIGH_DETAIL_MIN_SCREEN_HEX_PX = 12;
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
const ORGANISM_PICK_RADIUS_SCALE = 0.42;
const ORGANISM_TAIL_LENGTH_SCALE = 0.7;
const ORGANISM_SIDE_SPAN_SCALE = 0.34;
const FOCUSED_ORGANISM_STROKE_SCALE = 0.035;
const FOCUSED_ORGANISM_MIN_STROKE_PX = 0.35;
const ORGANISM_STROKE_SCALE = 0.012;
const ORGANISM_MIN_STROKE_PX = 0.18;
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

type LayerSurface = HTMLCanvasElement;

type TerrainLayerCache = {
  surface: LayerSurface | null;
  width: number;
  height: number;
  worldWidth: number;
  terrainSeed: number;
  wallMask: Uint8Array | null;
};

type PlantLayerCache = {
  surface: LayerSurface | null;
  width: number;
  height: number;
  worldWidth: number;
  foods: WorldSnapshot['foods'] | null;
  foodMask: Uint8Array | null;
};

type OrganismLayerCache = {
  surface: LayerSurface | null;
  width: number;
  height: number;
  worldWidth: number;
  organisms: WorldSnapshot['organisms'] | null;
  focusedOrganismId: number | null;
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
  surface: LayerSurface;
  anchorX: number;
  anchorY: number;
};

export type WorldRenderCache = {
  terrain: TerrainLayerCache;
  plants: PlantLayerCache;
  organisms: OrganismLayerCache;
  geometry: LayoutGeometryCache | null;
  hexSprites: Map<string, HexSprite>;
};

export function createWorldRenderCache(): WorldRenderCache {
  return {
    terrain: {
      surface: null,
      width: 0,
      height: 0,
      worldWidth: 0,
      terrainSeed: Number.NaN,
      wallMask: null,
    },
    plants: {
      surface: null,
      width: 0,
      height: 0,
      worldWidth: 0,
      foods: null,
      foodMask: null,
    },
    organisms: {
      surface: null,
      width: 0,
      height: 0,
      worldWidth: 0,
      organisms: null,
      focusedOrganismId: null,
    },
    geometry: null,
    hexSprites: new Map(),
  };
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

function getLayoutGeometry(
  cache: WorldRenderCache,
  width: number,
  height: number,
  worldWidth: number,
) {
  const cachedGeometry = cache.geometry;
  if (
    cachedGeometry &&
    cachedGeometry.width === width &&
    cachedGeometry.height === height &&
    cachedGeometry.worldWidth === worldWidth
  ) {
    return cachedGeometry;
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

  const geometry = {
    width,
    height,
    worldWidth,
    layout,
    centerXs,
    centerYs,
  };
  cache.geometry = geometry;
  return geometry;
}

function getHexSprite(
  cache: WorldRenderCache,
  canvas: HTMLCanvasElement,
  size: number,
  fillColor: string,
  strokeColor: string,
  lineWidth: number,
) {
  const key = `${size}|${fillColor}|${strokeColor}|${lineWidth}`;
  const cachedSprite = cache.hexSprites.get(key);
  if (cachedSprite) {
    return cachedSprite;
  }

  const padding = Math.max(2, Math.ceil(lineWidth * 2));
  const spriteWidth = Math.max(1, Math.ceil(SQRT_3 * size + padding * 2));
  const spriteHeight = Math.max(1, Math.ceil(size * 2 + padding * 2));
  const surface = canvas.ownerDocument.createElement('canvas');
  surface.width = spriteWidth;
  surface.height = spriteHeight;
  const context = surface.getContext('2d');
  if (!context) return null;

  const anchorX = spriteWidth / 2;
  const anchorY = spriteHeight / 2;
  context.beginPath();
  traceHex(context, anchorX, anchorY, size);
  context.fillStyle = fillColor;
  context.fill();
  context.strokeStyle = strokeColor;
  context.lineWidth = lineWidth;
  context.stroke();

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

function cellIndex(worldWidth: number, q: number, r: number) {
  return r * worldWidth + q;
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
  const rMinEstimate = (minY - layout.originY) / (1.5 * layout.size) - 1;
  const rMaxEstimate = (maxY - layout.originY) / (1.5 * layout.size) + 1;
  return {
    minX,
    maxX,
    minY,
    maxY,
    rStart: Math.max(0, Math.floor(rMinEstimate)),
    rEnd: Math.min(layout.worldWidth - 1, Math.ceil(rMaxEstimate)),
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

function buildWallMask(snapshot: WorldSnapshot) {
  const worldWidth = snapshot.config.world_width;
  const wallMask = new Uint8Array(worldWidth * worldWidth);
  const occupancy = Array.isArray(snapshot.occupancy) ? snapshot.occupancy : [];
  for (const cell of occupancy) {
    if (cell.occupant.type !== 'Wall') continue;
    wallMask[cellIndex(worldWidth, cell.q, cell.r)] = 1;
  }
  return wallMask;
}

function getWallMask(cache: WorldRenderCache, snapshot: WorldSnapshot): Uint8Array {
  const worldWidth = snapshot.config.world_width;
  const terrainSeed = snapshot.rng_seed;
  const shouldReuse =
    cache.terrain.wallMask != null &&
    cache.terrain.worldWidth === worldWidth &&
    cache.terrain.terrainSeed === terrainSeed;
  if (shouldReuse) {
    return cache.terrain.wallMask as Uint8Array;
  }
  const wallMask = buildWallMask(snapshot);
  cache.terrain.worldWidth = worldWidth;
  cache.terrain.terrainSeed = terrainSeed;
  cache.terrain.wallMask = wallMask;
  return wallMask;
}

function buildFoodMask(snapshot: WorldSnapshot) {
  const worldWidth = snapshot.config.world_width;
  const foodMask = new Uint8Array(worldWidth * worldWidth);
  const foods = Array.isArray(snapshot.foods) ? snapshot.foods : [];
  for (const food of foods) {
    foodMask[cellIndex(worldWidth, food.q, food.r)] = 1;
  }
  return foodMask;
}

function getFoodMask(cache: WorldRenderCache, snapshot: WorldSnapshot): Uint8Array {
  const worldWidth = snapshot.config.world_width;
  const foods = snapshot.foods;
  const shouldReuse =
    cache.plants.foodMask != null &&
    cache.plants.worldWidth === worldWidth &&
    cache.plants.foods === foods;
  if (shouldReuse) {
    return cache.plants.foodMask as Uint8Array;
  }
  const foodMask = buildFoodMask(snapshot);
  cache.plants.worldWidth = worldWidth;
  cache.plants.foods = foods;
  cache.plants.foodMask = foodMask;
  return foodMask;
}

function ensureLayerSurface(
  existingSurface: LayerSurface | null,
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

function getLowDetailLayerDimensions(canvas: HTMLCanvasElement) {
  return {
    width: Math.max(1, Math.round(canvas.width * LOW_DETAIL_LAYER_RESOLUTION_SCALE)),
    height: Math.max(1, Math.round(canvas.height * LOW_DETAIL_LAYER_RESOLUTION_SCALE)),
  };
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
  const earthHex = getHexSprite(cache, canvas, geometry.layout.size, EARTH_COLOR, GRID_STROKE_COLOR, GRID_STROKE_WIDTH);
  const wallHex = getHexSprite(cache, canvas, geometry.layout.size, WALL_COLOR, WALL_STROKE_COLOR, GRID_STROKE_WIDTH);
  if (!earthHex || !wallHex) return;

  for (let index = 0; index < geometry.centerXs.length; index += 1) {
    drawHexSpriteAt(ctx, earthHex, geometry.centerXs[index], geometry.centerYs[index]);
  }

  const occupancy = Array.isArray(snapshot.occupancy) ? snapshot.occupancy : [];
  for (const cell of occupancy) {
    if (cell.occupant.type !== 'Wall') continue;
    const index = cellIndex(geometry.worldWidth, cell.q, cell.r);
    drawHexSpriteAt(ctx, wallHex, geometry.centerXs[index], geometry.centerYs[index]);
  }
}

function renderPlantLayer(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  cache: WorldRenderCache,
  width: number,
  height: number,
  snapshot: WorldSnapshot,
) {
  ctx.clearRect(0, 0, width, height);
  const geometry = getLayoutGeometry(cache, width, height, snapshot.config.world_width);
  const plantHex = getHexSprite(cache, canvas, geometry.layout.size, PLANT_COLOR, GRID_STROKE_COLOR, GRID_STROKE_WIDTH);
  if (!plantHex) return;

  const plants = Array.isArray(snapshot.foods) ? snapshot.foods : [];
  for (const plant of plants) {
    const index = cellIndex(geometry.worldWidth, plant.q, plant.r);
    drawHexSpriteAt(ctx, plantHex, geometry.centerXs[index], geometry.centerYs[index]);
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
  const { layout } = geometry;

  for (const org of snapshot.organisms) {
    const id = unwrapId(org.id);
    const speciesId = unwrapId(org.species_id);
    const index = cellIndex(geometry.worldWidth, org.q, org.r);
    const centerX = geometry.centerXs[index];
    const centerY = geometry.centerYs[index];

    const radius = Math.max(3, layout.size * ORGANISM_RADIUS_SCALE);
    const { x: ux, y: uy } = FACING_UNIT_VECTORS[org.facing];

    // Triangle pointing in facing direction
    const tipX = centerX + ux * radius;
    const tipY = centerY + uy * radius;
    const backX = centerX - ux * radius * ORGANISM_TAIL_LENGTH_SCALE;
    const backY = centerY - uy * radius * ORGANISM_TAIL_LENGTH_SCALE;
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

function getTerrainLayer(
  cache: WorldRenderCache,
  canvas: HTMLCanvasElement,
  snapshot: WorldSnapshot,
) {
  const { width, height } = getLowDetailLayerDimensions(canvas);
  const worldWidth = snapshot.config.world_width;
  const terrainSeed = snapshot.rng_seed;
  const shouldReuse =
    cache.terrain.surface != null &&
    cache.terrain.width === width &&
    cache.terrain.height === height &&
    cache.terrain.worldWidth === worldWidth &&
    cache.terrain.terrainSeed === terrainSeed;

  if (shouldReuse) {
    return cache.terrain.surface;
  }

  const surface = ensureLayerSurface(cache.terrain.surface, canvas, width, height);
  const context = surface.getContext('2d');
  if (!context) return null;

  renderTerrainLayer(context, canvas, cache, width, height, snapshot);
  cache.terrain = {
    surface,
    width,
    height,
    worldWidth,
    terrainSeed,
    wallMask: cache.terrain.wallMask,
  };
  return surface;
}

function getPlantLayer(
  cache: WorldRenderCache,
  canvas: HTMLCanvasElement,
  snapshot: WorldSnapshot,
) {
  const { width, height } = getLowDetailLayerDimensions(canvas);
  const worldWidth = snapshot.config.world_width;
  const foods = snapshot.foods;
  const shouldReuse =
    cache.plants.surface != null &&
    cache.plants.width === width &&
    cache.plants.height === height &&
    cache.plants.worldWidth === worldWidth &&
    cache.plants.foods === foods;

  if (shouldReuse) {
    return cache.plants.surface;
  }

  const surface = ensureLayerSurface(cache.plants.surface, canvas, width, height);
  const context = surface.getContext('2d');
  if (!context) return null;

  renderPlantLayer(context, canvas, cache, width, height, snapshot);
  cache.plants = {
    surface,
    width,
    height,
    worldWidth,
    foods,
    foodMask: cache.plants.foodMask,
  };
  return surface;
}

function getOrganismLayer(
  cache: WorldRenderCache,
  canvas: HTMLCanvasElement,
  snapshot: WorldSnapshot,
  focusedOrganismId: number | null,
) {
  const { width, height } = getLowDetailLayerDimensions(canvas);
  const worldWidth = snapshot.config.world_width;
  const organisms = snapshot.organisms;
  const shouldReuse =
    cache.organisms.surface != null &&
    cache.organisms.width === width &&
    cache.organisms.height === height &&
    cache.organisms.worldWidth === worldWidth &&
    cache.organisms.organisms === organisms &&
    cache.organisms.focusedOrganismId === focusedOrganismId;

  if (shouldReuse) {
    return cache.organisms.surface;
  }

  const surface = ensureLayerSurface(cache.organisms.surface, canvas, width, height);
  const context = surface.getContext('2d');
  if (!context) return null;

  renderOrganismLayer(context, cache, width, height, snapshot, focusedOrganismId);
  cache.organisms = {
    surface,
    width,
    height,
    worldWidth,
    organisms,
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
  const wallMask = getWallMask(cache, snapshot);

  ctx.beginPath();
  forEachVisibleCell(geometry, range, (_index, centerX, centerY) => {
    traceHex(ctx, centerX, centerY, geometry.layout.size);
  });
  ctx.fillStyle = EARTH_COLOR;
  ctx.fill();
  ctx.strokeStyle = GRID_STROKE_COLOR;
  ctx.lineWidth = GRID_STROKE_WIDTH;
  ctx.stroke();

  ctx.beginPath();
  forEachVisibleCell(geometry, range, (index, centerX, centerY) => {
    if (wallMask[index] === 0) return;
    traceHex(ctx, centerX, centerY, geometry.layout.size);
  });
  ctx.fillStyle = WALL_COLOR;
  ctx.fill();
  ctx.strokeStyle = WALL_STROKE_COLOR;
  ctx.lineWidth = GRID_STROKE_WIDTH;
  ctx.stroke();
}

function renderVisiblePlants(
  ctx: CanvasRenderingContext2D,
  cache: WorldRenderCache,
  snapshot: WorldSnapshot,
  width: number,
  height: number,
  viewport: WorldViewport,
) {
  const geometry = getLayoutGeometry(cache, width, height, snapshot.config.world_width);
  const range = computeVisibleHexRange(geometry.layout, width, height, viewport);
  const foodMask = getFoodMask(cache, snapshot);

  ctx.beginPath();
  forEachVisibleCell(geometry, range, (index, centerX, centerY) => {
    if (foodMask[index] === 0) return;
    traceHex(ctx, centerX, centerY, geometry.layout.size);
  });
  ctx.fillStyle = PLANT_COLOR;
  ctx.fill();
  ctx.strokeStyle = GRID_STROKE_COLOR;
  ctx.lineWidth = GRID_STROKE_WIDTH;
  ctx.stroke();
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
  const minX = range.minX - geometry.layout.size;
  const maxX = range.maxX + geometry.layout.size;
  const minY = range.minY - geometry.layout.size;
  const maxY = range.maxY + geometry.layout.size;

  for (const org of snapshot.organisms) {
    const id = unwrapId(org.id);
    const speciesId = unwrapId(org.species_id);
    const index = cellIndex(geometry.worldWidth, org.q, org.r);
    const centerX = geometry.centerXs[index];
    const centerY = geometry.centerYs[index];
    if (centerX < minX || centerX > maxX || centerY < minY || centerY > maxY) {
      continue;
    }

    const radius = Math.max(3, geometry.layout.size * ORGANISM_RADIUS_SCALE);
    const { x: ux, y: uy } = FACING_UNIT_VECTORS[org.facing];
    const tipX = centerX + ux * radius;
    const tipY = centerY + uy * radius;
    const backX = centerX - ux * radius * ORGANISM_TAIL_LENGTH_SCALE;
    const backY = centerY - uy * radius * ORGANISM_TAIL_LENGTH_SCALE;
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
        ? Math.max(FOCUSED_ORGANISM_MIN_STROKE_PX, geometry.layout.size * FOCUSED_ORGANISM_STROKE_SCALE)
        : Math.max(ORGANISM_MIN_STROKE_PX, geometry.layout.size * ORGANISM_STROKE_SCALE);
    ctx.strokeStyle =
      id === focusedOrganismId ? FOCUSED_ORGANISM_STROKE_COLOR : ORGANISM_STROKE_COLOR;
    ctx.stroke();
  }
}

export function renderWorld(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  snapshot: WorldSnapshot | null,
  focusedOrganismId: number | null,
  viewport: WorldViewport,
  visibility: RenderVisibility = { organisms: true, plants: true },
  cache: WorldRenderCache = createWorldRenderCache(),
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

  const geometry = getLayoutGeometry(cache, width, height, snapshot.config.world_width);
  const renderHighDetail = geometry.layout.size * viewport.zoom >= HIGH_DETAIL_MIN_SCREEN_HEX_PX;
  const terrainLayer = renderHighDetail ? null : getTerrainLayer(cache, canvas, snapshot);
  const plantLayer =
    renderHighDetail || !visibility.plants ? null : getPlantLayer(cache, canvas, snapshot);
  const organismLayer =
    renderHighDetail || !visibility.organisms
      ? null
      : getOrganismLayer(cache, canvas, snapshot, focusedOrganismId);

  ctx.save();
  ctx.translate(width / 2 + viewport.panX, height / 2 + viewport.panY);
  ctx.scale(viewport.zoom, viewport.zoom);
  ctx.translate(-width / 2, -height / 2);
  if (renderHighDetail) {
    renderVisibleTerrain(ctx, cache, snapshot, width, height, viewport);
    if (visibility.plants) {
      renderVisiblePlants(ctx, cache, snapshot, width, height, viewport);
    }
    if (visibility.organisms) {
      renderVisibleOrganisms(ctx, cache, snapshot, focusedOrganismId, width, height, viewport);
    }
  } else if (terrainLayer) {
    ctx.drawImage(terrainLayer, 0, 0, width, height);
    if (plantLayer) {
      ctx.drawImage(plantLayer, 0, 0, width, height);
    }
    if (organismLayer) {
      ctx.drawImage(organismLayer, 0, 0, width, height);
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
