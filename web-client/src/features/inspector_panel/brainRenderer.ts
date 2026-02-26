import type { BrainState } from '../../types';
import { unwrapId } from '../../protocol';

export type BrainTransform = { x: number; y: number; scale: number };

const GEOMETRY_DOMAIN_MIN = 0;
const GEOMETRY_DOMAIN_MAX = 10;
const GEOMETRY_RENDER_SCALE = 70;
const GEOMETRY_RENDER_OFFSET = (GEOMETRY_DOMAIN_MIN + GEOMETRY_DOMAIN_MAX) * 0.5 * GEOMETRY_RENDER_SCALE;
const NODE_RADIUS = 10;
const OVERLAP_MIN_DISTANCE = NODE_RADIUS * 2.2;
const OVERLAP_RELAX_ITERS = 6;
const DEFAULT_ZOOM_INSET = 1.1;

type BrainNode = {
  id: number;
  type: string;
  label?: string;
  activation: number;
  bias: number;
  gx: number;
  gy: number;
  timeConstant?: number;
  interneuronType?: 'Excitatory' | 'Inhibitory';
  isActive: boolean;
};

type BrainNodePos = BrainNode & { x: number; y: number };

export type BrainLayout = {
  positions: Map<number, BrainNodePos>;
  bounds: { minX: number; minY: number; maxX: number; maxY: number };
};

export function computeBrainLayout(
  brain: BrainState,
  activeActionNeuronId: number | null,
): BrainLayout {
  const nodes: BrainNode[] = [];

  brain.sensory.forEach((neuron, sensoryIdx) => {
    const nid = unwrapId(neuron.neuron.neuron_id);
    nodes.push({
      id: nid,
      type: 'sensory',
      label:
        neuron.receptor_type === 'LookRay'
          ? `Vision[${neuron.ray_offset ?? 0}]: ${neuron.look_target ?? 'Look'}`
          : neuron.receptor_type,
      activation: neuron.neuron.activation,
      bias: neuron.neuron.bias,
      gx: finiteOr(neuron.neuron.x, sensoryIdx),
      gy: finiteOr(neuron.neuron.y, sensoryIdx * 0.9),
      isActive: false,
    });
  });

  brain.inter.forEach((neuron, interIdx) => {
    const nid = unwrapId(neuron.neuron.neuron_id);
    nodes.push({
      id: nid,
      type: 'inter',
      activation: neuron.neuron.activation,
      bias: neuron.neuron.bias,
      gx: finiteOr(neuron.neuron.x, 3.5 + (interIdx % 6) * 0.7),
      gy: finiteOr(neuron.neuron.y, 1 + Math.floor(interIdx / 6) * 0.9),
      timeConstant: timeConstantFromAlpha(neuron.alpha),
      interneuronType: neuron.interneuron_type,
      isActive: false,
    });
  });

  brain.action.forEach((neuron, actionIdx) => {
    const nid = unwrapId(neuron.neuron.neuron_id);
    nodes.push({
      id: nid,
      type: 'action',
      label: neuron.action_type,
      activation: neuron.neuron.activation,
      bias: neuron.neuron.bias,
      gx: finiteOr(neuron.neuron.x, 8.5),
      gy: finiteOr(neuron.neuron.y, actionIdx * 1.2 + 2),
      isActive: activeActionNeuronId === nid,
    });
  });

  const positions = new Map<number, BrainNodePos>();
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;

  const rawPositions = nodes
    .map((node) => ({
      ...node,
      x: node.gx * GEOMETRY_RENDER_SCALE - GEOMETRY_RENDER_OFFSET,
      y: node.gy * GEOMETRY_RENDER_SCALE - GEOMETRY_RENDER_OFFSET,
    }))
    .sort((a, b) => a.id - b.id);

  relaxOverlaps(rawPositions);

  for (const node of rawPositions) {
    positions.set(node.id, node);
    minX = Math.min(minX, node.x);
    minY = Math.min(minY, node.y);
    maxX = Math.max(maxX, node.x);
    maxY = Math.max(maxY, node.y);
  }

  // Pad for node radius + labels to the right
  const pad = NODE_RADIUS + 10;
  return {
    positions,
    bounds: {
      minX: minX - pad,
      minY: minY - pad,
      maxX: maxX + pad + 110,
      maxY: maxY + pad,
    },
  };
}

function finiteOr(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback;
}

function timeConstantFromAlpha(alpha: number): number {
  if (!Number.isFinite(alpha)) return Number.POSITIVE_INFINITY;

  const oneMinusAlpha = 1 - alpha;
  if (oneMinusAlpha <= 0) return 0;
  if (oneMinusAlpha >= 1) return Number.POSITIVE_INFINITY;

  const logTerm = Math.log(oneMinusAlpha);
  if (!Number.isFinite(logTerm) || logTerm === 0) return Number.POSITIVE_INFINITY;

  return -1 / logTerm;
}

function formatTimeConstant(timeConstant: number): string {
  if (!Number.isFinite(timeConstant)) return '\u221e';
  return timeConstant.toFixed(2);
}

function relaxOverlaps(nodes: BrainNodePos[]) {
  if (nodes.length < 2) return;

  const minDist = OVERLAP_MIN_DISTANCE;
  const minDistSq = minDist * minDist;

  for (let iter = 0; iter < OVERLAP_RELAX_ITERS; iter++) {
    let movedAny = false;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        let distSq = dx * dx + dy * dy;
        if (distSq >= minDistSq) continue;

        let ux = dx;
        let uy = dy;
        if (distSq < 1.0e-6) {
          const angle = (((a.id * 73856093) ^ (b.id * 19349663)) >>> 0) % 360;
          const radians = (angle * Math.PI) / 180;
          ux = Math.cos(radians);
          uy = Math.sin(radians);
          distSq = 1;
        }

        const dist = Math.sqrt(distSq);
        const nx = ux / dist;
        const ny = uy / dist;
        const push = (minDist - dist) * 0.5;
        a.x -= nx * push;
        a.y -= ny * push;
        b.x += nx * push;
        b.y += ny * push;
        movedAny = true;
      }
    }
    if (!movedAny) break;
  }
}

export function zoomToFitBrain(
  layout: BrainLayout,
  canvasW: number,
  canvasH: number,
): BrainTransform {
  const { bounds } = layout;
  const padding = 30;
  const contentW = bounds.maxX - bounds.minX + padding * 2;
  const contentH = bounds.maxY - bounds.minY + padding * 2;
  const fitScale = Math.min(canvasW / contentW, canvasH / contentH);
  const scale = Math.min(fitScale * DEFAULT_ZOOM_INSET, 2.5);
  const cx = (bounds.minX + bounds.maxX) / 2;
  const cy = (bounds.minY + bounds.maxY) / 2;
  return {
    x: canvasW / 2 - cx * scale,
    y: canvasH / 2 - cy * scale,
    scale,
  };
}

export function renderBrain(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  focusedBrain: BrainState | null,
  activeActionNeuronId: number | null,
  transform: BrainTransform,
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

  const layout = computeBrainLayout(focusedBrain, activeActionNeuronId);
  const { positions } = layout;
  const { scale } = transform;

  ctx.save();
  ctx.translate(transform.x, transform.y);
  ctx.scale(scale, scale);

  const showWeightLabels = scale > 0.65;
  const showMetrics = scale > 0.45;

  const drawSynapseWeightLabel = (
    pre: { x: number; y: number },
    post: { x: number; y: number },
    weight: number,
  ) => {
    const wLabel = weight.toFixed(2);
    const vx = post.x - pre.x;
    const vy = post.y - pre.y;
    const length = Math.hypot(vx, vy) || 1;
    const nx = -vy / length;
    const ny = vx / length;
    const midX = (pre.x + post.x) / 2 + nx * 7;
    const midY = (pre.y + post.y) / 2 + ny * 7;

    ctx.font = '10px Space Grotesk';
    const textWidth = ctx.measureText(wLabel).width;
    ctx.fillStyle = 'rgba(248, 250, 252, 0.88)';
    ctx.fillRect(midX - textWidth / 2 - 2, midY - 7, textWidth + 4, 14);

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = weight >= 0 ? '#0f4f86' : '#8a1634';
    ctx.fillText(wLabel, midX, midY);
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

    const arrowLength = Math.max(6, lineWidth * 4);
    const arrowHalfWidth = Math.max(3, lineWidth * 2.2);

    const tipX = post.x - ux * (NODE_RADIUS + 1);
    const tipY = post.y - uy * (NODE_RADIUS + 1);
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

  const drawSelfSynapse = (
    node: { x: number; y: number },
    color: string,
    lineWidth: number,
  ) => {
    const startX = node.x;
    const startY = node.y + NODE_RADIUS + 1;
    const endX = node.x;
    const endY = node.y - NODE_RADIUS - 1;
    const loopLeft = 42;
    const loopDown = 38;
    const loopUp = 38;
    const c1x = node.x - loopLeft;
    const c1y = node.y + loopDown;
    const c2x = node.x - loopLeft;
    const c2y = node.y - loopUp;

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.bezierCurveTo(c1x, c1y, c2x, c2y, endX, endY);
    ctx.stroke();

    const tipX = endX;
    const tipY = endY;
    const tangentX = endX - c2x;
    const tangentY = endY - c2y;
    const tangentLen = Math.hypot(tangentX, tangentY) || 1;
    const tx = tangentX / tangentLen;
    const ty = tangentY / tangentLen;
    const nx = -ty;
    const ny = tx;
    const arrowLen = Math.max(5, lineWidth * 3.5);
    const arrowHW = Math.max(2.5, lineWidth * 2);

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - tx * arrowLen + nx * arrowHW, tipY - ty * arrowLen + ny * arrowHW);
    ctx.lineTo(tipX - tx * arrowLen - nx * arrowHW, tipY - ty * arrowLen - ny * arrowHW);
    ctx.closePath();
    ctx.fill();
  };

  const drawSelfSynapseWeightLabel = (
    node: { x: number; y: number },
    weight: number,
  ) => {
    const labelX = node.x - 50;
    const labelY = node.y - 2;
    const wLabel = weight.toFixed(2);

    ctx.font = '10px Space Grotesk';
    const textWidth = ctx.measureText(wLabel).width;
    ctx.fillStyle = 'rgba(248, 250, 252, 0.88)';
    ctx.fillRect(labelX - textWidth / 2 - 2, labelY - 7, textWidth + 4, 14);

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = weight >= 0 ? '#0f4f86' : '#8a1634';
    ctx.fillText(wLabel, labelX, labelY);
    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
  };

  // Draw synapses
  ctx.lineWidth = 1;
  for (const neuron of focusedBrain.sensory) {
    const pre = positions.get(unwrapId(neuron.neuron.neuron_id));
    if (!pre) continue;
    for (const synapse of neuron.synapses) {
      const postId = unwrapId(synapse.post_neuron_id);
      const post = positions.get(postId);
      if (!post) continue;
      const strokeColor = synapse.weight >= 0 ? 'rgba(17,103,177,0.6)' : 'rgba(177,28,59,0.7)';
      const strokeWidth = Math.max(0.5, (Math.abs(synapse.weight) / 8) * 2);
      if (pre.id === postId) {
        drawSelfSynapse(pre, strokeColor, strokeWidth);
        if (showWeightLabels) drawSelfSynapseWeightLabel(pre, synapse.weight);
      } else {
        drawDirectedSynapse(pre, post, strokeColor, strokeWidth);
        if (showWeightLabels) drawSynapseWeightLabel(pre, post, synapse.weight);
      }
    }
  }

  for (const neuron of focusedBrain.inter) {
    const pre = positions.get(unwrapId(neuron.neuron.neuron_id));
    if (!pre) continue;
    for (const synapse of neuron.synapses) {
      const postId = unwrapId(synapse.post_neuron_id);
      const post = positions.get(postId);
      if (!post) continue;
      const strokeColor = synapse.weight >= 0 ? 'rgba(17,103,177,0.55)' : 'rgba(177,28,59,0.65)';
      const strokeWidth = Math.max(0.5, (Math.abs(synapse.weight) / 8) * 2);
      if (pre.id === postId) {
        drawSelfSynapse(pre, strokeColor, strokeWidth);
        if (showWeightLabels) drawSelfSynapseWeightLabel(pre, synapse.weight);
      } else {
        drawDirectedSynapse(pre, post, strokeColor, strokeWidth);
        if (showWeightLabels) drawSynapseWeightLabel(pre, post, synapse.weight);
      }
    }
  }

  // Draw nodes
  for (const [, node] of positions) {
    ctx.beginPath();
    ctx.fillStyle =
      node.type === 'sensory'
        ? '#f59e0b'
        : node.type === 'inter'
          ? node.interneuronType === 'Inhibitory'
            ? '#dc2626'
            : '#2563eb'
          : '#16a34a';
    ctx.arc(node.x, node.y, NODE_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    if (node.isActive) {
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    if (showMetrics) {
      ctx.fillStyle = '#10233f';
      ctx.font = '11px Space Grotesk';
      const hasLabel = typeof node.label === 'string' && node.label.length > 0;
      if (hasLabel) {
        ctx.fillText(node.label as string, node.x + 12, node.y + 4);
      }
      ctx.fillStyle = '#43556f';
      const metricsY = hasLabel ? node.y + 16 : node.y + 4;
      ctx.fillText(`h=${node.activation.toFixed(2)}`, node.x + 12, metricsY);
      if (node.type === 'inter' || node.type === 'action') {
        ctx.fillText(`b=${node.bias.toFixed(2)}`, node.x + 12, metricsY + 12);
      }
      if (node.type === 'inter') {
        if (node.timeConstant != null) {
          ctx.fillText(`\u03c4=${formatTimeConstant(node.timeConstant)}`, node.x + 12, metricsY + 24);
        }
      }
    }
  }

  ctx.restore();
}
