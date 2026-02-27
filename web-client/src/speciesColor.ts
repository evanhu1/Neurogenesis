const speciesColorCache = new Map<string, string>();

export function hashSpeciesId(speciesId: string): number {
  let hash = 2166136261;
  for (let i = 0; i < speciesId.length; i += 1) {
    hash ^= speciesId.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function colorForSpeciesId(speciesId: string): string {
  const cached = speciesColorCache.get(speciesId);
  if (cached) return cached;

  const hash = hashSpeciesId(speciesId);
  const hue = hash % 360;
  const saturation = 68 + ((hash >>> 9) % 22);
  const lightness = 50 + ((hash >>> 17) % 14);
  const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  speciesColorCache.set(speciesId, color);
  return color;
}
