import { useEffect, useRef, useState } from 'react';
import { WorldRenderer } from '../../../rendering/WorldRenderer';
import type { WorldOrganismState } from '../../../types';

type WorldCanvasProps = {
  /// Receives the renderer instance on mount (and `null` on unmount) so the
  /// owning hook can drive it (setWorld / applyDelta / focus) imperatively.
  onRenderer: (renderer: WorldRenderer | null) => void;
  onOrganismSelect: (organism: WorldOrganismState) => void;
};

export function WorldCanvas({ onRenderer, onOrganismSelect }: WorldCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<WorldRenderer | null>(null);
  const onRendererRef = useRef(onRenderer);
  onRendererRef.current = onRenderer;
  const onOrganismSelectRef = useRef(onOrganismSelect);
  onOrganismSelectRef.current = onOrganismSelect;
  const [showOrganisms, setShowOrganisms] = useState(true);
  const [showPlants, setShowPlants] = useState(true);

  // The renderer owns all stateful rendering (in-memory world, RAF loop,
  // viewport, input); this component creates it, exposes it to the owner, and
  // disposes it. Data (snapshots / deltas / focus) is driven by the owner via
  // the exposed instance.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const renderer = new WorldRenderer(canvas, {
      onPick: (organism) => {
        if (organism) onOrganismSelectRef.current(organism);
      },
    });
    rendererRef.current = renderer;
    onRendererRef.current(renderer);
    return () => {
      onRendererRef.current(null);
      renderer.dispose();
      rendererRef.current = null;
    };
  }, []);

  useEffect(() => {
    rendererRef.current?.setVisibility({ organisms: showOrganisms, plants: showPlants });
  }, [showOrganisms, showPlants]);

  return (
    <div className="relative h-full w-full">
      <canvas
        ref={canvasRef}
        id="world-canvas"
        width={900}
        height={900}
        className="block h-full w-full max-h-full max-w-full shrink-0 select-none bg-transparent"
      />

      <div className="absolute bottom-3 left-3 z-10 flex gap-1 rounded-full border border-line bg-panel/80 p-1 backdrop-blur-sm">
        <LayerToggle label="Organisms" checked={showOrganisms} onChange={setShowOrganisms} />
        <LayerToggle label="Plants" checked={showPlants} onChange={setShowPlants} />
      </div>

      <div className="pointer-events-none absolute bottom-3 right-3 z-10 hidden rounded-full border border-line bg-panel/60 px-3 py-1 text-[10px] text-ink/40 backdrop-blur-sm md:block">
        Scroll to zoom · Space + drag to pan · Click an organism to inspect
      </div>
    </div>
  );
}

function LayerToggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      aria-pressed={checked}
      className={`rounded-full px-2.5 py-1 text-[11px] font-medium transition ${
        checked ? 'bg-accent/20 text-accent' : 'text-ink/40 hover:text-ink/70'
      }`}
    >
      {label}
    </button>
  );
}
