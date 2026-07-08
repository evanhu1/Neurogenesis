import { useEffect, useState } from 'react';
import { api, type OrganismDetail } from './api';
import { BrainView } from './BrainView';

function Row({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
      <span style={{ color: '#8b949e' }}>{label}</span>
      <span>{value}</span>
    </div>
  );
}

export function Inspector({ id }: { id: number | null }) {
  const [detail, setDetail] = useState<OrganismDetail | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setSaved(false);
    if (id === null) {
      setDetail(null);
      return;
    }
    let live = true;
    api
      .organism(id)
      .then((d) => live && setDetail(d))
      .catch(() => live && setDetail(null));
    return () => {
      live = false;
    };
  }, [id]);

  if (id === null) return <div style={{ color: '#8b949e' }}>Click an organism to inspect it.</div>;
  if (!detail) return <div style={{ color: '#8b949e' }}>Organism {id} — no longer alive.</div>;

  const g = detail.genome;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <h3 style={{ margin: 0 }}>Organism #{detail.id}</h3>
      <Row label="energy" value={detail.energy.toFixed(1)} />
      <Row label="health" value={detail.health.toFixed(1)} />
      <Row label="age" value={detail.age_turns} />
      <Row label="generation" value={detail.generation} />
      <Row label="CPPN nodes / conns" value={`${g.cppn.nodes.length} / ${g.cppn.conns.length}`} />
      <Row
        label="brain in/hidden/out"
        value={`${detail.brain.input_count} / ${detail.brain.hidden_count} / ${detail.brain.output_count}`}
      />
      <Row label="brain edges" value={detail.brain.edges.length} />
      <div style={{ marginTop: 6, color: '#8b949e', fontSize: 12 }}>Phenotype brain</div>
      <BrainView brain={detail.brain} />
      <button
        onClick={() => api.saveChampion(detail.id).then(() => setSaved(true))}
        style={btnStyle}
      >
        {saved ? '★ saved to QD archive' : 'Save to QD champion archive'}
      </button>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  background: '#238636',
  color: 'white',
  border: 'none',
  borderRadius: 6,
  padding: '6px 10px',
  cursor: 'pointer',
  fontSize: 13,
};
