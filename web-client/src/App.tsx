import { useCallback, useEffect, useState } from 'react';
import { api, type Champions, type PopulationStats, type RenderSnapshot } from './api';
import { WorldCanvas } from './WorldCanvas';
import { Inspector } from './Inspector';

// NeuroGenesis — new-substrate viewer. Polls the lean REST server for the world
// snapshot + population stats, renders the hex world, inspects an organism's
// CPPN genome + developed brain, and shows the Quality-Diversity champion pool.
export default function App() {
  const [snapshot, setSnapshot] = useState<RenderSnapshot | null>(null);
  const [stats, setStats] = useState<PopulationStats | null>(null);
  const [champions, setChampions] = useState<Champions | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [running, setRunning] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const poll = useCallback(async () => {
    try {
      const [snap, st] = await Promise.all([api.snapshot(), api.state()]);
      setSnapshot(snap);
      setStats(st);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    poll();
    const h = setInterval(poll, 250);
    return () => clearInterval(h);
  }, [poll]);

  useEffect(() => {
    const h = setInterval(() => api.champions().then(setChampions).catch(() => {}), 1500);
    return () => clearInterval(h);
  }, []);

  const control = async (cmd: 'play' | 'pause' | 'step') => {
    await api.control(cmd);
    if (cmd === 'play') setRunning(true);
    if (cmd === 'pause') setRunning(false);
  };

  return (
    <div style={page}>
      <header style={{ display: 'flex', alignItems: 'baseline', gap: 16 }}>
        <h1 style={{ margin: 0, fontSize: 20 }}>NeuroGenesis</h1>
        <span style={{ color: '#8b949e', fontSize: 13 }}>
          indirectly-encoded evolutionary substrate · hex world
        </span>
      </header>
      {error && <div style={{ color: '#f85149' }}>server unreachable: {error}</div>}

      <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
        <div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
            <button style={btn} onClick={() => control(running ? 'pause' : 'play')}>
              {running ? 'Pause' : 'Play'}
            </button>
            <button style={btn} onClick={() => control('step')}>
              Step
            </button>
          </div>
          <WorldCanvas snapshot={snapshot} onSelect={setSelected} selectedId={selected} />
          {stats && (
            <div style={statsGrid}>
              <Stat label="turn" value={stats.turn} />
              <Stat label="alive" value={stats.alive} />
              <Stat label="born" value={stats.total_ever} />
              <Stat label="max gen" value={stats.max_generation} />
              <Stat label="mean neurons" value={stats.mean_neurons.toFixed(1)} />
              <Stat label="mean edges" value={stats.mean_edges.toFixed(1)} />
              {stats.extinct_at !== null && <Stat label="EXTINCT @" value={stats.extinct_at} />}
            </div>
          )}
        </div>

        <div style={panel}>
          <Inspector id={selected} />
        </div>

        <div style={panel}>
          <h3 style={{ marginTop: 0 }}>QD champions</h3>
          {champions ? (
            <>
              <div style={{ color: '#8b949e', fontSize: 13 }}>
                schema v{champions.schema_version} · coverage {champions.coverage} · QD-score{' '}
                {champions.qd_score.toFixed(1)}
              </div>
              <ul style={{ paddingLeft: 16, fontSize: 13 }}>
                {champions.entries.slice(0, 20).map((c, i) => (
                  <li key={i}>
                    quality {c.quality.toFixed(2)} · niche [
                    {c.descriptor.values.map((v) => v.toFixed(2)).join(', ')}]
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <div style={{ color: '#8b949e' }}>no champions saved yet</div>
          )}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <div style={{ color: '#8b949e', fontSize: 11 }}>{label}</div>
      <div style={{ fontSize: 16 }}>{value}</div>
    </div>
  );
}

const page: React.CSSProperties = {
  fontFamily: 'system-ui, sans-serif',
  background: '#010409',
  color: '#c9d1d9',
  minHeight: '100vh',
  padding: 20,
  display: 'flex',
  flexDirection: 'column',
  gap: 16,
};
const panel: React.CSSProperties = {
  background: '#0d1117',
  border: '1px solid #30363d',
  borderRadius: 8,
  padding: 16,
  width: 320,
};
const btn: React.CSSProperties = {
  background: '#21262d',
  color: '#c9d1d9',
  border: '1px solid #30363d',
  borderRadius: 6,
  padding: '6px 14px',
  cursor: 'pointer',
};
const statsGrid: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(4, 1fr)',
  gap: 10,
  marginTop: 10,
  maxWidth: 560,
};
