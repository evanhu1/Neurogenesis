import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import { pickOrganismAtCanvasPoint, renderBrain, renderWorld } from './canvas';
import { applyTickDelta, findBrain, unwrapId } from './protocol';
import type {
  BrainState,
  CreateSessionResponse,
  Envelope,
  MetricsSnapshot,
  OrganismState,
  ServerEvent,
  SessionMetadata,
  TickDelta,
  WorldSnapshot,
} from './types';
import { DEFAULT_CONFIG } from './types';

const apiBase = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8080';
const wsBase = apiBase.replace('http://', 'ws://').replace('https://', 'wss://');
const protocolVersion = 2;

export default function App() {
  const [session, setSession] = useState<SessionMetadata | null>(null);
  const [snapshot, setSnapshot] = useState<WorldSnapshot | null>(null);
  const [focusedOrganismId, setFocusedOrganismId] = useState<number | null>(null);
  const [focusedBrain, setFocusedBrain] = useState<BrainState | null>(null);
  const [focusMetaText, setFocusMetaText] = useState('Click an organism');
  const [errorText, setErrorText] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const preferEnvelopeHttpBodyRef = useRef(false);
  const sessionRef = useRef<SessionMetadata | null>(null);
  const snapshotRef = useRef<WorldSnapshot | null>(null);
  const focusedOrganismIdRef = useRef<number | null>(null);
  const worldCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const brainCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    sessionRef.current = session;
  }, [session]);

  useEffect(() => {
    snapshotRef.current = snapshot;
  }, [snapshot]);

  useEffect(() => {
    focusedOrganismIdRef.current = focusedOrganismId;
  }, [focusedOrganismId]);

  const request = useCallback(
    async <T,>(path: string, method: string, body?: unknown): Promise<T> => {
      const send = async (wrapBodyInEnvelope: boolean) => {
        const response = await fetch(`${apiBase}${path}`, {
          method,
          headers: {
            'Content-Type': 'application/json',
          },
          body:
            body === undefined
              ? undefined
              : JSON.stringify(
                  wrapBodyInEnvelope
                    ? { protocol_version: protocolVersion, payload: body }
                    : body,
                ),
        });

        const text = await response.text();
        let parsed: unknown = null;
        if (text) {
          try {
            parsed = JSON.parse(text);
          } catch {
            parsed = null;
          }
        }
        return { response, text, parsed };
      };

      let { response, text, parsed } = await send(preferEnvelopeHttpBodyRef.current);
      const startedWrapped = preferEnvelopeHttpBodyRef.current;

      if (
        !startedWrapped &&
        response.status === 422 &&
        body !== undefined &&
        /protocol_version|payload/i.test(text)
      ) {
        ({ response, text, parsed } = await send(true));
        if (response.ok) {
          preferEnvelopeHttpBodyRef.current = true;
        }
      }

      const parsedRecord =
        parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : null;
      const payload = parsedRecord?.payload;

      if (!response.ok) {
        let message: string | null = null;
        if (payload && typeof payload === 'object') {
          const payloadRecord = payload as Record<string, unknown>;
          if (typeof payloadRecord.message === 'string') {
            message = payloadRecord.message;
          }
        }
        if (!message && typeof parsedRecord?.message === 'string') {
          message = parsedRecord.message;
        }
        if (!message && text.trim()) {
          message = text.trim();
        }
        throw new Error(message ?? `request failed (${response.status})`);
      }

      if (payload !== undefined) {
        return payload as T;
      }
      if (parsedRecord) {
        return parsedRecord as unknown as T;
      }
      throw new Error('request succeeded but response was not JSON');
    },
    [],
  );

  const handleServerEvent = useCallback(
    (event: ServerEvent) => {
      switch (event.type) {
        case 'StateSnapshot': {
          const nextSnapshot = event.data as WorldSnapshot;
          setSnapshot(nextSnapshot);
          const currentFocusedId = focusedOrganismIdRef.current;
          if (currentFocusedId !== null) {
            setFocusedBrain(findBrain(nextSnapshot, currentFocusedId));
          }
          break;
        }
        case 'TickDelta': {
          const delta = event.data as TickDelta;
          setSnapshot((prev) => (prev ? applyTickDelta(prev, delta) : prev));
          break;
        }
        case 'FocusBrain': {
          const organism = event.data as OrganismState;
          const organismId = unwrapId(organism.id);
          setFocusedOrganismId(organismId);
          setFocusedBrain(organism.brain);
          setFocusMetaText(`focused organism: ${organismId}`);
          break;
        }
        case 'Metrics': {
          setSnapshot((prev) =>
            prev ? { ...prev, metrics: event.data as MetricsSnapshot } : prev,
          );
          break;
        }
        case 'Error': {
          const message =
            typeof event.data === 'string' ? event.data : 'Simulation server reported an error';
          setErrorText(message);
          break;
        }
        default:
          break;
      }
    },
    [],
  );

  const connectWs = useCallback(
    (sessionId: string) => {
      wsRef.current?.close();
      const nextWs = new WebSocket(`${wsBase}/v1/sessions/${sessionId}/stream`);

      nextWs.onmessage = (evt) => {
        try {
          const envelope = JSON.parse(String(evt.data)) as Envelope<ServerEvent>;
          handleServerEvent(envelope.payload);
        } catch (err) {
          console.error('ws parse error', err);
        }
      };

      nextWs.onclose = () => {
        wsRef.current = null;
      };

      wsRef.current = nextWs;
    },
    [handleServerEvent],
  );

  const createSession = useCallback(async () => {
    try {
      setErrorText(null);
      const payload = await request<CreateSessionResponse>('/v1/sessions', 'POST', {
        config: DEFAULT_CONFIG,
        seed: Math.floor(Date.now() / 1000),
      });

      setSession(payload.metadata);
      setSnapshot(payload.snapshot);
      setFocusedOrganismId(null);
      setFocusedBrain(null);
      setFocusMetaText('Click an organism');

      connectWs(payload.metadata.id);
    } catch (err) {
      setErrorText(err instanceof Error ? err.message : 'Failed to create session');
    }
  }, [connectWs, request]);

  const sendCommand = useCallback((command: unknown) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ protocol_version: 2, payload: command }));
  }, []);

  const onReset = useCallback(() => {
    if (!session) return;
    void request<WorldSnapshot>(`/v1/sessions/${session.id}/reset`, 'POST', { seed: null })
      .then((nextSnapshot) => {
        setSnapshot(nextSnapshot);
        setFocusedOrganismId(null);
        setFocusedBrain(null);
        setFocusMetaText('Click an organism');
      })
      .catch((err) => {
        setErrorText(err instanceof Error ? err.message : 'Failed to reset session');
      });
  }, [request, session]);

  const onWorldCanvasClick = useCallback(
    (evt: MouseEvent<HTMLCanvasElement>) => {
      if (!snapshot || !session) return;
      const canvas = evt.currentTarget;
      const rect = canvas.getBoundingClientRect();
      const xPx = ((evt.clientX - rect.left) / rect.width) * canvas.width;
      const yPx = ((evt.clientY - rect.top) / rect.height) * canvas.height;

      const picked = pickOrganismAtCanvasPoint(snapshot, canvas.width, canvas.height, xPx, yPx);
      if (!picked) return;

      const selectedId = unwrapId(picked.organism.id);
      setFocusedOrganismId(selectedId);
      setFocusedBrain(picked.organism.brain);
      setFocusMetaText(`focused organism: ${selectedId} at (${picked.gridQ}, ${picked.gridR})`);

      void request(`/v1/sessions/${session.id}/focus`, 'POST', {
        organism_id: selectedId,
      }).catch((err) => {
        setErrorText(err instanceof Error ? err.message : 'Failed to focus organism');
      });
    },
    [request, session, snapshot],
  );

  useEffect(() => {
    let frameId = 0;
    const draw = () => {
      const canvas = worldCanvasRef.current;
      if (canvas) {
        const context = canvas.getContext('2d');
        if (context) {
          renderWorld(context, canvas, snapshotRef.current, focusedOrganismIdRef.current);
        }
      }
      frameId = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(frameId);
  }, []);

  useEffect(() => {
    const canvas = brainCanvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    renderBrain(context, canvas, focusedBrain);
  }, [focusedBrain]);

  useEffect(() => {
    void createSession();
  }, [createSession]);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const sessionMeta = useMemo(() => {
    if (!session) return 'No session';
    return `session=${session.id}\ncreated=${new Date(session.created_at_unix_ms).toISOString()}`;
  }, [session]);

  const metricsText = useMemo(() => {
    if (!snapshot) return 'No metrics';
    return [
      `turn=${snapshot.turn}`,
      `organisms=${snapshot.metrics.organisms}`,
      `meals_last_turn=${snapshot.metrics.meals_last_turn}`,
      `starvations_last_turn=${snapshot.metrics.starvations_last_turn}`,
      `births_last_turn=${snapshot.metrics.births_last_turn}`,
      `synapse_ops_last_turn=${snapshot.metrics.synapse_ops_last_turn}`,
      `actions_last_turn=${snapshot.metrics.actions_applied_last_turn}`,
    ].join('\n');
  }, [snapshot]);

  return (
    <div className="min-h-screen bg-page px-4 py-4 text-ink sm:px-6 lg:px-8">
      <div className="mx-auto grid max-w-[1720px] gap-4 xl:grid-cols-[320px_minmax(480px,1fr)_460px]">
        <aside className="rounded-2xl border border-accent/15 bg-panel/95 p-4 shadow-panel">
          <h1 className="text-2xl font-semibold tracking-tight">NeuroGenesis</h1>
          <pre className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-100/80 p-3 font-mono text-xs">
            {sessionMeta}
          </pre>

          <div className="mt-3 flex flex-wrap gap-2">
            <ControlButton label="Create Session" onClick={() => void createSession()} />
            <ControlButton label="Reset" onClick={onReset} />
            <ControlButton
              label="Start"
              onClick={() => sendCommand({ type: 'Start', data: { ticks_per_second: 10 } })}
            />
            <ControlButton label="Pause" onClick={() => sendCommand({ type: 'Pause' })} />
            <ControlButton label="Step 1" onClick={() => sendCommand({ type: 'Step', data: { count: 1 } })} />
            <ControlButton label="Step 10" onClick={() => sendCommand({ type: 'Step', data: { count: 10 } })} />
            <ControlButton label="Step 100" onClick={() => sendCommand({ type: 'Step', data: { count: 100 } })} />
          </div>

          <pre className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-100/80 p-3 font-mono text-xs">
            {metricsText}
          </pre>

          {errorText ? (
            <div className="mt-3 rounded-xl border border-rose-300 bg-rose-50 px-3 py-2 font-mono text-xs text-rose-700">
              {errorText}
            </div>
          ) : null}
        </aside>

        <main className="flex items-center justify-center rounded-2xl border border-accent/15 bg-panel/70 p-3 shadow-panel">
          <canvas
            ref={worldCanvasRef}
            onClick={onWorldCanvasClick}
            id="world-canvas"
            width={900}
            height={900}
            className="w-full max-w-[1000px] rounded-xl border border-accent/20 bg-white"
          />
        </main>

        <aside className="rounded-2xl border border-accent/15 bg-panel/95 p-4 shadow-panel">
          <h2 className="text-xl font-semibold tracking-tight">Brain Inspector</h2>
          <div className="mt-2 rounded-xl bg-slate-100/80 p-3 font-mono text-xs">{focusMetaText}</div>
          <canvas
            ref={brainCanvasRef}
            id="brain-canvas"
            width={420}
            height={560}
            className="mt-3 w-full rounded-xl border border-accent/20 bg-white"
          />
        </aside>
      </div>
    </div>
  );
}

type ControlButtonProps = {
  label: string;
  onClick: () => void;
};

function ControlButton({ label, onClick }: ControlButtonProps) {
  return (
    <button
      onClick={onClick}
      className="rounded-lg bg-accent px-3 py-2 text-sm font-semibold text-white transition hover:brightness-110"
    >
      {label}
    </button>
  );
}
