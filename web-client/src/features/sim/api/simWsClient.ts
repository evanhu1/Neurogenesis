import type { Envelope, ServerEvent } from '../../../types';
import { protocolVersion } from '../constants';

export function connectSimulationWs(
  wsBase: string,
  sessionId: string,
  onEvent: (event: ServerEvent) => void,
  onClose: () => void,
): WebSocket {
  const socket = new WebSocket(`${wsBase}/v1/sessions/${sessionId}/stream`);

  socket.onmessage = (evt) => {
    try {
      const envelope = JSON.parse(String(evt.data)) as Envelope<ServerEvent>;
      onEvent(envelope.payload);
    } catch (err) {
      console.error('ws parse error', err);
    }
  };

  socket.onclose = () => {
    onClose();
  };

  return socket;
}

export function sendSimulationCommand(ws: WebSocket | null, command: unknown): boolean {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  ws.send(JSON.stringify({ protocol_version: protocolVersion, payload: command }));
  return true;
}

