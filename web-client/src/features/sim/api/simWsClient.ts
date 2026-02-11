import type { ServerEvent } from '../../../types';

export function connectSimulationWs(
  wsBase: string,
  sessionId: string,
  onEvent: (event: ServerEvent) => void,
  onClose: () => void,
): WebSocket {
  const socket = new WebSocket(`${wsBase}/v1/sessions/${sessionId}/stream`);

  socket.onmessage = (evt) => {
    try {
      const event = JSON.parse(String(evt.data)) as ServerEvent;
      onEvent(event);
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
  ws.send(JSON.stringify(command));
  return true;
}
