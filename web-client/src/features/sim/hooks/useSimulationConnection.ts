import { useCallback, useEffect, useRef } from 'react';
import type { ApiServerEvent } from '../../../types';
import { connectSimulationWs, sendSimulationCommand } from '../api/simWsClient';
import { wsBase } from '../constants';

type UseSimulationConnectionArgs = {
  onServerEvent: (event: ApiServerEvent) => void;
  onSocketClose: () => void;
};

export function useSimulationConnection({
  onServerEvent,
  onSocketClose,
}: UseSimulationConnectionArgs) {
  const wsRef = useRef<WebSocket | null>(null);

  const connectWs = useCallback(
    (sessionId: string) => {
      wsRef.current?.close();
      let nextSocket: WebSocket;
      nextSocket = connectSimulationWs(
        wsBase,
        sessionId,
        onServerEvent,
        () => {
          onSocketClose();
          if (wsRef.current === nextSocket) {
            wsRef.current = null;
          }
        },
      );
      wsRef.current = nextSocket;
    },
    [onServerEvent, onSocketClose],
  );

  const sendCommand = useCallback((command: unknown): boolean => {
    return sendSimulationCommand(wsRef.current, command);
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return {
    connectWs,
    sendCommand,
  };
}
