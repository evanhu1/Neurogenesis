import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { findOrganism, unwrapId } from '../../../protocol';
import type {
  FocusBrainData,
  OrganismState,
  SessionMetadata,
  WorldOrganismState,
  WorldSnapshot,
} from '../../../types';
import type { SimRequestFn } from '../api/simHttpClient';

export const NO_FOCUS_TURN = -1;

function mergeFocusedOrganismWithWorld(
  focused: OrganismState,
  worldOrganism: WorldOrganismState,
): OrganismState {
  return {
    ...focused,
    species_id: worldOrganism.species_id,
    q: worldOrganism.q,
    r: worldOrganism.r,
    generation: worldOrganism.generation,
    age_turns: worldOrganism.age_turns,
    facing: worldOrganism.facing,
  };
}

type UseSimulationFocusArgs = {
  snapshot: WorldSnapshot | null;
  session: SessionMetadata | null;
  request: SimRequestFn;
  setErrorText: (message: string | null) => void;
};

export function useSimulationFocus({
  snapshot,
  session,
  request,
  setErrorText,
}: UseSimulationFocusArgs) {
  const [focusedOrganismId, setFocusedOrganismId] = useState<number | null>(null);
  const [focusedOrganismDetails, setFocusedOrganismDetails] = useState<OrganismState | null>(null);
  const [focusedOrganismTurn, setFocusedOrganismTurn] = useState<number>(NO_FOCUS_TURN);
  const [activeActionNeuronId, setActiveActionNeuronId] = useState<number | null>(null);
  const focusedOrganismIdRef = useRef<number | null>(null);
  const latestFocusedTurnRef = useRef<number>(NO_FOCUS_TURN);
  const nextFocusPollAtMsRef = useRef(0);

  const setFocusedOrganismIdTracked = useCallback((organismId: number | null, resetPoll = false) => {
    const changed = focusedOrganismIdRef.current !== organismId;
    focusedOrganismIdRef.current = organismId;
    if (changed) {
      latestFocusedTurnRef.current = NO_FOCUS_TURN;
    }
    if (resetPoll) {
      nextFocusPollAtMsRef.current = 0;
    }
    setFocusedOrganismId(organismId);
  }, []);

  const resetFocusState = useCallback(
    (resetPoll = true) => {
      setFocusedOrganismIdTracked(null, resetPoll);
      setFocusedOrganismDetails(null);
      setFocusedOrganismTurn(NO_FOCUS_TURN);
      setActiveActionNeuronId(null);
    },
    [setFocusedOrganismIdTracked],
  );

  const focusOrganism = useCallback(
    (organism: WorldOrganismState) => {
      const organismId = unwrapId(organism.id);
      const isSameFocus = focusedOrganismIdRef.current === organismId;
      setFocusedOrganismIdTracked(organismId, true);
      if (!isSameFocus) {
        setFocusedOrganismDetails(null);
        setFocusedOrganismTurn(NO_FOCUS_TURN);
        setActiveActionNeuronId(null);
      }
      if (!session) return;
      void request(`/v1/sessions/${session.id}/focus`, 'POST', {
        organism_id: organismId,
      }).catch((err) => {
        setErrorText(err instanceof Error ? err.message : 'Failed to focus organism');
      });
    },
    [request, session, setErrorText, setFocusedOrganismIdTracked],
  );

  const defocusOrganism = useCallback(() => {
    resetFocusState(true);
  }, [resetFocusState]);

  const handleFocusBrain = useCallback((data: FocusBrainData, latestSnapshotTurn: number) => {
    const { turn, organism, active_action_neuron_id } = data;
    const organismId = unwrapId(organism.id);
    const trackedFocusedId = focusedOrganismIdRef.current;
    if (trackedFocusedId === null || trackedFocusedId !== organismId) {
      return;
    }
    const minimumAcceptedTurn = Math.max(latestSnapshotTurn, latestFocusedTurnRef.current);
    if (turn < minimumAcceptedTurn) {
      return;
    }
    latestFocusedTurnRef.current = turn;
    setFocusedOrganismDetails(organism);
    setFocusedOrganismTurn(turn);
    setActiveActionNeuronId(active_action_neuron_id);
  }, []);

  const focusedWorldOrganism = useMemo(() => {
    if (!snapshot || focusedOrganismId === null) {
      return null;
    }
    return findOrganism(snapshot, focusedOrganismId);
  }, [focusedOrganismId, snapshot]);

  const focusedOrganism = useMemo(() => {
    if (!focusedOrganismDetails || focusedOrganismId === null) {
      return null;
    }
    if (unwrapId(focusedOrganismDetails.id) !== focusedOrganismId) {
      return null;
    }
    if (snapshot && focusedWorldOrganism && snapshot.turn >= focusedOrganismTurn) {
      return mergeFocusedOrganismWithWorld(focusedOrganismDetails, focusedWorldOrganism);
    }
    return focusedOrganismDetails;
  }, [
    focusedOrganismDetails,
    focusedOrganismId,
    focusedOrganismTurn,
    focusedWorldOrganism,
    snapshot,
  ]);

  useEffect(() => {
    if (!snapshot || focusedOrganismId === null || focusedWorldOrganism) {
      return;
    }
    resetFocusState(true);
  }, [focusedOrganismId, focusedWorldOrganism, resetFocusState, snapshot]);

  return {
    focusedOrganismId,
    focusedOrganism,
    activeActionNeuronId,
    focusedOrganismIdRef,
    latestFocusedTurnRef,
    nextFocusPollAtMsRef,
    handleFocusBrain,
    focusOrganism,
    defocusOrganism,
    resetFocusState,
  };
}
