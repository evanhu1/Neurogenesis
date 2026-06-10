import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { findOrganism } from '../../../protocol';
import type {
  FocusBrainData,
  OrganismState,
  SessionMetadata,
  WorldOrganismState,
  WorldSnapshot,
} from '../../../types';
import type { SimRequestFn } from '../api/simHttpClient';
import { captureError } from './captureError';

export const NO_FOCUS_TURN = -1;

// Focused-brain details arrive on a slower channel than world snapshots; overlay
// the world's authoritative position/lifecycle fields so the inspector never lags.
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

  const resetFocusState = useCallback(() => {
    focusedOrganismIdRef.current = null;
    latestFocusedTurnRef.current = NO_FOCUS_TURN;
    nextFocusPollAtMsRef.current = 0;
    setFocusedOrganismId(null);
    setFocusedOrganismDetails(null);
    setFocusedOrganismTurn(NO_FOCUS_TURN);
    setActiveActionNeuronId(null);
  }, []);

  const focusOrganism = useCallback(
    (organism: WorldOrganismState) => {
      const organismId = organism.id;
      const isSameFocus = focusedOrganismIdRef.current === organismId;
      focusedOrganismIdRef.current = organismId;
      nextFocusPollAtMsRef.current = 0;
      setFocusedOrganismId(organismId);
      if (!isSameFocus) {
        latestFocusedTurnRef.current = NO_FOCUS_TURN;
        setFocusedOrganismDetails(null);
        setFocusedOrganismTurn(NO_FOCUS_TURN);
        setActiveActionNeuronId(null);
      }
      if (!session) return;
      void request(`/v1/sessions/${session.id}/focus`, 'POST', {
        organism_id: organismId,
      }).catch((err) => {
        captureError(setErrorText, err, 'Failed to focus organism');
      });
    },
    [request, session, setErrorText],
  );

  const handleFocusBrain = useCallback((data: FocusBrainData, latestSnapshotTurn: number) => {
    const { turn, organism, active_action_neuron_id } = data;
    if (focusedOrganismIdRef.current !== organism.id) {
      return;
    }
    if (turn < Math.max(latestSnapshotTurn, latestFocusedTurnRef.current)) {
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
    if (!focusedOrganismDetails || focusedOrganismDetails.id !== focusedOrganismId) {
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

  // Defocus when the focused organism disappears from the world (death/removal).
  useEffect(() => {
    if (!snapshot || focusedOrganismId === null || focusedWorldOrganism) {
      return;
    }
    resetFocusState();
  }, [focusedOrganismId, focusedWorldOrganism, resetFocusState, snapshot]);

  return {
    focusedOrganismId,
    focusedOrganism,
    activeActionNeuronId,
    focusedOrganismIdRef,
    nextFocusPollAtMsRef,
    handleFocusBrain,
    focusOrganism,
    defocusOrganism: resetFocusState,
    resetFocusState,
  };
}
