import { SESSION_ID_STORAGE_KEY } from './constants';

export function loadPersistedSessionId(): string | null {
  try {
    return localStorage.getItem(SESSION_ID_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function persistSessionId(sessionId: string) {
  try {
    localStorage.setItem(SESSION_ID_STORAGE_KEY, sessionId);
  } catch {
    // ignore storage failures; session still functions without persistence
  }
}

export function clearPersistedSessionId() {
  try {
    localStorage.removeItem(SESSION_ID_STORAGE_KEY);
  } catch {
    // ignore storage failures
  }
}

