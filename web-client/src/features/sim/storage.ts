import { WORLD_NAME_STORAGE_KEY } from './constants';

export function loadPersistedWorldName(): string | null {
  try {
    return localStorage.getItem(WORLD_NAME_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function persistWorldName(name: string) {
  try {
    localStorage.setItem(WORLD_NAME_STORAGE_KEY, name);
  } catch {
    // ignore storage failures; the world still functions without persistence
  }
}

export function clearPersistedWorldName() {
  try {
    localStorage.removeItem(WORLD_NAME_STORAGE_KEY);
  } catch {
    // ignore storage failures
  }
}
