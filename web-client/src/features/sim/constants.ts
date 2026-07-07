export const apiBase = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8080';
export const wsBase = apiBase.replace('http://', 'ws://').replace('https://', 'wss://');

export const WORLD_NAME_STORAGE_KEY = 'neurogenesis.world_name';
export const SPEED_LEVELS = [2, 8, 30, 60, 120] as const;
