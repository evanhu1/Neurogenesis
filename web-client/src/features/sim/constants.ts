export const apiBase = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8080';
export const wsBase = apiBase.replace('http://', 'ws://').replace('https://', 'wss://');
export const protocolVersion = 5;

export const SESSION_ID_STORAGE_KEY = 'neurogenesis.session_id';
export const SPEED_LEVELS = [2, 10, 30] as const;
