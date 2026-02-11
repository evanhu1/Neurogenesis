export type SimRequestFn = <T>(path: string, method: string, body?: unknown) => Promise<T>;

export function createSimHttpClient(baseUrl: string): SimRequestFn {
  return async function request<T>(path: string, method: string, body?: unknown): Promise<T> {
    const response = await fetch(`${baseUrl}${path}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
      body: body === undefined ? undefined : JSON.stringify(body),
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

    if (!response.ok) {
      let message: string | null = null;
      const parsedRecord =
        parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : null;
      if (parsedRecord && typeof parsedRecord.message === 'string') {
        message = parsedRecord.message;
      }
      if (!message && text.trim()) {
        message = text.trim();
      }
      throw new Error(message ?? `request failed (${response.status})`);
    }

    if (parsed !== null) {
      return parsed as T;
    }
    throw new Error('request succeeded but response was not JSON');
  };
}
