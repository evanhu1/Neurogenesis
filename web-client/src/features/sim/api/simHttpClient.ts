import { protocolVersion } from '../constants';

export type SimRequestFn = <T>(path: string, method: string, body?: unknown) => Promise<T>;

export function createSimHttpClient(baseUrl: string): SimRequestFn {
  let preferEnvelopeHttpBody = false;

  return async function request<T>(path: string, method: string, body?: unknown): Promise<T> {
    const send = async (wrapBodyInEnvelope: boolean) => {
      const response = await fetch(`${baseUrl}${path}`, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body:
          body === undefined
            ? undefined
            : JSON.stringify(
                wrapBodyInEnvelope
                  ? { protocol_version: protocolVersion, payload: body }
                  : body,
              ),
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
      return { response, text, parsed };
    };

    let { response, text, parsed } = await send(preferEnvelopeHttpBody);
    const startedWrapped = preferEnvelopeHttpBody;

    if (
      !startedWrapped &&
      response.status === 422 &&
      body !== undefined &&
      /protocol_version|payload/i.test(text)
    ) {
      ({ response, text, parsed } = await send(true));
      if (response.ok) {
        preferEnvelopeHttpBody = true;
      }
    }

    const parsedRecord =
      parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : null;
    const payload = parsedRecord?.payload;

    if (!response.ok) {
      let message: string | null = null;
      if (payload && typeof payload === 'object') {
        const payloadRecord = payload as Record<string, unknown>;
        if (typeof payloadRecord.message === 'string') {
          message = payloadRecord.message;
        }
      }
      if (!message && typeof parsedRecord?.message === 'string') {
        message = parsedRecord.message;
      }
      if (!message && text.trim()) {
        message = text.trim();
      }
      throw new Error(message ?? `request failed (${response.status})`);
    }

    if (payload !== undefined) {
      return payload as T;
    }
    if (parsedRecord) {
      return parsedRecord as unknown as T;
    }
    throw new Error('request succeeded but response was not JSON');
  };
}

