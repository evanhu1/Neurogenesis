export function captureError(
  setErrorText: (message: string | null) => void,
  error: unknown,
  fallbackMessage: string,
) {
  setErrorText(error instanceof Error ? error.message : fallbackMessage);
}
