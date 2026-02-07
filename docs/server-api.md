# sim-server API (v1)

Base URL: `http://127.0.0.1:8080`

## Health
- `GET /health`

## Sessions
- `POST /v1/sessions`
- `GET /v1/sessions/{id}`
- `GET /v1/sessions/{id}/state`
- `POST /v1/sessions/{id}/step` body: `{ "count": <u32> }`
- `POST /v1/sessions/{id}/epoch` body: `{ "count": <u32> }`
- `POST /v1/sessions/{id}/scatter`
- `POST /v1/sessions/{id}/survivors/process`
- `POST /v1/sessions/{id}/reset` body: `{ "seed": <u64|null> }`
- `POST /v1/sessions/{id}/focus` body: `{ "organism_id": <u64> }`

## WebSocket
- `GET /v1/sessions/{id}/stream`

Client sends:
- `Start { ticks_per_second }`
- `Pause`
- `Step { count }`
- `Epoch { count }`
- `SetFocus { organism_id }`

Server emits:
- `StateSnapshot`
- `TickDelta`
- `EpochCompleted`
- `FocusBrain`
- `Metrics`
- `Error`

All HTTP and WS payloads are wrapped in:
```json
{ "protocol_version": 1, "payload": ... }
```
