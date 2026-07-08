# sim-cli — design (superseded)

> **This design note is fully superseded and no longer accurate.** It described
> early `sim-cli` prototypes (the `forage` foraging-debug command, then the
> `sim-metrics`-backed `pillars`/dashboards "research cockpit"), none of which
> exist after the substrate redesign.
>
> - **Current usage reference:** `docs/sim-cli.md` (the live command set).
> - The current CLI is a lean, stateless world-as-file tool over a bincode
>   `HexSim`; there is no metric sidecar, `pillars`, `sweep`, or `query`.
