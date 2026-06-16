# sim-cli — design (superseded)

> **This original v1 design note is superseded and no longer accurate.** It
> described the first foraging-debug prototype (the `forage` command,
> `p_fwd_food`, etc.), all of which has been replaced.
>
> - **Usage reference:** `docs/sim-cli.md`
> - **Current design / rationale:** `SPEC.md`
> - **Live source of truth:** run `cargo run -p sim-cli` and type `help`.
>
> sim-cli v2 is a general research cockpit: a live metric `Recorder` built on the
> shared `sim-metrics` crate (so live `pillars` match the eval byte-for-byte),
> text/JSON output, and dashboards (`eco`/`lineage`/`genome`/`timeseries`/`watch`)
> plus per-organism inspection (`find`/`brain`/`decide`). The `forage` command
> described in the old draft was removed — use `pillars` instead.
