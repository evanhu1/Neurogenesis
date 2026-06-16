---
type: Index
title: Research bundle index
description: Navigational index of the autonomous-research OKF knowledge bundle.
tags: [okf, index, autoresearch]
---

# Research bundle

Autonomous-research knowledge base for NeuroGenesis. Format:
[OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md).
Schema + provenance rules: [/CONVENTIONS.md](/CONVENTIONS.md).

## Start here
- **[/STATE.md](/STATE.md)** — the planner's live dashboard. **Read first.**
- [/best-program.md](/best-program.md) — current champion (git ref + lineage).
- [/log.md](/log.md) — append-only iteration history.

## Collections
- `experiments/` — every research-agent run (code change + evaluation). Atomic provenance.
- `findings/` — validated conclusions (survived the verification gate).
- `directions/` — promising, not-yet-exhausted avenues ("untapped alpha").
- `mechanisms/` — durable laws of the system.
- `dead-ends/` — ruled-out avenues (never re-explore).
- `templates/` — concept skeletons (not knowledge; copy when authoring).

## Operating procedure
The planner that maintains this bundle is the `autoresearch` skill:
[/../.claude/skills/autoresearch/SKILL.md](/../.claude/skills/autoresearch/SKILL.md).
