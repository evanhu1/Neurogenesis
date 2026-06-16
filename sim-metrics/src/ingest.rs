//! Drive a [`Ledger`] from live simulation tick data. This is the exact
//! per-tick accumulation sequence shared by the eval orchestration (which
//! persists the resulting rows to Parquet) and the CLI recorder (which keeps
//! them in memory) — so both observe byte-identical metrics.

use crate::ledger::Ledger;
use crate::schema::OrganismLifetimeRow;
use sim_types::{ActionRecord, EntityId, OrganismState, TickDelta};

/// Register the initial founder population with the ledger before the tick
/// loop. Founders are born at tick 0, which marks them `InitialFounder`.
pub fn register_founders(ledger: &mut Ledger, organisms: &[OrganismState]) {
    for organism in organisms {
        ledger.birth(organism.id, 0);
    }
}

/// Feed one tick's emissions into the ledger and return the lifetime rows for
/// organisms that died this tick, in `delta.removed_positions` order, for the
/// caller to persist or collect.
///
/// The call order — actions, then reproductions, then births, then deaths —
/// is load-bearing: reproductions register child→parent edges that the
/// following births consume to classify descendants. `action_records` is the
/// per-organism slice from `Simulation::action_records()` (index-aligned to
/// `organisms()`); `delta` is the value returned by `Simulation::tick()`.
pub fn ingest_tick(
    ledger: &mut Ledger,
    tick: u64,
    delta: &TickDelta,
    action_records: &[Option<ActionRecord>],
) -> Vec<OrganismLifetimeRow> {
    for record in action_records.iter().flatten() {
        ledger.record_action(record);
    }
    for event in &delta.reproduction_events {
        ledger.record_reproduction(event);
    }
    for spawned in &delta.spawned {
        ledger.birth(spawned.id, tick);
    }
    let mut deaths = Vec::new();
    for removed in &delta.removed_positions {
        if let EntityId::Organism(id) = removed.entity_id {
            if let Some(row) = ledger.death(id, tick) {
                deaths.push(row);
            }
        }
    }
    deaths
}
