use crate::grid::hex_neighbor;
use types::{Occupant, OrganismId, OrganismState, SensoryReceptor, Symbol};

#[cfg(feature = "profiling")]
use crate::profiling::{self, BrainStage};
#[cfg(feature = "profiling")]
use std::time::Instant;

pub(super) fn begin_sensing_tick(organism: &mut OrganismState) {
    organism.energy_at_last_sensing = organism.energy;
    organism.energy_flow_last_tick = 0;
}

/// Encode the single cell immediately ahead of the organism.
pub(super) fn encode_sensory_symbol(
    organism: &mut OrganismState,
    world_width: i32,
    occupancy: &[Option<Occupant>],
) -> Symbol {
    #[cfg(feature = "profiling")]
    let scan_started = Instant::now();
    let (q, r) = hex_neighbor((organism.q, organism.r), organism.facing, world_width);
    let cell_index = r as usize * world_width as usize + q as usize;
    let symbol = occupant_symbol(occupancy[cell_index], organism.id);
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ScanAhead, scan_started.elapsed());

    #[cfg(feature = "profiling")]
    let encoding_started = Instant::now();
    for input in &mut organism.brain.sensory {
        input.neuron.activation = match input.receptor {
            SensoryReceptor::Symbol {
                symbol: input_symbol,
            } => f32::from(input_symbol == symbol),
        };
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::SensoryEncoding, encoding_started.elapsed());
    symbol
}

#[inline(always)]
fn occupant_symbol(occupant: Option<Occupant>, self_id: OrganismId) -> Symbol {
    match occupant {
        None => Symbol::A,
        Some(Occupant::Organism(id)) if id == self_id => Symbol::A,
        Some(Occupant::Wall) => Symbol::B,
        Some(Occupant::Organism(_)) => Symbol::C,
    }
}

#[cfg(test)]
mod tests {
    use super::occupant_symbol;
    use types::{Occupant, OrganismId, Symbol};

    #[test]
    fn vision_symbols_distinguish_empty_wall_and_other_organism() {
        let self_id = OrganismId(1);
        assert_eq!(occupant_symbol(None, self_id), Symbol::A);
        assert_eq!(occupant_symbol(Some(Occupant::Wall), self_id), Symbol::B);
        assert_eq!(
            occupant_symbol(Some(Occupant::Organism(OrganismId(2))), self_id),
            Symbol::C
        );
    }
}
