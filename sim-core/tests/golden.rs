use sim_core::Simulation;
use sim_protocol::WorldConfig;

#[test]
fn golden_seed42_epoch3_snapshot_matches() {
    let fixture = std::fs::read_to_string("tests/fixtures/golden_seed42_epoch3.json")
        .expect("read golden fixture");

    let mut sim = Simulation::new(WorldConfig::default(), 42).expect("simulation init");
    sim.epoch_n(3);
    let actual = serde_json::to_string(&sim.snapshot()).expect("serialize snapshot");

    assert_eq!(actual, fixture.trim());
}
