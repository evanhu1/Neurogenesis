use sim_core::Simulation;
use sim_protocol::WorldConfig;

#[test]
fn golden_seed42_turn30_snapshot_matches() {
    let fixture =
        std::fs::read_to_string("tests/fixtures/golden_seed42_turn30.json").expect("read fixture");

    let mut sim = Simulation::new(WorldConfig::default(), 42).expect("simulation init");
    sim.step_n(30);
    let actual = serde_json::to_string(&sim.snapshot()).expect("serialize snapshot");

    assert_eq!(actual, fixture.trim());
}
