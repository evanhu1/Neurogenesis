use sim_core::Simulation;
use sim_protocol::WorldConfig;

#[test]
fn golden_seed42_turn30_snapshot_matches() {
    let fixture_path = "tests/fixtures/golden_seed42_turn30.json";

    let mut sim = Simulation::new(WorldConfig::default(), 42).expect("simulation init");
    sim.step_n(30);
    let actual = serde_json::to_string(&sim.snapshot()).expect("serialize snapshot");

    if std::env::var_os("BLESS").is_some() {
        std::fs::write(fixture_path, format!("{actual}\n")).expect("write blessed fixture");
    }

    let fixture = std::fs::read_to_string(fixture_path).expect("read fixture");
    assert_eq!(actual, fixture.trim());
}
