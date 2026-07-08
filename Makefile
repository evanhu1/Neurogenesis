.PHONY: fmt lint test check examples evaluate web-install web-build start

fmt:
	cargo fmt --all

lint:
	cargo clippy --workspace --all-targets -- -D warnings

check:
	cargo check --workspace

test:
	cargo test --workspace

# Deterministic proof-of-life examples (each asserts a live, reproducing
# population and byte-identical results across identical-seed runs).
examples:
	cargo run -p sim-substrate --example headless --release
	cargo run -p sim-hexworld  --example headless --release
	cargo run -p sim-hexworld  --example simsmoke --release
	cargo run -p sim-toyenv    --example headless --release

evaluate:
	cargo run -p sim-evaluation --release -- $(ARGS)

web-install:
	cd web-client && npm install

web-build:
	cd web-client && npm run build

start:
	cargo run -p sim-server --release
