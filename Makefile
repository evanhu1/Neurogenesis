.PHONY: fmt lint test check perf-test evaluate web-install web-build start

fmt:
	cargo fmt --all

lint:
	cargo clippy --workspace --all-targets -- -D warnings

check:
	cargo check --workspace

test:
	cargo test --workspace

perf-test:
	cargo test -p sim-core --release performance_regression -- --ignored --nocapture

evaluate:
	cargo run -p sim-evaluation --release -- $(ARGS)

web-install:
	cd web-client && npm install

web-build:
	cd web-client && npm run build

start:
	cargo run -p sim-server --release
