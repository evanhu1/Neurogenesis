.PHONY: fmt lint test check web-install web-build

fmt:
	cargo fmt --all

lint:
	cargo clippy --workspace --all-targets -- -D warnings

check:
	cargo check --workspace

test:
	cargo test --workspace

web-install:
	cd web-client && npm install

web-build:
	cd web-client && npm run build
