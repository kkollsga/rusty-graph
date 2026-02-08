# rusty_graph development tasks
# All targets handle CONDA_PREFIX and .venv activation automatically.

SHELL := /bin/bash
ACTIVATE := unset CONDA_PREFIX && source .venv/bin/activate

.PHONY: dev test test-rust test-py bench check clean fmt clippy

## Build and install the package into the local .venv
dev:
	$(ACTIVATE) && maturin develop

## Run all tests (Rust + Python)
test: test-rust test-py

## Run Rust unit tests only
test-rust:
	$(ACTIVATE) && cargo test

## Run Python tests only (excludes benchmarks)
test-py:
	$(ACTIVATE) && pytest tests/ -v

## Run performance benchmarks
bench:
	$(ACTIVATE) && pytest tests/benchmarks/ -v -m benchmark -s

## Fast compilation check (no codegen)
check:
	cargo check

## Format Rust code
fmt:
	cargo fmt

## Run clippy lints
clippy:
	cargo clippy -- -D warnings

## Remove build artifacts
clean:
	cargo clean
