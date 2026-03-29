# kglite development tasks
# All targets handle CONDA_PREFIX and .venv activation automatically.

SHELL := /bin/bash
ACTIVATE := unset CONDA_PREFIX && source .venv/bin/activate

.PHONY: dev test test-rust test-py bench bench-save bench-compare check clean fmt fmt-py clippy lint lint-py cov stubtest

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

## Save benchmark baseline for comparison
bench-save:
	$(ACTIVATE) && pytest tests/benchmarks/test_bench_core.py -m benchmark --benchmark-save=baseline

## Compare current performance against saved baseline
bench-compare:
	$(ACTIVATE) && pytest tests/benchmarks/test_bench_core.py -m benchmark --benchmark-compare

## Run bug-path performance benchmarks (pre/post bugfix baseline)
bench-bugs:
	$(ACTIVATE) && python bench/benchmark_bugs.py

## Fast compilation check (no codegen)
check:
	cargo check

## Format Rust code
fmt:
	cargo fmt

## Format Python code
fmt-py:
	$(ACTIVATE) && ruff format . && ruff check --fix .

## Run clippy lints
clippy:
	cargo clippy -- -D warnings

## Run Python lint checks
lint-py:
	$(ACTIVATE) && ruff format --check . && ruff check .

## Run all lint checks (Rust + Python) — use before pushing
lint:
	cargo fmt -- --check
	cargo clippy -- -D warnings
	$(ACTIVATE) && ruff format --check . && ruff check .

## Run tests with coverage report
cov:
	$(ACTIVATE) && pytest tests/ -v --cov=kglite --cov-report=term-missing

## Verify type stubs match runtime (requires built extension)
stubtest:
	$(ACTIVATE) && python -m mypy.stubtest kglite --ignore-missing-stub --mypy-config-file mypy_stubtest.ini --allowlist stubtest_allowlist.txt

## Remove build artifacts
clean:
	cargo clean
