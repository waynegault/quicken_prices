# Architecture audit findings

This is a prioritized, evidence-based working list for the 2026-10-15 audit.
Items are not claimed fixed until marked **Resolved** and backed by a batch
commit. The audit is intentionally critical; it does not imply every concern
is a defect or that all technical debt can be eliminated.

## P0 — correctness, data safety, and failure visibility

- [ ] **Unsafe and opaque cache persistence** — `QuickenPrices.py` uses
  `pickle.load` for cache files (`load_cache`) and converts several read/write
  errors into an `"empty"` status or a log message. Use a non-executable,
  schema-checked format and make persistence failures visible to callers.
- [ ] **Currency conversion can silently use future rates** —
  `convert_prices` applies `.ffill().bfill()` to missing FX rates. Backfilling
  can apply a later-known exchange rate to an earlier equity date, which is
  financially misleading. Define and test the permitted rate direction and
  fail on unresolved dates.
- [ ] **Success-shaped fetch failures** — `download_data` catches every
  exception and returns an empty frame; `fetch_historical_data` also catches
  broadly and continues. Callers cannot distinguish no market data from API,
  parsing, or programming failures.
- [x] **Configuration isn't validated as a single contract** —
  `load_configuration` merges untyped dictionaries and silently falls back to
  defaults after parse errors; checks are split between it,
  `run_startup_self_check`, and individual functions. `settings.py` now defines
  strict Pydantic v2 models; config parsing rejects unknown keys, wrong types,
  malformed YAML, and missing files instead of silently changing behavior.
- [ ] **Startup mutates project state** — `run_startup_self_check` creates the
  cache directory and a probe. Configuration loading no longer creates a
  missing YAML file; make the remaining operational preflight repeatable and
  isolate tests from user files.
- [ ] **Main path hides failure status** — `main` catches general exceptions,
  logs them, and returns normally; callers and launchers can observe a zero
  exit status for failed runs. Propagate errors or return explicit nonzero
  process status.

## P1 — structural complexity and duplication

- [ ] **Monolithic module / mixed responsibilities** — `QuickenPrices.py`
  (about 1,960 lines) combines configuration, network access, caching,
  conversion, CSV output, Windows elevation, and GUI automation. Split along
  stable boundaries and remove circular imports rather than layering wrappers.
- [ ] **Large multi-responsibility functions** — `fetch_ticker_data` handles
  cache validation, downloading, merging, persistence, FX filling, and live
  price insertion; `convert_prices` validates, reports coverage, merges FX,
  imputes rates, converts, and formats. Extract cohesive transformations with
  contracts and focused tests.
- [ ] **Duplicated ticker/cache and DataFrame cleanup** — cache rows are
  normalized in `fetch_ticker_data`, then normalized again after merging;
  ticker metadata is carried as positional 5-tuples. Share one schema/model and
  one normalization path.
- [ ] **Global configuration and GUI lifecycle state** — retry behavior reads
  the global `config`, `main` replaces it, and Quicken ownership uses the
  `QUICKEN_SESSION_STARTED` global. Pass explicit dependencies and lifecycle
  state.
- [ ] **Weak orchestration dispatch** — `execute_import_sequence` decides
  whether to pass configuration by comparing function objects. Give each
  workflow step a uniform callable contract.
- [ ] **Overlapping configuration and health checks** — defaults, ad-hoc
  validation, and startup checks each describe parts of the same config
  contract. Replace overlapping checks with one strict model and limited
  operational preflight.

## P2 — launcher, tests, and maintainability

- [ ] **Launcher bypasses the project environment** — `quickenprice.bat`
  falls back to `py -3` if `.venv` is missing; the UAC relaunch uses
  `sys.executable`. Ensure all application launches use the repository
  `.venv` and report a missing environment directly.
- [ ] **Tests cover a narrow subset** — 12 unit tests cover date ranges,
  cache/fetch basics, one FX conversion, and CSV preparation. Add regression
  tests for strict config validation, retries/failure propagation, cache
  corruption, FX rate direction, idempotency, saving, and orchestration. Keep
  all filesystem effects under temporary directories and mock network/GUI.
- [ ] **No LLM benchmark coverage** — no LLM-using task is evident in this
  repository; reassess if such a task or pipeline is added.
- [ ] **Documentation contradicts runtime behavior** — `README.md` claims
  modular architecture, uses old `pickle`/config descriptions, and describes
  launch/exit behavior that does not match current code. Revise it after the
  structural changes.
- [ ] **Stale / noisy implementation details** — unused imports, repeated
  comments and formatting, weak annotations (`Any`/untyped `config`), and
  likely dead constants/branches should be removed after call-site checks.
- [ ] **Automatic version control metadata absent** — no generated build or
  version stamping mechanism exists. Determine whether a runtime version is
  actually required before adding one; avoid creating unnecessary
  configurability or generated files.

## Batch log

- **Baseline** — clean worktree before audit changes; created repository
  `.venv` with Python 3.14.7 and installed `requirements.txt`. Ran
  `.\.venv\Scripts\python.exe -m unittest -v`: 12 tests passed.
- **Findings recorded** — pending commit.
- **Strict configuration** — added strict settings models, removed recursive
  default merging and silent parse fallback, and added regression tests for
  model defaults, invalid types/keys, and missing-file non-creation.
  `.\.venv\Scripts\python.exe -m unittest -v` — 15 tests passed;
  `.\.venv\Scripts\python.exe -m compileall -q QuickenPrices.py settings.py
  test_quicken_prices.py` — passed. Pending commit.
