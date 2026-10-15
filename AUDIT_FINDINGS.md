# Architecture audit findings

This is a prioritized, evidence-based working list for the 2026-10-15 audit.
Items are not claimed fixed until marked **Resolved** and backed by a batch
commit. The audit is intentionally critical; it does not imply every concern
is a defect or that all technical debt can be eliminated.

## P0 — correctness, data safety, and failure visibility

- [x] **Unsafe and opaque cache persistence** — cache storage now uses
  schema-checked CSV instead of executable pickle data; corrupt/empty files
  and invalid writes raise explicit errors, and writes replace atomically.
-  Legacy `.pkl` caches are intentionally not read. Persistence failures now
  bubble through data fetching and the application entry point.
- [x] **Currency conversion can silently use future rates** —
  `convert_prices` no longer backfills; its backward as-of join uses a rate
  only from the same or an earlier date, and missing prior rates raise a
  descriptive error. A regression test verifies the future-rate case.
- [x] **Success-shaped fetch failures** — provider/parsing/cache failures now
  propagate from `download_data` and `fetch_historical_data`; genuinely empty
  provider responses remain an explicit no-data result.
- [x] **Configuration isn't validated as a single contract** —
  `load_configuration` merges untyped dictionaries and silently falls back to
  defaults after parse errors; checks are split between it,
  `run_startup_self_check`, and individual functions. `settings.py` now defines
  strict Pydantic v2 models; config parsing rejects unknown keys, wrong types,
  malformed YAML, and missing files instead of silently changing behavior.
- [x] **Startup mutates project state** — configuration loading no longer
  creates a missing YAML file, and startup self-checks only use a temporary
  cache directory; operational cache creation occurs only when fetching data.
- [x] **Main path hides failure status** — `main` logs unexpected exceptions
  with tracebacks and re-raises them, so CLI failures produce a nonzero status.
  CSV validation and write errors also propagate; cache and output CSVs share
  one atomic writer.

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
- [x] **Duplicated cache/DataFrame cleanup** — `normalize_cache_data` now owns
  cache load/merge/save schema cleanup, numeric/date normalization, sorting,
  and deduplication. It is idempotent and does not mutate its input.
- [x] **Positional ticker metadata** — replaced the five-element tuple with a
  strict Pydantic `TickerMetadata` model at provider and pipeline boundaries.
  Configuration rejects duplicate ticker symbols case-insensitively.
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

- [x] **Launcher bypasses the project environment** — the batch launcher and
  VS Code run configuration now require `.venv\Scripts\python.exe`, the UAC
  relaunch explicitly selects that interpreter, and `main` verifies it before
  running.
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
- **Safe cache format** — replaced pickle reads/writes with schema-checked CSV,
  atomic writes, and explicit corrupt/invalid-cache errors. Added tests for the
  format and invalid file/data cases. `.\.venv\Scripts\python.exe -m unittest
  -v` — 17 tests passed; `.\.venv\Scripts\python.exe -m compileall -q
  QuickenPrices.py settings.py test_quicken_prices.py` — passed. Pending commit.
- **FX rate direction** — removed the backward-fill of exchange rates and fail
  when an equity price has no same-day-or-prior FX rate. Added a regression
  test showing that a future quote is never applied. `.\.venv\Scripts\python.exe
  -m unittest -v` — 18 tests passed. Pending commit.
- **Failure visibility and atomic output** — removed broad success-shaped
  catches from ticker validation, historical fetching, and downloads; the CLI
  re-raises logged errors; output CSV writes are atomic and raise on invalid
  data or I/O failures. Added regression tests for provider errors, repeatable
  output, write failure, and main error propagation. `.\.venv\Scripts\python.exe
  -m unittest -v` — 21 tests passed; compileall passed. Pending commit.
- **Startup isolation and interpreter enforcement** — removed startup writes
  to the configured cache, made the cache self-check temporary, disallowed
  Python executable fallbacks, enforced the project venv in the CLI/UAC path,
  and updated launcher/VS Code/installation docs. Added a test proving the
  self-check leaves the configured cache absent. `.\.venv\Scripts\python.exe
  -m unittest -v` — 22 tests passed; compileall and `git diff --check` passed.
  Pending commit.
- **Validated ticker domain model** — introduced strict Pydantic metadata for
  ticker-provider results and all fetch-pipeline inputs, and reject duplicate
  configured tickers ignoring case. Added provider model, invalid-type, order,
  and downstream error-propagation tests. `.\.venv\Scripts\python.exe -m
  unittest -v` — 25 tests passed; compileall and `git diff --check` passed.
  Pending commit.
- **Cache normalization** — consolidated repeated cache/merge/save cleanup in
  one idempotent normalizer, removed defensive `None` branches inconsistent
  with the typed APIs, and added an immutability/idempotence regression test.
  `.\.venv\Scripts\python.exe -m unittest -v` — 26 tests passed; compileall and
  `git diff --check` passed. Pending commit.
