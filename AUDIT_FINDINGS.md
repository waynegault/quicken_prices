# Architecture audit findings

This is the tracked, evidence-based record of the QuickenPrices audit. A
resolved item names the commit that implemented it. This is not a claim that
all technical debt has been eliminated.

## Resolved

### Correctness, validation, and data safety

- **Unsafe cache serialization** — replaced executable pickle with
  schema-checked, atomic CSV persistence; old `.pkl` files are not read.
  (`e852ce5`)
- **Cache corruption and interrupted writes were unpinned** — a cache truncated
  mid-row is now proven to read as nothing usable, so the caller refetches
  rather than trusting a fragment; an unreadable file fails; a grown schema is
  ignored; and a failed write leaves the previous cache byte-identical with no
  temporary file behind. (`8d2561f`)
- **Future FX rates used for past prices** — conversion is a backward as-of
  join and errors when a non-native price has no same-day-or-prior rate.
  (`bac1546`, `69f10b3`)
- **Unspecified FX policy for duplicates, stale rates, and missing series** —
  conversion now states and tests its rules: duplicate rates for one day keep
  the first and are logged, a rate older than `MAX_FX_RATE_AGE_DAYS` (7) is
  applied but warned about, a currency with no series at all is named in a
  warning before the run fails, and the pair is built from the configured home
  currency rather than a hardcoded `GBP`. (`bcf0c31`)
- **Silent fetch, conversion, and output failures** — provider, cache, parsing,
  conversion, and CSV-write failures are surfaced; the CLI logs and re-raises
  unexpected errors. (`9cb2690`, `69f10b3`)
- **Unvalidated settings and ticker metadata** — strict Pydantic v2 models
  reject unknown/coerced settings and duplicate symbols; validated settings
  objects are passed through runtime boundaries. Ticker metadata is modeled
  explicitly. (`cb64064`, `e350cbf`, `0125729`)
- **Explicit Quicken path silently falling back** — an invalid configured
  executable now fails instead of being replaced by auto-detection.
  (`0125729`)
- **Historical window and FX context errors** — results are constrained to
  the requested window, one prior FX row is retained for conversion, and a
  current quote is not added when the requested end date is in the past.
  (`39e4a7f`)
- **Ticker path and cache cleanup scope** — Pydantic ticker symbols reject path
  separators before cache use; expiry cleanup only removes `.csv` cache files.
  (`ef8095b`, `8ebe1bd`)
- **Non-idempotent output date formatting** — formatted `DD/MM/YYYY` output
  can be processed repeatedly without date reinterpretation. (`ba6147f`)
- **Unvalidated market date boundaries** — ticker metadata dates must be
  timezone-aware and history orchestration normalizes ranges to UTC.
  (`a10a238`)
- **Transient history request failures** — configured retry policy now applies
  to price downloads; the default retry decorator catches only network
  transient exceptions. (`b03723b`, `d800620`)
- **Rate-limited provider responses were not retried** — `YFRateLimitError` was
  missing from the default transient set, so the provider failure a scheduled
  run is most likely to meet aborted the run on its first attempt while the
  retry policy sat unused. The response edges around it are pinned too: empty,
  missing, wholly unusable, reordered columns, the same-day guard, and retry
  exhaustion. (`b54814b`)
- **Malformed-row drops were silent** — download, cache normalization, and
  conversion emit warnings when invalid or duplicate rows are discarded.
  (`466c4e6`)
- **Oversized FX conversion regression setup** — extracted the multi-row
  conversion scenario into named reusable test data, reducing duplication
  within the test and keeping the inverse-rate case explicit. (`4894685`)
- **Startup touching user cache / missing config creation** — startup checks
  use temporary storage and missing configuration is an error.
  (`ba63033`)

### Structure and maintainability

- **Monolithic module and mixed responsibilities** — market data, Windows
  automation, conversion, shared helpers, and storage have separate modules.
  (`fe7e95e`, `2532412`)
- **Repeated cache cleanup** — one immutable, idempotent normalizer is used for
  load, merge, and save. (`e809b90`)
- **Positional ticker metadata** — replaced tuples with a strict
  `TickerMetadata` model. (`e350cbf`)
- **Global runtime configuration** — removed; retry settings are explicit.
  (`2532412`)
- **Hidden GUI ownership and step dispatch** — Quicken lifecycle is owned by an
  explicit session object; workflow steps share a uniform callable contract
  and a failed step raises. (`0125729`)
- **Conversion monster and redundant arithmetic check** — conversion is split
  into focused transformations; invalid required input columns now fail, and
  a tautological floating-point check was removed. (`69f10b3`)
- **History-fetch orchestration** — split cache, download, merge, FX fill,
  latest quote, and requested-window responsibilities into small functions.
  (`39e4a7f`, `a10a238`)
- **Entry-point noise and success-shaped import status** — removed obsolete
  section scaffolding, clarified completion logging, and only configures GUI
  automation when elevated. (`da0dcac`)
- **Inconsistent entry-point names** — renamed the script and launcher to
  snake_case without compatibility aliases. (`5b01723`)
- **Interpreter fallback** — batch launcher, VS Code, application startup, and
  UAC relaunch require the repository `.venv`. (`ba63033`)
- **No automated gate** — a Windows CI workflow byte-compiles every module, runs
  the suite and lints on each push and pull request. `ruff.toml` pins the lint
  selection so the gate cannot drift with a Ruff release. (`bf248e1`)
- **Environment-dependent test** — `test_main_propagates_failures` asserted a
  configuration failure while depending on the checkout having a `.venv`, so it
  could not fail on a clean checkout. It neutralizes the interpreter check now,
  and the interpreter contract it depended on has its own test. Caught by the CI
  gate on that gate's first run. (`fa8ea9b`)
- **Quicken automation branches were untested** — elevation granted and refused,
  a missing or immediately-dead executable, a window that never appears, a
  failed step still cleaning up the process it launched, and both import-dialog
  failures now have mocked tests. The GUI-only remainder has a documented manual
  smoke procedure with expected results and cleanup. (`cf9ddaa`)
- **Concentrated test module** — the suite is split by responsibility into
  `test_quicken_prices.py` (configuration and entry point), `test_market_data.py`,
  `test_conversion.py`, `test_storage.py` and `test_quicken_automation.py`, with
  the conversion scenario data travelling with the tests that use it. Coverage is
  unchanged at 78 tests. (`31940c1`)

## Remaining and deferred

- **Internal cache gaps are not reconciled.** Cache planning fetches only
  leading and trailing ranges from date bounds; it cannot detect missing
  trading dates inside those bounds. Correct detection needs an exchange-aware
  calendar and a policy for holidays, so this remains open rather than adding
  a naive weekday-only heuristic.
- **Cache writes assume a single writer.** Replacement is atomic, so a reader
  sees the old file or the new one and never a half-written one, and the
  corruption cases are covered. Two overlapping runs still share the cache
  directory and the last writer wins. No lock was added because nothing has
  demonstrated the need: an overlap needs a lock, not a stronger atomicity
  guarantee.
- **Calendar assumptions are incomplete.** Date adjustment handles weekends
  and a small set of fixed-date holidays, not full exchange-specific holiday
  calendars or observed holidays.
- **Provider and Windows integration remain partly untested.** The mocked
  branches are covered and the GUI path has a documented manual smoke procedure,
  but no live network run or Quicken 2004 GUI end-to-end run has been performed,
  so both seams are still unverified against the real thing.
- **Operational logs are not tracked.** `prices.log` is ignored and no
  representative runtime log was available in the worktree to inspect.
- **Some provider/orchestration functions remain moderately large**, notably
  `get_tickers` and `validate_ticker` (63 and 48 lines in the current source).
  No remaining function is over 80 lines. Decompose further only when
  behavioral tests establish a useful boundary.
- **GUI success flows have no unit test.** They stay reachable only through the
  manual smoke procedure in the README, because driving Quicken's real dialogs is
  exactly what the mocked tests avoid.
- **No LLM task or evidence-classification subsystem exists in this
  repository.** LLM benchmark coverage and consolidation of evidence
  classification models are therefore not applicable to the audited code.
- **No generated version-stamping or AI-generated VCS metadata exists.** No
  versioning requirement was evident; adding automatic commits or generated
  version files would introduce behavior without a demonstrated need.
- **Some comments/docstrings and typing can still be improved.** The audit
  found no unused imports in the extracted application modules at the time of
  the last static check, no warning-suppression rules, and no `TODO`, `FIXME`,
  `pass`, or `NotImplemented` stubs in tracked Python code.
- **Tests do not access live user data.** File-writing tests use temporary
  directories; provider and GUI behavior is mocked. All Python validation was
  run with `.venv\Scripts\python.exe`.
- **The lint gate is deliberately narrow.** Ruff 0.16.6's own default rule set
  reports 159 findings on this code, so a bare `ruff check` cannot gate it.
  `ruff.toml` pins an explicit `select` covering only the families the code
  already satisfies; the rest stay open and visible rather than fixed blindly
  or suppressed: `LOG015` x101 (modules log through the root logger instead of
  a module logger), `E501` x52, the `PT009`/`PT027` x116 test-assertion style,
  `TRY003`/`TRY300`/`TRY301`/`TRY400` x50 with `EM101`/`EM102` x38 (exception
  construction), `TRY004` x2, `UP006`/`UP035`/`UP045`/`UP007` x40 (modern
  annotations), `SIM117` x8, `BLE001` x5 (broad handlers, each needing a stated
  contract), `PLR0913`/`PLR0917`/`PLR2004`/`PLW2901` x11, `PERF203` x2,
  `RUF005`/`RUF046` x3, `Q003`, `RET503`/`RET505` x2, `ARG005`. Each family
  needs a decision rather than a blanket `--fix`.

## Batch history

- Baseline — clean worktree; created `.venv` with Python 3.14.7 and installed
  requirements. `.\.venv\Scripts\python.exe -m unittest -v`: 12 passed.
- Findings recorded — `9db6fd7`.
- Strict configuration — `cb64064`; 15 tests passed.
- Safe atomic CSV cache — `e852ce5`; 17 tests passed.
- Historical FX direction — `bac1546`; 18 tests passed.
- Failure propagation and atomic output — `9cb2690`; 21 tests passed.
- Startup isolation and `.venv` enforcement — `ba63033`; 22 tests passed.
- Strict ticker metadata — `e350cbf`; 25 tests passed.
- Idempotent cache normalization — `e809b90`; 26 tests passed.
- Snake-case entry points — `5b01723`; 26 tests passed.
- Processing/storage module split — `fe7e95e`; 27 tests passed.
- Market/automation module split — `2532412`; 28 tests passed.
- Validated runtime settings and explicit Quicken lifecycle — `0125729`;
  33 tests passed.
- Conversion decomposition and input contracts — `69f10b3`; 37 tests passed.
- Bounded cache results with prior FX context — `39e4a7f`; 40 tests passed.
- Idempotent output date formatting — `ba6147f`; 40 tests passed.
- Restrict expiry cleanup to cache CSVs — `ef8095b`; 40 tests passed.
- UTC market metadata and smaller history orchestration — `a10a238`;
  41 tests passed.
- Entry-point cleanup and explicit GUI skip reporting — `da0dcac`;
  42 tests passed.
- Strict ticker symbols and cache-path validation — `8ebe1bd`;
  43 tests passed.
- Prior-day and pence FX regression coverage — `0f7894d`;
  44 tests passed.
- Configured transient retries for price downloads — `b03723b`;
  45 tests passed.
- Transient-only retry defaults — `d800620`; 46 tests passed.
- Malformed provider/cache/price rows now log discard warnings — `466c4e6`;
  48 tests passed.
- Documentation aligned with the final audit and runtime behavior —
  `d626ea1`.
- Extracted the reusable FX conversion test scenario — `4894685`; 48 tests
  passed, compileall and `git diff --check` passed.
- Windows CI workflow and a pinned lint baseline — `bf248e1`; 48 tests passed,
  Ruff clean, compileall clean.
- Stated and tested FX conversion policy — `bcf0c31`; 53 tests passed, Ruff
  clean, compileall clean.
- Provider response edges and rate-limit retries — `b54814b`; 60 tests passed,
  Ruff clean, compileall clean.
- Hermetic `main()` test and the interpreter contract — `fa8ea9b`; 61 tests
  passed, Ruff clean. The first CI run failed on the test this batch fixed.
- Quicken automation branch coverage and the manual smoke procedure —
  `cf9ddaa`; 73 tests passed, Ruff clean.
- Cache corruption and interrupted-write coverage — `8d2561f`; 78 tests passed,
  Ruff and compileall clean.
- Test suite split by responsibility — `31940c1`; 78 tests passed, Ruff and
  compileall clean.
