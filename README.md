# QuickenPrices

QuickenPrices downloads daily market prices from Yahoo Finance, converts
non-home-currency prices using historical exchange rates, writes a Quicken
import CSV, and can automate importing it into Quicken 2004 on Windows.

## Run

Windows and a project-local virtual environment are required. The supported
interpreter is Python 3.14 — the release the repository `.venv` is built with and
the one CI installs, taken from `.python-version`:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe quicken_prices.py
```

Name the release explicitly when creating the venv. A bare `py -3` takes the
launcher's own default, which is 3.12 on this machine, and the application checks
the venv's *interpreter path* rather than its version — so a venv built from
another release runs instead of warning.

`requirements.txt` is the only declaration of the dependency constraints; there is
no lock file. A single-user script rebuilt from those ranges is reproducible
enough, and a freeze would pin transitive packages to one machine's resolution.

### Launch it through the venv

`quicken_prices.bat` uses only `.venv\Scripts\python.exe`, and that is the way to
start the script. VS Code is configured for the same interpreter.

Double-clicking `quicken_prices.py`, or pointing a shortcut at the `.py` instead
of at the launcher, runs it with the base Python that Windows associates with
`.py`. That is `py.exe` here, and it never picks the repository venv: a script run
from this directory selects `python314`, while the launcher's own default is 3.12.
Whichever it selects lacks the project's packages, so the run ends at its first
import and never reaches `prices.log`:

```
'pydantic' is not installed for this interpreter:
  C:\Programs\Python\python314\python.exe

Run the script through the repository environment instead:
  quicken_prices.bat
  .\.venv\Scripts\python.exe quicken_prices.py
```

Point shortcuts at `quicken_prices.bat`, not at the `.py`. The application also
refuses to start under a different Python once it gets far enough to check.

The script reads the tracked `configuration.yaml`; it does not generate or
rewrite a missing configuration. Configuration is validated strictly with
Pydantic: unknown keys and wrong value types are errors. Defaults are applied
only for fields defined by the settings models.

`paths.quicken` is optional. If omitted, Quicken is discovered from the Windows
registry or standard install locations. An explicitly configured but invalid
path is an error; it is not silently replaced by auto-detection. Without
elevation, the script still writes the CSV and prints manual import steps.
Failures during an attempted GUI import are reported as failures.

## Configuration

Edit `configuration.yaml` to set ticker symbols, home currency, collection
period, retries, logging, paths, and cache expiry. Example:

```yaml
tickers:
  - AAPL
  - MSFT.L
home_currency: GBP
collection_period_years: 1
paths:
  data_file: data.csv
  log_file: prices.log
  cache: cache
```

The checked-in configuration contains actual ticker symbols. Tickers are sent
to Yahoo Finance for validation and price retrieval, so treat the configuration
as potentially sensitive when sharing the repository.

## Data and failure behavior

- Per-ticker cache entries are schema-checked CSV files in `cache/`; writes
  replace atomically. Expiry cleanup only removes CSV caches. Legacy pickle
  caches are intentionally not read.
- Cache results are limited to the requested date range. FX data retains one
  prior rate as conversion context.
- FX conversion looks the rate up as `<CURRENCY><HOME>=X` — home-currency units
  per unit of the foreign currency — matched to the price date or the closest
  earlier rate. A same-day rate is preferred and a future rate is never used. A
  currency with no series at all is named in a warning and fails the run, so a
  missing series is never mistaken for a flat price. Duplicate rates for one day
  keep the first and are logged. A rate more than `MAX_FX_RATE_AGE_DAYS` (7) days
  old is still applied but logged as a warning: a long market closure is not a
  data error, a stale series usually is.
- `data.csv` is written without a header using Quicken's `DD/MM/YYYY` date
  format. A failed fetch, conversion, validation, or write is not reported as
  success.
- Malformed rows discarded during provider, cache, or conversion
  normalization are logged as warnings.
- Transient validation and historical-download network errors use the
  configured retry policy; malformed data and other non-transient errors are
  not retried.
- `prices.log` is configured with rotation in `configuration.yaml`.
- When Quicken is launched by this script, only that owned process is closed.
  An already-running user-owned Quicken process is left open.

## Project layout

| Module | Responsibility |
| --- | --- |
| `quicken_prices.py` | Strict configuration loading and application orchestration |
| `settings.py`, `market_models.py` | Pydantic settings and ticker metadata |
| `market_data.py` | Yahoo Finance validation, retry, cache planning, and fetching |
| `price_processing.py` | Historical FX conversion and Quicken output formatting |
| `storage.py` | Cache normalization, atomic CSV persistence, and expiry cleanup |
| `quicken_automation.py` | Windows elevation, Quicken process ownership, and GUI workflow |
| `common.py` | Small shared formatting and currency helpers |

## Tests

The suite is split by responsibility and run with the repository environment, so
imports use the project's installed dependencies:

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

| Module | Covers |
| --- | --- |
| `test_quicken_prices.py` | Configuration loading, startup checks and `main()` |
| `test_market_data.py` | Ticker validation, retries, downloads and cache planning |
| `test_conversion.py` | FX conversion policy and Quicken output formatting |
| `test_storage.py` | Cache round-trips, corruption, and atomic CSV writes |
| `test_quicken_automation.py` | Quicken process ownership, elevation and GUI steps |

Every test that writes a file uses a temporary directory, and provider and GUI
calls are mocked, so the suite needs no network, no Quicken installation, and no
access to your cached prices.

## Manual smoke test (Windows + Quicken 2004)

The GUI path cannot be covered by the unit suite, since provider and GUI calls
are mocked there. Run this by hand after changing `quicken_automation.py`,
elevation handling, or the CSV output format.

**Before starting:** Quicken installed with its portfolio already set up; the
repository `.venv` present; `configuration.yaml` holding your own tickers; and
nothing else touching the mouse or keyboard while an import runs.

| # | Do this | Expect |
| --- | --- | --- |
| 1 | Close Quicken, run `quicken_prices.bat`, accept the UAC prompt | "Successfully launched elevated process in a new window", then Quicken opens, the portfolio view is reached, and the run logs "Successfully imported ..." |
| 2 | Watch the run | The import dialog title is `Import Price Data`, and the confirmation appears in the portfolio window |
| 3 | Leave Quicken open, run the script again | "Quicken is already open." — and afterwards Quicken is **still open**: only a process this script launched gets closed |
| 4 | Run again and decline the UAC prompt | The run continues without elevation, logs "Quicken GUI automation requires elevation; CSV export is ready.", prints the manual import steps, and leaves `data.csv` ready |
| 5 | Point `paths.quicken` at a path that does not exist, then run | "Quicken executable not found" and a non-zero exit — the script must not imply it imported anything |
| 6 | Quit Quicken mid-import, then run again | "Quicken window did not appear within the expected time" and a non-zero exit |

**Cleanup:** an interrupted run can leave Quicken open with a modal import
dialog. Close Quicken by hand — the script terminates only the process it
started — then read `prices.log`, and delete a partial `data.csv` if the run
failed before writing it.

**Record:** keep the `prices.log` from a passing run. No representative runtime
log is tracked in this repository, so that file is the only evidence of what the
GUI path actually does.
