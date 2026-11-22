# Text-to-SQL Analyst

A production-grade, safety-first Text-to-SQL agent that translates natural language questions into safe, verified SQL queries against a clinic database. Built with strict AST-based security validation, read-only database execution guarantees, an LLM self-correction retry loop, auto-selected Plotly data visualizations, and an evaluation harness.

---

## Architecture & Security Boundary

```
                     +---------------------------------+
                     |     User Natural Language Q     |
                     +---------------------------------+
                                      |
                                      v
                     +---------------------------------+
                     |   Deterministic Safety Check    |  <-- Fast Path (NOT the security boundary)
                     |   (Refuses obvious write/PII)   |
                     +---------------------------------+
                                      | (Allowed)
                                      v
                     +---------------------------------+
                     |      LLM Provider Interface     |  <-- Ollama (default) / Gemini / Mock
                     |    (Pydantic Structured Output) |
                     +---------------------------------+
                                      |
                                      v
    +-----------------------------------------------------------------------+
    |                         SECURITY BOUNDARY                             |
    |                                                                       |
    |  1. SQLValidator (sqlglot AST Parser)                                 |
    |     - Enforces single SELECT statement only                           |
    |     - Table & column allowlists; CTE shadowing prevention            |
    |     - Sensitive column denylist (phone, email, address, ssn)          |
    |     - Safe wildcard expansion (SELECT * / alias.*)                    |
    |     - Function allowlist (blocks zeroblob, load_extension, etc.)      |
    |     - Clamps LIMIT <= 1,000; rewrites dates to 2024-12-31 boundary    |
    |                                                                       |
    |  2. SQLExecutor (Read-Only Database Engine)                           |
    |     - Type-safe ValidatedSQL token requirement                        |
    |     - SQLite connection in read-only mode (mode=ro, PRAGMA query_only)|
    |     - Query timeout progress handler (cancels slow queries)           |
    |     - Memory limit enforcement (SQLITE_LIMIT_LENGTH)                  |
    +-----------------------------------------------------------------------+
                                      |
                                      +--- (Validation or DB Error) ---> [Self-Correction Loop]
                                      |                                  (Feeds error back, up to 3x)
                                      v
                     +---------------------------------+
                     |      Result Envelope (pandas)   |
                     |  + Auto-Picked Plotly Chart     |
                     |  + Structured Audit Log (JSONL) |
                     +---------------------------------+
```

> [!IMPORTANT]
> **Security Boundary Architecture & PII Denylist Policy**:
> 1. The **AST validator (`SQLValidator`) plus the read-only database connection (`SQLExecutor`) constitutes the actual security boundary**.
> 2. The keyword pre-check (`_check_deterministic_refusal`) is **only a fast path** designed to avoid unnecessary LLM latency and API token consumption on obvious mutations, prompt injections, and raw PII exfiltration.
> 3. **PII Denylist Policy**: Protected columns (`phone`, `email`, `address`, `ssn`) **cannot be referenced anywhere in a query, even in aggregate functions or conditional expressions** (e.g. `COUNT(email)`, `WHERE email IS NOT NULL`, `CASE WHEN phone IS NULL...`). Any SQL referencing a denylisted column is blocked closed by the AST validator (`SecurityValidationError`). Inquiries attempting to query, infer, or count patient contact details are refused under the clinic data privacy policy.

---

## Key Features

- **Strict AST Validation (`sqlglot`)**: Parses every generated query into an abstract syntax tree. Validates against schema allowlists, expands wildcards on sensitive tables, rewrites relative dates to a fixed reference point (`2024-12-31`), and clamps queries to a safe `LIMIT`.
- **Read-Only SQLite Engine (`sqlite3` / `SQLAlchemy`)**: Enforces `mode=ro`, sets a SQLite progress handler to terminate runaway queries within `QUERY_TIMEOUT_SECONDS`, and restricts maximum query memory size. Only queries wrapped in `ValidatedSQL` can be executed.
- **Self-Correction Retry Loop**: Automatically intercepts syntax errors, missing columns, or database runtime issues, feeding detailed diagnostic feedback back into the model for up to 3 attempts.
- **Provider Interface**: Single clean abstraction supporting:
  - **Ollama** (default, free, local execution)
  - **Google Gemini** (via official Google GenAI SDK with rate-limiting, exponential backoff, and daily quota detection)
  - **Mock Provider** (for deterministic unit tests and CI without network)
- **Auto-Selected Visualizations**: Intelligently analyzes result sets (shapes, categorical columns, date series, distributions) to automatically render Plotly bar charts, line graphs, pie charts, or tables.
- **Comprehensive Evaluation Harness**: Benchmark suite of 56 accuracy queries across easy, medium, hard, and benign trigger-word queries, 1 explicit policy refusal query (`q57`), plus 22 adversarial safety cases. Includes rate limiting (12 req/min), disk response caching (`eval/cache/`), and graceful daily quota suspension.

---

## Evaluation Benchmark

The evaluation harness compares generated queries against human-validated gold SQL using an order-insensitive, float-rounded execution comparator.

```bash
# Verify all gold benchmark queries against the seeded database
python eval/verify_gold_sql.py

# Verify all safety benchmark cases against defensive boundaries
python eval/verify_safety_cases.py

# Run the complete evaluation harness
python eval/run_eval.py
```

*Results are recorded in `eval/results/latest.json` alongside model name, timestamp, and git commit hash.*

---

## Streamlit Interactive Application

Launch the interactive web interface locally:

```bash
streamlit run app/streamlit_app.py
```

### Application Features & Safety Architecture
- **Question Input & Benchmark Example Chips**: 6 clickable example chips covering single-value metrics, ranked tables, monthly trend lines, and payment distributions.
- **SQL Code & Interactive Visualizations**: Inspect generated SQL code blocks, scrollable pandas dataframes, and responsive auto-picked Plotly charts.
- **Strict AST Security**: Human-readable banners for policy refusals, validator blocks, and provider quota limits (zero stack traces or API key leaks).
- **Sidebar Data Explorer**:
  - Non-denylisted schema viewer (tables, column types, row counts; sensitive PII fields completely hidden).
  - Fixed reference date (`2024-12-31`) and synthetic dataset notice.
  - Evaluation summary dynamically loaded from `eval/results/latest.json`.
  - Clickable session history to re-inspect previous results without re-running.
- **Demo Mode & Quota Protection**: Max 10 live queries per session. When quota is reached or demo mode is toggled, answers to benchmark questions replay from verified cache.

