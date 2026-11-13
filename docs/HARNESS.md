# NSRC-solution

A budget-aware coding-agent harness for the openJiuwen Challenge. It solves math,
Python coding, local document search and CSV analysis tasks through the official
openJiuwen client and CPU tools. The default policy uses `qwen3.8-27b`, adaptive
repair and the guarded cost predictor. One weighted-token budget covers the whole
run, including estimation prompts and retries.

## Run from a fresh clone

Docker with BuildKit is required. Python 3.11 and Git are needed on the host only
for the optional release checks. Run the following POSIX-shell commands from the
repository root. Supply the organizer's proxy credentials and an absolute task
path through the environment.

```sh
export TASKS=/absolute/path/to/tasks
export LLM_BASE_URL=https://your-contest-proxy/v1
export LLM_API_KEY='<proxy-key>'
export BUDGET=100000

docker build -t nsrc-harness .
docker run --rm -e LLM_BASE_URL -e LLM_API_KEY -e BUDGET -e HOME=/tmp \
  -v "$TASKS:/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g \
  --cpus 8 --memory 16g --pids-limit 512 \
  --cap-drop ALL --security-opt no-new-privileges \
  --workdir /app --entrypoint python nsrc-harness \
  /app/run_agent.py --tasks-dir /data/tasks \
  --trace-file /data/tasks/nsrc-trajectory.jsonl
```

On the contest host, add `--network hack-net` as shown in the organizer's launch
command. That command requests 8 CPUs. The prose resource table says 4. The
harness also runs with `--cpus 4`. There is no GPU dependency. The agent's internal
deadline is 55 minutes, within the published one-hour execution limit.

`--time-limit N` sets a shorter positive run limit, up to the default 3,300
seconds. It includes task discovery, local preparation and model execution.
`--sla-policy off` is the release default and uses token queue scores with the
shared deadline guards and latency forecasts in traces. `--sla-policy adaptive`
enables joint token/time admission and continuation. Classic comparison runs
use SLA mode `off`.

Budget precedence is `--budget N`, then `BUDGET`, then `100000`. Values are positive
finite numbers in weighted output-token units. Decimal input reaches the ledger
exactly, which rounds the spendable limit down to 0.1 token. Tasks and physical
requests share this budget throughout the run. A CLI value can be appended to the run
command, for example `--budget 20000`.

The model can be selected with `--initial-model`. Supported identifiers are
`qwen3.8-27b`, `qwen3.8-27b-noreason`, `qwen3.6-35b-a3b` and
`qwen3.6-35b-a3b-noreason`. Fixed-model execution is the default.
`--allow-model-switch` enables a separate routing policy. `--retry-policy classic`
selects the earlier two-attempt comparison policy. `--cost-estimator residual`
selects the optional residual predictor. Guarded remains the default.

`--confidence-policy heuristic` is the default continuation controller.
`--confidence-policy off` selects the earlier confidence behavior. The classic
comparison policy also uses that earlier behavior.

`--memory-policy revise` enables bounded cross-task tool experience by default.
`append` keeps each initial lesson text fixed, and `off` disables this memory.
Classic comparison runs use memory mode `off`.

## Task and answer files

The `--tasks-dir` path contains one directory per task. Each directory has
`task.json`, whose visible fields include `task_id`, `task_type`, `instruction`,
`reward` and an optional `check`. The runtime ignores sample `expected` fields.
Task files normally live in a `workspace` subdirectory. When that subdirectory is
absent, the task directory itself is used as the workspace.

```text
tasks/
  task_001/
    task.json
    workspace/
      mylib.py
      test_mylib.py
```

Math, search and data-analysis answers contain one final value on one UTF-8 line.
For coding, the harness edits the actual source files, preserves the tests and
writes `DONE` after the available tests pass. Hidden tests determine the final
score, so a public-test pass is recorded as public verification only.

Each answer is saved immediately with an atomic file replacement. The same
`answer.txt` is written in both the task directory and its workspace. This covers
the two locations described by the regulation and sample instructions. A skipped,
failed or abstained task has no answer file. `UNSOLVABLE` is an internal refusal
marker used by the solver.

Submission follows the provided Docker command and its per-task `answer.txt`
contract. The mounted task directory contains saved answers and coding edits.
Existing answer files mark completed tasks. Use a fresh task copy for each
independent run.

Traces default to `/tmp/nsrc-trajectory.jsonl` and disappear with the container.
The explicit `--trace-file` above preserves them on the task mount. Traces contain
forecasts, normalized usage, caps, finish reasons, latency and public verification
status. They omit prompts and credentials. Console output also reports progress.

A committed trajectory example records one real API call from the earlier public
study, including its source image, prediction and reconciled token usage.
[Trajectory example](trajectory-example.json)

## Budgeting and execution

Weighted cost is input tokens divided by 10 plus all generated tokens, with
reasoning counted once. Provider caching never discounts input. Before every
physical request, the ledger reserves prompt UTF-8 bytes divided by 10, 128 units
of overhead and twice the output cap. It reduces the cap when the remaining budget
requires it. The reservation assumes that the proxy enforces the requested
generation cap.

Complete and consistent provider usage replaces the reservation with the measured
charge. Unknown usage and ambiguous failures retain a conservative charge. SDK
retries are disabled. An explicitly retried transient request gets its own
reservation. Quota and authentication failures stop the run. A usage report above
the reservation stops further requests. Traces distinguish reconciled usage from
conservative holds.

Supported local math, CSV and search patterns run first. The entire question must
match the supported semantics. Additional conditions defer to the general solver.
The planner then ranks tasks using reward, estimated success and predicted cost.
Online cost corrections are separated by domain, model and execution policy.
Unverified answers do not provide hidden-correctness labels. Unsupported input and
model errors are recorded separately from narrow, explicit impossibility checks.

In the default adaptive policy, before running task tests or constructing the model client, coding feasibility
checks look for contradictory requirements under an explicit integrity contract.
They cover conflicting deterministic integer results and bounded finite domains
of exact immutable Python values. The finite checker interprets supported test
ASTs and exhausts a proven superset of candidate strings, bytes or integer
permutations. It evaluates at most 512 candidates and 100,000 operations per task.
Unknown syntax, ambiguous contracts and exhausted analysis limits leave the task
unproven. A proved contradiction is skipped with zero model tokens and its reason
recorded in the trace. The analyzer returns the exhausted domain and assumptions
as evidence.

| Domain | CPU tools and verification |
| --- | --- |
| Math | Bounded arithmetic AST and exact rational calculations. Generated arithmetic cannot execute imports or Python statements. |
| CSV | Bounded complete tables in SQLite, JOIN/CTE support, quoted Unicode columns, typed numbers and a schema preview. Queries are read-only, instruction-limited and return one scalar. SQL errors survive correction attempts. Integer-result contracts and abstention markers are checked. |
| Search | Unicode lexical retrieval, excerpts from large files and source provenance. Answers are checked against retrieved evidence. Retrieval may omit relevant evidence. |
| Coding | Nested source files, exact edits, protected tests and rollback on failed repair. pytest runs in a disposable copy without proxy credentials. Test execution is verified through a plugin report. |

## Resource-control checklist

| Control | Current behavior |
| --- | --- |
| Token allocation | A shared weighted budget, task-cost forecasts and reservations before each physical request. Generation caps adapt to the remaining balance. |
| Task admission | Affordable tasks are ranked by reward, heuristic success score and predicted token cost. Proven contradictions are skipped. Doubtful continuations can be deferred. |
| Dynamic resources | Reconciled usage updates cost corrections. Tool feedback, progress, competing tasks and truncated generations guide continuation and cap changes. |
| Early stopping | Refusals, repeated candidates, lack of progress and weak continuation value can stop further calls. Decisions occur between API calls. |
| Output economy | Local tools, short final formats, exact source edits, bounded retrieval and selected experience cards reduce repeated context and generation work. |
| Cost and time | Per-model/domain latency forecasts and deadline-bounded API/local tools run by default. Optional adaptive mode uses both token and time estimates for task admission, ranking, continuations and generation caps. |

## Deadline-aware planning

The CPU latency estimator starts from 989 completed historical API calls with
known usage. Initial model-specific request intercepts, token slopes and empirical
tail buffers are recorded with source hashes. Observed latency includes provider
queueing, networking and complete response generation. Per-model/domain rolling
histories hold at most 32 observations of each type. Completed known-usage calls
update a robust rate correction. Unknown usage, failures and timeouts can raise
the conservative duration estimate. Tool and preparation overhead is measured
within each active visit, excluding API time, transport backoff and time spent
waiting behind another task.
[Latency prior provenance](../experiments/SLA_LATENCY_PRIORS_2026-11-13.json)

In adaptive mode, with enough time, task order follows the existing reward/confidence-per-token
priority. If the queue's point duration estimates exceed the remaining time, its score
uses expected reward divided by the larger token or time share of the remaining
resources. New tasks must pass both token admission and point-estimate time
admission. Empirical upper estimates remain available in traces as uncertainty
diagnostics. Continuations retain their existing token guard and get a fresh
comparison with affordable alternatives. A task deferred by SLA scheduling keeps
its attempts, candidates and SQL feedback even with confidence mode `off`.

In adaptive mode, before each physical call, a monotone search reduces the generation cap to fit
the point duration estimate within the remaining time. Latency uses the existing character-based expected
prompt-cost prior. Token reservations independently use the UTF-8 upper estimate.
A truncated reply whose reachable cap cannot grow
stops before another reservation. Local SQL execution, public-test copying and
the test subprocess also check the run deadline. Traces record latency forecasts,
active visit duration, answer time and queue/continuation/cap decisions.

These are heuristic point forecasts with empirical tail diagnostics. Provider load can
change, and synchronous file operations use cooperative deadline checks. The
default 55-minute run limit leaves five minutes within the contest hour for
startup and finalization.

## Confidence and continuation

The controller uses hand-written rules and existing model calls. Before execution,
visible context coverage and advisory feasibility flags can reduce a task's
success score. Overlapping signals apply their strongest reduction once. This
score stays positive and affects queue ordering. The fixed adjustments are
heuristics with no empirical probability-calibration guarantee.

During a solver call, a short instruction lets the model optionally return a
versioned assessment with a result or a concrete proposed next check. Ordinary
SQL, CALC, answer and code formats continue to work. A structured response uses
the following shape.

```json
{"protocol":"nsrc-assessment-v1","assessment":{"action":"defer","confidence":"low","reason_code":"uncertain","next_step":"Inspect the failed assertion and check the integer division."},"result":null}
```

The supported actions are `continue`, `defer` and `stop`, with confidence levels
`low`, `medium` and `high`. Reasons are `missing_information`, `no_progress`,
`tool_error`, `contradictory_requirements`, `uncertain` and `solvable`. A continue
result can contain the ordinary answer text or the existing `files`/`edits`
repair object. Defer and stop carry a null result. Invalid or truncated envelopes
use the existing parsing and generation-limit handling.

The continuation decision combines that self-report with test progress, repeated
candidates, tool errors, newly retrieved evidence and the last reconciled call
cost. High self-reported confidence cannot override a terminal or repeated-error
guard. A locally verified result can be accepted with low self-reported confidence.
A concrete diagnostic repair can proceed with surplus budget. Under competition,
the controller compares the next attempt's heuristic reward per token with
affordable alternatives and can defer a doubtful task.

Deferred tasks return after the main queue, with at most one return each. Their
attempt counts, candidate history and SQL feedback persist across the return.
Failed coding changes are rolled back. A search failure on partial excerpts can
trigger one bounded CPU retrieval expansion. A retrieval-driven retry requires
actual new excerpts and must pass the usual budget and continuation guards.

Traces record aggregate assessment fields and the task score. A deferred visit
has `outcome=deferred` and `execution_complete=false`. `actual_tokens` describes
that visit, while `task_actual_tokens` accumulates the task's visits. Online cost
updates occur once on final completion, against the first plan and all recorded
task costs. Assessment text and proposed next checks stay out of the trace.

## Cross-task tool experience

The memory follows ReHarness's retrieve-and-revise idea with CPU-generated
operational lessons. A run starts with an empty in-process bank. An allowlisted
tool error followed by a later successful check on the same task, model and
domain can create a card. The eight supported categories cover arithmetic
format, SQLite identifiers/query shape/integer format and coding edit
shape/paths/unique spans/large sources. Advice comes from fixed conditional
templates. Source references contain hashes and event positions.
[ReHarness](https://arxiv.org/html/2610.12668v1)

The bank holds at most 32 cards globally. Model and domain scopes are isolated.
Lexical operation hints and current tool errors select at most two cards,
rendered within 1,000 characters. A task is excluded from retrieving cards to
which it contributed. Answers, table values, source fragments, raw error text
and model self-assessments are excluded from stored cards.

In `revise` mode, a recurring known tool error can narrow a supplied card's
applicability to that error and increment its caution counter. A subsequent
verified repair can update its support evidence. The bank retains up to three
versions per card and preserves the original witness references. In `append`
mode, existing card text stays fixed. The verification levels are
`tool_execution` and `public_tests`, describing the actual local checks.

All retrieval, writing and revision run on the CPU. They add zero model calls.
Rendered lessons travel inside ordinary solver prompts and share their ledger
reservations. Each physical request, including transport retries, records the
exact injected UTF-8 byte count, card IDs and versions. A memory failure leaves
the ordinary solver available. The bank resets between process runs.

Software checks cover error-to-repair learning, transfer to a different task,
revision after recurring errors, routing/model isolation, protected coding
tests, and exact prompt accounting. The real openJiuwen SDK also exercises an
error, repair and cross-task lesson through the normal CLI. The final API panel
below measures the cold-built memory image. The earlier confidence probe keeps
its own image and measurements.

## Cost predictors

`cost_artifact.json` stores CPU regression parameters for first-call and total
policy cost, separated by model. Training and calibration families are disjoint.
Method selection uses grouped training cross-validation. `forecast_guard.py`
falls back to a prompt-aware prior for unsupported extrapolation or invalid
predictions. Classic and routed policies use their own prior.

`residual_cost_artifact.json` provides an optional stdlib predictor using robust
global corrections and domain corrections shrunk toward them. Its point forecasts
were selected for absolute error. Calibration upper estimates are empirical
residual diagnostics.
`--cost-artifact` and `--residual-cost-artifact` select the two artifact files
independently. Invalid or mismatched artifacts fall back with a recorded reason.

Both artifacts are included in the repository and image. No neural model runs
locally. All LLM requests go through the configured proxy to the four allowed
models. Runtime downloads are unnecessary. `requirements-runtime.txt` pins the
97 runtime distributions, including `openjiuwen==0.1.20`. Base-image `pip` and
`wheel` are additional bootstrap distributions. The broader
`requirements.txt` describes the development environment. Python and the temporary
uv build-tool image have pinned version tags. BLAS/OpenMP threads are bounded.

The confidence controller has a separate `|confidence-v1` execution-policy suffix.
Memory modes add `|memory-v1-append` or `|memory-v1-revise`. The frozen cost
artifacts predate these controllers. Their forecasts serve as
transfer estimates, with added prompt overhead and a `policy_transfer` trace
flag. Their earlier calibration measurements describe the recorded source
policy. Subsequent online cost corrections use the new policy's observations.
Adaptive deadline planning adds `|sla-v3` and the same `policy_transfer` marker.
Prompt overhead uses actual retrieved text length and the frozen total/first
cost ratio as a heuristic request-count proxy. Actual billing uses provider
usage and the shared reservation ledger.

For direct Python development, use a separate Python 3.11 virtual environment
and install the tested runtime closure without resolving additional SDK extras.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-deps --only-binary=:all: -r requirements-runtime.txt
```

## Measured validation and limits

The tested SLA v3 image passed 468 portable runtime checks. Two additional checks require
local archived fixtures and were skipped. The set includes 37 latency-planner
checks and 33 independent deadline/continuation regressions. The real SDK passed
the normal CLI integration against a local test API. Its 21 runtime inputs match
the tested checkout, and the Docker CLI reports 644MB. Its source rebuild took
3.418 seconds with dependency layers cached. A full isolated
`--pull --no-cache --load` build of the preceding SLA v1 source took 821.547
seconds. Both versions use the same Dockerfile and pinned dependency inputs.

The first SLA study completed 16 runs. At a 90-second deadline, its reward
off/adaptive was 8/11, 20/6, 10/12 and 16/18 for Qwen 3.6 plain, 3.8 plain,
3.6 reasoning and 3.8 reasoning. The Qwen 3.8 plain regression exposed inflated
prompt cost in generation-time admission. V2 uses the existing expected prompt
prior for latency. Three dedicated tests cover long prompts, Unicode and exact
token reservations. The original measurements and conservative local timeout
holds remain recorded in the first study's report.
[SLA v1 study](../experiments/SLA_RESULTS_V1_2026-11-13.json)

The second study used 12 new public groups and four reused negative controls.
Buffered time admission reduced timeouts and lost reward for three models.
Its results motivated point-estimate planning in v3. The runtime enforces the
absolute deadline while the point forecast guides reward allocation.
[SLA v2 study](../experiments/SLA_RESULTS_V2_2026-11-13.json)

The third study used another 12 fresh public groups and the four reused negative
controls. Every model had separate 100,000-token off/adaptive pipelines with a
90-second limit. Verified on-time reward was measured out of 31 points.

| Model | Reward off / adaptive | Weighted tokens off / adaptive |
| --- | ---: | ---: |
| Qwen 3.6 plain | 27 / 27 | 5,578.4 / 4,643.8 |
| Qwen 3.8 plain | 26 / 27 | 2,630.8 / 2,132.2 |
| Qwen 3.6 reasoning | 20 / 22 | 7,566.7 / 5,873.7 |
| Qwen 3.8 reasoning | 25 / 18 | 4,017.1 / 2,318.2 |

All eight processes completed, with 126 provider requests and 34,760.9 weighted
tokens. Late provider charges and conservative local timeout holds are reported
separately. The default Qwen 3.8 reasoning model lost seven points in adaptive
mode. Its two missing tasks were unattempted under the forecast, and actual
service delays varied across paired calls. Adaptive therefore remains an opt-in
development policy. The release selects `off`, with shared deadlines, bounded
tools and diagnostic latency forecasts. The post-study source change sets that
CLI default. Explicit `off` and `adaptive` executions keep the tested behavior.
[SLA v3 study and audit](../experiments/SLA_RESULTS_2026-11-13.json)

These panels contain one run per setting. Search uses frozen exact accepted
spans. The four negative controls were reused across studies. Forecasts and
reward differences have no universal performance or calibrated SLA guarantee.

The final release changes the CLI default to `off`. An exact byte comparison
against the API-tested source verifies this single runtime change. It passed
469 portable checks, with two additional optional fixture skips, and the normal
SDK integration. The set includes 37 latency-planner and 34 SLA integration
checks. Another 144 research tests passed locally. All 21 runtime inputs match
the final image. The cached source rebuild took 4.647 seconds.
[Release configuration decision](../experiments/SLA_RELEASE_DECISION_2026-11-14.json)

### Earlier memory release

[The release validation summary](../experiments/FINAL_RELEASE_VALIDATION_2026-11-13.json)
records the preceding memory image's cold-build checks, API panel, resource
controls and cleanup.

That release passed 398 portable runtime checks in a clean source export.
Two additional checks require local archived-forecast and training fixtures and
were skipped. Another 25 hardening checks, eight memory/accounting checks and all
eight organizer examples passed on
4 CPUs. The real openJiuwen SDK also exercised the normal CLI with `BUDGET=500`
against a local test API, including reduced output caps, a tool repair,
cross-task memory and two saved answers in three calls.
The package checker compares all 20 Docker inputs with their image copies.

The preceding memory image completed eight primary runs on 20 fresh tasks, with
16 answerable tasks, four authored negative cases and 47 available reward points.
Twelve tasks use previously unused public source groups from GSM8K, HumanEval,
SQuAD2 and WikiSQL. Eight are authored structural or negative cases. Each model
and memory mode had its own 100,000-token pipeline. All primary runs visited all
20 tasks. Results include hidden coding checks. Search uses frozen exact accepted
spans, so a longer answer expressing the same fact can lose credit.

| Model | Reward off / revise | Solved off / revise, of 16 | Weighted tokens off / revise |
| --- | ---: | ---: | ---: |
| Qwen 3.6 plain | 35 / 36 | 13 / 13 | 5,177.9 / 5,909.7 |
| Qwen 3.8 plain | 32 / 40 | 12 / 14 | 3,273.2 / 4,001.3 |
| Qwen 3.6 reasoning | 35 / 40 | 13 / 14 | 16,054.6 / 21,744.8 |
| Qwen 3.8 reasoning | 40 / 40 | 14 / 14 | 12,097.6 / 14,700.6 |

All four negative cases were left unanswered in every primary run. The default
Qwen 3.8 reasoning/revise arm spent 7,926.2 tokens on them, within its total of
14,700.6. Early stopping operates after responses, so a refusal can consume a
substantial reasoning generation before that decision.

One memory card reached version 2 and was injected once in the Qwen 3.6
reasoning/revise arm. The other primary arms had zero retrievals. This single-pair development panel
provides limited evidence for a causal memory benefit. Model variability and
the limited task mix affect the observed differences.

Initial online total-cost MAE on each revise arm's 20 completed trajectories was
232.70, 171.11, 672.48 and 553.40 weighted tokens in the model order above.
The frozen and legacy forecasts use the same rows and are preserved in the report.
Forecast gains vary by model. No parameters were fitted using this panel.

Two additional Qwen 3.8 reasoning runs received `BUDGET=5000.5` through the
environment. Off/revise spent 4,243.7/3,973.6 tokens and earned 29/26 points,
visiting 14/11 tasks. The remaining 6/9 tasks were unattempted. Across all ten
runs, 206 physical requests reconciled with the proxy and task traces, with zero
API errors, pending requests or budget breaches. The stress runs demonstrate
budget-pressure behavior with censored task coverage.
[Final release API validation](../experiments/FINAL_RELEASE_API_2026-11-13.json)

The finite-domain skip checker was audited on 69 already-known coding tasks.
It proved three contradictions and made zero hard skips among 54 answerable
tasks. Twelve authored negative cases remained unproven. The normal CLI skipped
the three proved cases with zero model tokens, without running their code or
writing answers. Those cases had consumed 11,862.6 weighted tokens in the earlier
default-model run. This is a post-development audit on those three cases. The
fresh final panel above evaluates the complete current policy on different tasks.
[Finite-domain audit](../experiments/FINITE_SKIP_RESULTS_2026-11-13.json)

A targeted confidence A/B probe used eight new authored tasks, four answerable
and four negative, across all four models. Each on/off arm had its own 100,000
token ledger. All 64 task episodes and 81 physical requests completed, with zero
API errors and pending reservations. Complete-policy results were as follows.

| Model | Points off / on | Weighted tokens off / on |
| --- | ---: | ---: |
| Qwen 3.6 plain | 4 / 4 | 684.5 / 2,401.0 |
| Qwen 3.8 plain | 3 / 3 | 622.9 / 735.5 |
| Qwen 3.6 reasoning | 3 / 4 | 12,203.9 / 5,888.5 |
| Qwen 3.8 reasoning | 4 / 4 | 6,736.6 / 2,952.6 |

Only two explicit assessments were emitted. Prompt wording, task order and
generation variability can contribute to the measured differences. The probe
has surplus budget and provides a small development check. Budget-pressure
allocation is covered separately by scripted CLI tests. After the probe, a
regression-tested guard was tightened to stop a model refusal without an
actionable plan before deferral. The measured API image and final release
image are recorded separately. Coding grading used a recovered image with the
original executable image configuration and matching runtime files, after its
original provenance manifest became unavailable locally.
[Confidence probe results](../experiments/CONFIDENCE_RESULTS_2026-11-13.json)

The public-source validation panel contains 100 tasks across four domains, with
80 answerable tasks and 20 authored negative cases. On the frozen fitted solver,
adaptive Qwen 3.8 reasoning earned **178/247 reward**, solving **60/80** answerable
tasks for **67,381.7 weighted tokens**. This result motivated the default model.
The choice used validation results. Subsequent runtime hardening, SQL fixes and finite-domain skip checks
have their own checks. The 178-point score belongs to the earlier frozen solver.

Public total-policy cost MAE, in weighted tokens, was measured on the same
observed adaptive trajectories per model.

| Model | Old heuristic | Prompt prior | Guarded | Residual |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.6 plain | 293.33 | 307.97 | 314.85 | 310.44 |
| Qwen 3.8 plain | 138.08 | 165.36 | 155.07 | 136.95 |
| Qwen 3.6 reasoning | 750.52 | 483.03 | 437.14 | 352.21 |
| Qwen 3.8 reasoning | 527.53 | 388.42 | 430.34 | 442.17 |

Predictor gains vary by model. Residual forecasts were frozen before reading the
public labels and evaluated offline on guarded-policy trajectories. That
comparison measures forecasts on the original guarded-policy trajectories.
All public arms attempted all 100 tasks with the available budget. Model
pretraining overlap and documented source-label ambiguities limit generalization.

The SQL-guard candidate passed **196 runtime tests and 25 hardening tests in
Linux**, with no skips when the archived parity fixtures were present. All eight
organizer examples and a real openJiuwen client against a local fake API passed.
A targeted SQL development probe subsequently gave the default model 7/7 positive
answers and abstention on two negative cases. Eight cases were reused regressions.
This is a small development check with one fresh case.

The measured image size is **644MB** under `docker images --format`. The current
memory source export built in **236.703 seconds** with a fresh isolated BuildKit
builder, `--pull --no-cache --load`, and matching hashes for all 20 runtime inputs.
Builder bootstrap took **2.922 seconds** separately. All 97 pinned runtime
distributions matched, and the new image passed the portable checks and SDK
integration. This duration describes the measured host and network.

A saved earlier
fitted context built in **796.875 seconds** with a fresh BuildKit builder and
`--pull --no-cache --load`, including application base images, wheels and image
loading. Builder bootstrap took 20.937 seconds separately. Later source changes
belong to later source snapshots using the same pinned dependency installation.
The reported duration describes that machine and network measurement.
The repository must remain below 100MB and the image below 4GB.

[The completed study summary](../experiments/RESULTS_POLICY_2026-11-13.txt) preserves
all model results, accounting, uncertainty intervals, source-label issues and
post-validation changes. [The policy protocol](../experiments/POLICY_STUDY_PROTOCOL.md)
describes the 64/32/48 synthetic split and the public panel.
[The earlier production study](../experiments/PRODUCTION_RESULTS_2026-11-12.md) and
[the first two studies](../experiments/RESULTS_2026-11-12.md) retain their original
measurements. [The research report](../CPU_HARNESS_REPORT.md) explains design choices.

## Reproduce the software checks

The release checker requires no Python packages on the host. Its static mode
checks the Git-eligible file tree, Docker COPY inputs, source size and common
credential patterns. Potential secret matches are reported as file paths only.
The pattern scan cannot recognize every possible credential format.

```sh
python3 -B scripts/release_check.py
python3 -B scripts/release_check.py --image nsrc-harness
```

Image mode verifies that the copied files match the checkout, then runs portable
runtime tests and the real SDK against a local fake API. It uses 8 CPUs, 16GB RAM,
a read-only root and isolated networking. Test files are mounted individually.
Imports use modules from the image. Counts and skipped cases are printed. The
frozen-forecast parity test is explicitly skipped when its optional local fixtures
are absent. The finite-domain checker and its shared result-constraint helpers are
included in the image and this runtime test set. These checks spend no model tokens.

Organizer examples are supplied separately and are excluded from Git. Once placed
at `TASK/task-dir`, they can be checked without API calls.

```sh
python3 -B smoke_test.py --docker --image nsrc-harness
```

Synthetic and public-source suites can be generated and their oracles checked
without LLM calls. Public-source construction downloads the pinned source data.

```sh
python3 -B experiments/build_policy_suite.py --image nsrc-harness
python3 -B experiments/build_public_panel.py --download --image nsrc-harness
```

These commands create local, ignored suites. Public source-group exclusions depend
on previously available suites, so a fresh checkout can select a different panel
from the archived study. Dataset provenance and source licenses are recorded by
the builders. Full historical replay additionally requires the archived input
snapshots, frozen images, protocol files and ledgers referenced by the study.

Raw records under `experiments/runs/`, generated suites, downloaded data and local
Docker tags are development artifacts excluded from Git. Paths to these records
in historical reports identify local development archives.
Summary reports, generators, runners, tests and production cost artifacts are
included. Runtime cost artifacts remain usable without the raw training traces.
New paid experiments need their own frozen protocol and persistent per-pipeline
ledger. Resuming a partial run must preserve its recorded charges.
