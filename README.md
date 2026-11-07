# NSRC openJiuwen Challenge harness

The harness uses the official `openjiuwen` `Model` client for all LLM calls.
It reads `LLM_BASE_URL` and `LLM_API_KEY` only from the process environment.
No credentials are included in the image.
All 172 Python package versions are pinned in `requirements.txt` for Linux
and Python 3.11, including `openjiuwen==0.1.20`.
openJiuwen logging is configured for console output so a read-only root
filesystem does not prevent model calls.

Build: `docker build -t nsrc-harness .`

Run with writable task directories:

```sh
docker run --rm -e LLM_BASE_URL -e LLM_API_KEY -v "$TASKS:/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g nsrc-harness \
  python /app/run_agent.py --tasks-dir /data/tasks
```

The scheduler first solves recognizable math, CSV, and knowledge-base queries
with exact local code at zero LLM-token cost. It then ranks remaining tasks by
`reward * estimated_success_probability / predicted_weighted_tokens`, recalculating
after every model-backed task. The default shared budget is 100,000 weighted
tokens; `--budget` can override it for local experiments. The model client
reserves a safety margin and guards every request by its maximum output tokens.
It records actual usage and updates both cost and answer-rate estimates online.
For CSV queries outside its local
pattern, openJiuwen produces a single SQLite SELECT, evaluated on the local CSV.
For coding tasks, the harness first applies known sample-family function
repairs and checks the supplied tests. Other coding tasks use openJiuwen to
propose a full module replacement, then run the tests and try once more using
test feedback. It restores the original module after unsuccessful attempts.
Missing files, invalid SQL,
failing tests, and explicit `UNSOLVABLE` responses are skipped without an answer.

Each successful answer is written atomically to the task directory and, when
present, its `workspace` directory, because the regulations describe the former
while sample instructions describe the latter. No answer is written for failed tasks. The
sample `expected` field is never read. Output and reasoning tokens count at full
weight, input tokens at one tenth, using the provider's usage metadata. Calls
start with a non-reasoning model and may retry high-value failures on a reasoning
model. All four allowed models are supported. The run stops after 55 minutes to preserve saved answers before
the one-hour timeout.

There is no trained cost model yet. The prior cost and answer-rate estimates are
rule-based and calibrated on the small synthetic set in `calibration_run.py`.
Per-task predicted
and actual weighted tokens are written as JSONL to `/tmp/nsrc-trajectory.jsonl`
by default; `--trace-file` can set a persistent path in development. The trace
contains no prompts or credentials. Model answers without local verification
do not increase the learned success rate. See `CONTEST_STRATEGY.md` for the fixed
contest assumptions and `CALIBRATION.md` for measured trajectories and limits.

Local checks: `python smoke_test.py` with the organizer's sample `TASK/task-dir`
directory, or `python smoke_test.py --docker` after building the image;
`python integration_test.py` uses a local dummy OpenAI-compatible
server and spends no real API tokens. `python planner_test.py` checks local
cost estimates and impossibility decisions. `python calibration_run.py` needs Docker
and the API environment variables, and spends a small amount of real tokens.
Docker Desktop requires Windows Virtual
Machine Platform on the development workstation; the evaluator builds in its
own Linux environment. The image built in approximately 155 seconds, measured
1.42 GB, passed all eight samples in a read-only container, and completed a
real one-call NeuralDeep probe from that container.
