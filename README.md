# NSRC openJiuwen Challenge harness

The harness uses the official `openjiuwen` `Model` client for all LLM calls.
It reads `LLM_BASE_URL` and `LLM_API_KEY` only from the process environment.
No credentials are included in the image.

Build: `docker build -t nsrc-harness .`

Run with writable task directories:

```sh
docker run --rm -e LLM_BASE_URL -e LLM_API_KEY -v "$TASKS:/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g nsrc-harness \
  python /app/run_agent.py --tasks-dir /data/tasks
```

The scheduler first solves recognizable math, CSV, and knowledge-base queries
with exact local code at zero LLM-token cost. It then ranks remaining tasks by
reward divided by a rough task-type token estimate. For CSV queries outside its local
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
are short and use the non-reasoning models unless a coding repair needs the
larger model. The run stops after 55 minutes to preserve saved answers before
the one-hour timeout.

There is no trained cost model yet. The cost estimate is a rule-based task-type
estimate calibrated to the supplied eight example tasks. No external trajectories
were provided, and no hidden task answers are used.

Local checks: `python smoke_test.py` with the organizer's sample `TASK/task-dir`
directory; `python integration_test.py` uses a local dummy OpenAI-compatible
server and spends no real API tokens. Docker Desktop requires Windows Virtual
Machine Platform and a reboot on the development workstation; the evaluator
builds in its own Linux environment.
