FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY requirements-runtime.txt /app/requirements.txt
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.24,source=/uv,target=/bin/uv \
    UV_CONCURRENT_DOWNLOADS=8 uv pip install --system --no-cache --no-deps --only-binary=:all: -r /app/requirements.txt
COPY run_agent.py /app/run_agent.py
COPY planner.py budget_runtime.py analytical_tools.py coding_tools.py context_tools.py feasibility.py finite_constraints.py result_constraints.py policy_features.py retry_policy.py confidence_policy.py task_confidence.py experience_memory.py latency_planner.py cost_model.py forecast_guard.py residual_predictor.py /app/
COPY cost_artifact.json /app/cost_artifact.json
COPY residual_cost_artifact.json /app/residual_cost_artifact.json
CMD ["python", "/app/run_agent.py", "--tasks-dir", "/data/tasks"]
