FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: -r /app/requirements.txt
COPY run_agent.py /app/run_agent.py
COPY planner.py /app/planner.py
CMD ["python", "/app/run_agent.py", "--tasks-dir", "/data/tasks"]
