FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install --no-cache-dir openjiuwen==0.1.20
COPY run_agent.py /app/run_agent.py
CMD ["python", "/app/run_agent.py", "--tasks-dir", "/data/tasks"]
