"""Verify openJiuwen can call an OpenAI-compatible contest proxy."""

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from run_agent import BudgetExhausted, JiuwenLLM, main


class Handler(BaseHTTPRequestHandler):
    last_max_tokens = None
    output_caps = []
    use_assessment = False
    assessment_responses = 0
    cli_calls = 0
    memory_prompt_seen = False

    def do_POST(self):
        assert self.path == "/v1/chat/completions", self.path
        size = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(size))
        Handler.last_max_tokens = request.get("max_tokens")
        Handler.output_caps.append(Handler.last_max_tokens)
        assert request["model"] == "qwen3.6-35b-a3b-noreason"
        content = "42"
        if Handler.use_assessment:
            Handler.cli_calls += 1
            assert 'nsrc-assessment-v1' in request['messages'][0]['content']
            if Handler.cli_calls == 1:
                content = 'CALC: unknown(6)'
            else:
                content = json.dumps({
                    'protocol': 'nsrc-assessment-v1',
                    'assessment': {'action': 'continue', 'confidence': 'low',
                                   'reason_code': 'uncertain', 'next_step': 'Calculate six times seven.'},
                    'result': 'CALC: 6*7'})
                Handler.assessment_responses += 1
            if Handler.cli_calls == 3:
                user = request['messages'][1]['content']
                assert 'Operational experience from local tool checks:' in user
                assert 'arithmetic checker' in user
                Handler.memory_prompt_seen = True
        body = json.dumps({
            "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
            "model": request["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


server = HTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
os.environ["LLM_BASE_URL"] = f"http://127.0.0.1:{server.server_port}/v1"
os.environ["LLM_API_KEY"] = "test-only"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"


async def check():
    llm = JiuwenLLM()
    answer = await llm.ask("Return 42", "What is the answer?")
    assert answer == "42", answer
    assert llm.spent == 3, llm.spent
    guarded = JiuwenLLM(budget=30)
    try:
        await guarded.ask("Return 42", "What is the answer?", max_tokens=40)
    except BudgetExhausted:
        pass
    else:
        raise AssertionError("budget guard allowed an oversized call")
    assert guarded.spent == 0
    bounded = JiuwenLLM(budget=260)
    assert await bounded.ask("Return 42", "What is the answer?", max_tokens=400) == "42"
    assert 32 <= Handler.last_max_tokens < 400, Handler.last_max_tokens

    # Exercise the judge's CLI entry point with the real SDK and a small env budget.
    with tempfile.TemporaryDirectory() as directory:
        tasks = Path(directory)
        for index in range(2):
            folder = tasks / f"math_{index}"
            (folder / "workspace").mkdir(parents=True)
            (folder / "task.json").write_text(json.dumps({
                "task_type": "math", "instruction": "Compute six times seven.",
                "reward": 5, "check": "exact_int"}), encoding="utf-8")
        first_call = len(Handler.output_caps)
        Handler.use_assessment = True
        argv = ["run_agent.py", "--tasks-dir", str(tasks),
                "--initial-model", "qwen3.6-35b-a3b-noreason",
                "--trace-file", str(tasks / "trace.jsonl")]
        with patch.object(sys, "argv", argv), patch.dict(os.environ, {"BUDGET": "500"}):
            await main()
        caps = Handler.output_caps[first_call:]
        assert len(caps) == 3, caps
        assert Handler.assessment_responses == 2
        assert Handler.memory_prompt_seen
        assert all(64 <= cap < 384 for cap in caps), caps
        for index in range(2):
            folder = tasks / f"math_{index}"
            assert (folder / "answer.txt").read_text(encoding="utf-8") == "42\n"
            assert (folder / "workspace" / "answer.txt").read_text(encoding="utf-8") == "42\n"


try:
    asyncio.run(check())
    print("openJiuwen proxy integration passed")
finally:
    server.shutdown()
