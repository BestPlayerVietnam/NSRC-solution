"""Verify openJiuwen can call an OpenAI-compatible contest proxy."""

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from run_agent import JiuwenLLM


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == "/v1/chat/completions", self.path
        size = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(size))
        assert request["model"] == "qwen3.6-35b-a3b-noreason"
        body = json.dumps({
            "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
            "model": request["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "42"},
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


try:
    asyncio.run(check())
    print("openJiuwen proxy integration passed")
finally:
    server.shutdown()
