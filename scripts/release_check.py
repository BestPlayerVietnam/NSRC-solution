"""Check source packaging and optionally exercise an already-built image offline.

Uses stdlib, Git and Docker. Writes no repository files, never calls a remote
model, and reports potential credential matches as paths only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = (
    re.compile(rb"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(rb"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}"),
    re.compile(rb"-----BEGIN (?:(?:RSA|EC|OPENSSH) )?PRIVATE KEY-----"),
)


def execute(argv, *, timeout=60):
    env = {k:v for k,v in os.environ.items()
           if not k.upper().startswith(("LLM_", "NEURALDEEP_", "OPENAI_", "ANTHROPIC_"))}
    return subprocess.run(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout, check=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packaging_check():
    tracked = execute(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"])
    if tracked.returncode:
        raise RuntimeError("Git could not enumerate this checkout")
    eligible = set(part.decode("utf-8") for part in tracked.stdout.split(b"\0") if part)
    sources = {}
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        if not line.strip().upper().startswith("COPY "):
            continue
        words = shlex.split(line.strip())[1:]
        if len(words)<2 or any(word.startswith("--") for word in words):
            raise RuntimeError("Release checker needs explicit local Docker COPY sources")
        destination = words[-1]
        for name in words[:-1]:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or not (ROOT/relative).is_file():
                raise RuntimeError("Docker COPY input is not a local file: "+name)
            if (ROOT/relative).is_symlink():
                raise RuntimeError("Docker COPY symlink is unsupported: "+name)
            sources[name] = destination+relative.name if destination.endswith("/") else destination
    if not sources:
        raise RuntimeError("Dockerfile contains no explicit COPY sources")
    mandatory = {"run_agent.py", "requirements-runtime.txt", "cost_artifact.json", "residual_cost_artifact.json"}
    if not mandatory.issubset(sources):
        raise RuntimeError("Missing mandatory runtime/artifact COPY inputs: "+", ".join(sorted(mandatory-set(sources))))
    missing_imports = set()
    for name in sources:
        if not name.endswith(".py"):
            continue
        tree = ast.parse((ROOT/name).read_text(encoding="utf-8-sig"),filename=name)
        for node in ast.walk(tree):
            modules = ([item.name for item in node.names] if isinstance(node,ast.Import)
                       else [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in modules:
                local = module.split(".")[0]+".py"
                if (ROOT/local).is_file() and local not in sources:
                    missing_imports.add(local)
    if missing_imports:
        raise RuntimeError("Local runtime imports missing from Docker COPY: "+", ".join(sorted(missing_imports)))
    missing = sorted(({"Dockerfile", ".dockerignore", *sources}-eligible))
    if missing:
        raise RuntimeError("Required files are absent from the Git-eligible tree: "+", ".join(missing))
    ignore = (ROOT/".dockerignore").read_text(encoding="utf-8").splitlines()
    whitelist = {line[1:] for line in ignore if line.startswith("!")}
    if "**" not in ignore or not set(sources).issubset(whitelist):
        raise RuntimeError("Docker context allowlist does not cover every COPY input")
    candidates, unsafe, size, count = [], [], 0, 0
    forbidden_parts = {".tools", ".venv", "venv", "__pycache__", ".pytest_cache",
                       ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints", ".codex",
                       ".vscode", ".idea", "logs", "tmp_repo-master", "TASK"}
    for name in sorted(eligible):
        path = ROOT/name
        if not path.is_file():
            continue
        if path.is_symlink():
            unsafe.append(name)
            continue
        parts = Path(name).parts
        if any(part in forbidden_parts for part in parts) or name.startswith("experiments/runs/") or any(part.startswith("suite") for part in parts[:-1] if parts[0]=="experiments"):
            unsafe.append(name)
        content = path.read_bytes()
        size += len(content); count += 1
        private_environment = ((path.name.startswith(".env") and name != ".env.example")
                               or path.suffix.lower() == ".env")
        if private_environment or path.suffix.lower() in {".key", ".pem", ".p12", ".pfx"} or any(pattern.search(content) for pattern in SECRET_PATTERNS):
            candidates.append(name)
    if candidates or unsafe:
        # Never print credential values or matching source lines.
        for name in sorted(set(candidates+unsafe)):
            print(name)
        raise RuntimeError("Potential credentials or excluded local artifacts found; paths printed above")
    if size>=100_000_000:
        raise RuntimeError("Git-eligible tree exceeds the100MB repository limit")
    print(json.dumps({"packaging":"passed", "git_eligible_files":count, "bytes":size,
                      "docker_copy_files":len(sources), "secret_candidate_paths":[]}))
    return sources


def docker_command(image, arguments, mounts=()):
    name = "nsrc-release-check-"+uuid.uuid4().hex
    command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
               "--cpus", "8", "--memory", "16g", "--pids-limit", "512", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,size=2g",
               "--tmpfs", "/app/.tools:rw,size=256m", "--workdir", "/app",
               "--env", "HOME=/tmp", "--env", "PYTHONPATH=/app", "--env", "PYTHONDONTWRITEBYTECODE=1"]
    for path in mounts:
        command += ["--mount", f"type=bind,source={path},target=/app/{path.name},readonly"]
    command += [image, "python", "-B", *arguments]
    try:
        result = execute(command, timeout=600)
        if result.returncode:
            print(result.stdout.decode("utf-8",errors="replace"))
            print(result.stderr.decode("utf-8",errors="replace"),file=sys.stderr)
            raise RuntimeError("Offline image check failed")
        return result.stdout.decode("utf-8",errors="replace"), result.stderr.decode("utf-8",errors="replace")
    finally:
        # Only this invocation can own the unpredictable container name above.
        execute(["docker", "container", "rm", "--force", name])


def image_check(image, sources):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*",image):
        raise ValueError("Invalid Docker image reference")
    code = "import hashlib,json,pathlib; paths="+repr(sources)+"; print(json.dumps({k:hashlib.sha256(pathlib.Path(v).read_bytes()).hexdigest() for k,v in paths.items()}))"
    stdout, _ = docker_command(image,["-c",code])
    actual = json.loads(stdout)
    if actual != {name:sha(ROOT/name) for name in sources}:
        raise RuntimeError("Image files differ from the checkout; rebuild the image")
    print(json.dumps({"image_hashes":"matched", "files":len(sources)}))
    excluded = {"integration_test.py", "smoke_test.py"}
    if "result_constraints.py" not in sources:
        excluded.add("result_constraints_test.py")
    tests = sorted(path for path in ROOT.glob("*_test.py") if path.name not in excluded)
    stdout, stderr = docker_command(image,["-m","unittest","-v",*[path.stem for path in tests]],tests)
    output = stdout+stderr
    counts = re.findall(r"Ran (\d+) tests? in",output)
    if len(counts)!=1:
        raise RuntimeError("Could not verify unit-test completion")
    skipped = re.findall(r"OK \(skipped=(\d+)\)",output)
    print(json.dumps({"runtime_tests":"passed", "run":int(counts[0]),
                      "skipped":sum(map(int,skipped))}))
    for line in output.splitlines():
        if " ... skipped " in line:
            print(line)
    stdout, _ = docker_command(image,["/app/integration_test.py"],(ROOT/"integration_test.py",))
    print(stdout.strip())
    print("Remote model API calls used by these checks: 0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",help="Also check image hashes, portable runtime units and the real SDK with a local fake API")
    args = parser.parse_args()
    sources = packaging_check()
    if args.image:
        image_check(args.image,sources)


if __name__=="__main__":
    try:
        main()
    except (OSError,RuntimeError,ValueError,subprocess.TimeoutExpired) as exc:
        print(str(exc),file=sys.stderr)
        raise SystemExit(1)
