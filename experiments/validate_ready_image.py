"""Build a separate ready image and run offline Linux QA against its own modules.

Never calls a remote model, copies credentials, changes the default image tag,
or changes runtime/dependency sources. Test inputs are copied before the build.
"""
from __future__ import annotations

import datetime as dt
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/2026-11-13-ready"
TAG = "nsrc-policy-v3-ready:20261113"
REPORT_NAME = "ready_image_validation.json"
RUNTIME = (
    "run_agent.py", "planner.py", "budget_runtime.py", "analytical_tools.py",
    "coding_tools.py", "context_tools.py", "feasibility.py", "policy_features.py",
    "retry_policy.py", "cost_model.py", "forecast_guard.py", "cost_artifact.json",
    "residual_predictor.py", "residual_cost_artifact.json",
)
PROTECTED = ("nsrc-harness:latest", "nsrc-policy-v3-fitted:20261112")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Validation:
    def __init__(self):
        self.token = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower() + "-" + uuid.uuid4().hex[:8]
        self.folder = RUN / self.token
        self.context = ROOT / ".tools" / ("ready-image-context-" + self.token)
        self.fixtures = ROOT / ".tools" / ("ready-image-tests-" + self.token)
        for path in (self.folder, self.context, self.fixtures):
            path.mkdir(parents=True, exist_ok=False)
        self.env = {k:v for k,v in os.environ.items()
                    if not k.upper().startswith(("LLM_", "NEURALDEEP_", "OPENAI_", "ANTHROPIC_"))}
        self.container_names = []
        self.data = {"status":"initializing", "started_at":dt.datetime.now(dt.timezone.utc).isoformat(),
            "tag":TAG, "api_calls":0, "commands":{}, "context":str(self.context),
            "fixture_snapshot":str(self.fixtures), "container_resources":{"cpus":8, "memory":"16g", "tmpfs":"2g", "network":"none", "read_only":True},
            "limitations":["This ready-image build may reuse Docker dependency caches; cold timing belongs to the separately frozen fitted context.",
                            "Secret scan is a pattern scan of git-eligible files, not a guarantee that every possible credential format is detectable."]}
        self.save()

    def save(self):
        path = self.folder / REPORT_NAME
        path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        (RUN / REPORT_NAME).write_text(json.dumps(self.data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    def command(self, stage, argv, timeout=300, required=True):
        log = self.folder / (stage+".log")
        start = time.monotonic()
        record = {"argv":argv, "timeout_seconds":timeout, "log":str(log)}
        self.data["commands"][stage] = record
        self.data["active_stage"] = stage
        self.save()
        print(json.dumps({"stage":stage,"state":"running"}), flush=True)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with log.open("wb") as output:
            proc = subprocess.Popen(argv, cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL,
                                    stdout=output, stderr=subprocess.STDOUT, creationflags=flags)
            record["owned_pid"] = proc.pid
            timed_out = False
            while proc.poll() is None:
                remaining = start+timeout-time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    if os.name == "nt":
                        subprocess.run(["taskkill","/PID",str(proc.pid),"/T","/F"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, creationflags=flags)
                    else:
                        proc.kill()
                    proc.wait(timeout=30)
                    break
                try:
                    proc.wait(timeout=min(20,remaining))
                except subprocess.TimeoutExpired:
                    print(json.dumps({"stage":stage,"elapsed_seconds":round(time.monotonic()-start,1)}), flush=True)
        record.update(returncode=proc.returncode, timed_out=timed_out, elapsed_seconds=round(time.monotonic()-start,3))
        text = log.read_text(encoding="utf-8", errors="replace")
        count = re.findall(r"Ran (\d+) tests? in", text)
        if count:
            record["unittest_count"] = sum(map(int,count))
            record["unittest_skipped"] = sum(map(int,re.findall(r"OK \(skipped=(\d+)\)",text)))
        self.save()
        print(json.dumps({"stage":stage, "returncode":proc.returncode, "elapsed_seconds":record["elapsed_seconds"]}), flush=True)
        if required and (timed_out or proc.returncode):
            raise RuntimeError("QA command failed: "+stage)
        return text.strip()

    def copy_fixture(self, relative):
        src, dst = ROOT / relative, self.fixtures / relative
        if src.is_symlink():
            raise RuntimeError("Fixture symlink is not allowed: "+relative)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if src.is_dir():
            shutil.copytree(src,dst,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
        else:
            shutil.copyfile(src,dst)

    def container(self, *args, fixture_files=(), fixture_dirs=()):
        name = "nsrc-ready-check-"+self.token+"-"+str(len(self.container_names))
        self.container_names.append(name)
        argv = ["docker","run","--rm","--name",name,"--network","none","--read-only","--cpus","8","--memory","16g",
                "--pids-limit","512","--cap-drop","ALL","--security-opt","no-new-privileges",
                "--tmpfs","/tmp:rw,size=2g","--tmpfs","/app/.tools:rw,size=256m",
                "--workdir","/app","--env","PYTHONPATH=/app","--env","PYTHONDONTWRITEBYTECODE=1"]
        for relative in (*fixture_files,*fixture_dirs):
            argv += ["--mount",f"type=bind,source={self.fixtures / relative},target=/app/{relative.replace(chr(92),'/')},readonly"]
        return argv + [self.data["image_id"], *args]

    def prepare(self):
        self.command("tag_absent",["docker","image","inspect",TAG],required=False)
        if self.data["commands"]["tag_absent"]["returncode"] == 0:
            raise RuntimeError("Refusing to overwrite an existing ready tag")
        self.data["default_builder_before"] = self.command("builder_before",["docker","buildx","inspect"])
        self.data["protected_before"] = {tag:self.command("protected_before_"+str(i),["docker","image","inspect",tag,"--format","{{.Id}}"])
                                         for i,tag in enumerate(PROTECTED)}
        runtime = list(RUNTIME)
        dockerfile = (ROOT/"Dockerfile").read_text(encoding="utf-8")
        if "result_constraints.py" in dockerfile:
            runtime.append("result_constraints.py")
        self.runtime = runtime
        context_files = ["Dockerfile","requirements-runtime.txt",*runtime]
        for relative in context_files:
            source = ROOT/relative
            if source.is_symlink():
                raise RuntimeError("Build context symlink is forbidden")
            shutil.copyfile(source,self.context/relative)
        self.data["source_sha256"] = {name:sha(self.context/name) for name in context_files}
        self.data["context_bytes"] = sum((self.context/name).stat().st_size for name in context_files)
        self.data["context_file_count"] = len(context_files)
        self.root_tests = sorted(p.name for p in ROOT.glob("*_test.py") if p.name not in ("integration_test.py","smoke_test.py"))
        if "result_constraints.py" not in runtime:
            self.root_tests.remove("result_constraints_test.py")
            self.data["excluded_host_only_tests"] = ["result_constraints_test.py: module is not deployed in the image"]
        for relative in [*self.root_tests,"integration_test.py","smoke_test.py",
                         "experiments/deployment_hardening_test.py", "experiments/abstention_hardening_test.py",
                         "experiments/math_parser_hardening_test.py", "experiments/math_parser_hardening.patch"]:
            self.copy_fixture(relative)
        self.copy_fixture("TASK/task-dir")
        # Only explicit TRAIN and visible forecast parity fixtures, never outcomes/oracles.
        train = "experiments/suite_policy/tasks/policy_62ab3bfa6706510a"
        if (ROOT/train).is_dir():
            self.copy_fixture(train)
        parity = ["experiments/runs/2026-11-13-residual-v1/"+name for name in
                  ("freeze_record.json","residual_artifact.json","predictions_synthetic.json","predictions_public100.json")]
        parity += ["experiments/runs/2026-11-12-policy/test/forecasts.json", "experiments/runs/2026-11-12-public100/test/forecasts.json"]
        for relative in parity:
            if (ROOT/relative).is_file():
                self.copy_fixture(relative)
        self.data["test_sha256"] = {p.relative_to(self.fixtures).as_posix():sha(p)
                                    for p in self.fixtures.rglob("*") if p.is_file()}
        self.save()

    def scan(self):
        result = subprocess.run(["git","ls-files","-c","-o","--exclude-standard","-z"],cwd=ROOT,env=self.env,
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        names = sorted(set(x.decode("utf-8") for x in result.stdout.split(b"\0") if x))
        patterns = [re.compile(rb"\bsk-[A-Za-z0-9_-]{16,}"),re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
                    re.compile(rb"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}")]
        matches, total, count = [],0,0
        for relative in names:
            path = ROOT/relative
            if not path.is_file() or path.is_symlink():
                continue
            content = path.read_bytes()
            total += len(content); count += 1
            if any(pattern.search(content) for pattern in patterns):
                matches.append(relative)
        self.data["eligible_repository"] = {"file_count":count,"bytes":total,"below_100_MB":total<100_000_000,
                                             "secret_candidate_paths":matches,"secret_scan_emits_values":False}
        for relative in matches:
            print(relative,flush=True)
        self.save()
        if matches:
            raise RuntimeError("Potential credentials found; only paths recorded")
        if total>=100_000_000:
            raise RuntimeError("Git-eligible footprint exceeds100MB")

    def run(self):
        try:
            self.prepare()
            self.scan()
            self.command("build",["docker","build","--progress","plain","--tag",TAG,str(self.context)],timeout=1000)
            self.data["image_id"] = self.command("image_id",["docker","image","inspect",TAG,"--format","{{.Id}}"])
            self.data["image_size_docker_images"] = self.command("image_size",["docker","images",TAG,"--format","{{.Size}}"])
            self.data["image_size_inspect_bytes"] = int(self.command("image_size_inspect",["docker","image","inspect",TAG,"--format","{{.Size}}"] ))
            paths = {name:"/app/"+name for name in self.runtime}
            paths["requirements-runtime.txt"] = "/app/requirements.txt"
            code = "import hashlib,json,pathlib;paths="+repr(paths)+";print(json.dumps({k:hashlib.sha256(pathlib.Path(v).read_bytes()).hexdigest() for k,v in paths.items()},sort_keys=True))"
            actual = json.loads(self.command("image_hashes",self.container("python","-B","-c",code)))
            self.data["image_sha256"] = actual
            self.data["host_image_hashes_match"] = all(actual[name]==self.data["source_sha256"][name] for name in actual)
            if not self.data["host_image_hashes_match"]:
                raise RuntimeError("Host/image runtime hash mismatch")
            self.command("runtime_units",self.container("python","-B","-m","unittest","-v",
                *[Path(name).stem for name in self.root_tests],fixture_files=self.root_tests,fixture_dirs=("experiments",)),timeout=600)
            self.command("runtime_hardening",self.container("python","-B","-m","unittest","-v",
                "experiments.deployment_hardening_test","experiments.abstention_hardening_test","experiments.math_parser_hardening_test",
                fixture_dirs=("experiments","TASK")),timeout=600)
            self.command("smoke_eight",self.container("python","-B","/app/smoke_test.py",
                fixture_files=("smoke_test.py",),fixture_dirs=("TASK",)),timeout=180)
            self.command("real_sdk_fake_api",self.container("python","-B","/app/integration_test.py",fixture_files=("integration_test.py",)),timeout=180)
            self.data["working_sources_still_match_snapshot"] = all(sha(ROOT/name)==value for name,value in self.data["source_sha256"].items())
            self.data["status"] = "success"
        except Exception as exc:
            self.data["status"] = "failed"
            self.data["failure"] = str(exc)
        finally:
            for index,name in enumerate(self.container_names):
                if not re.fullmatch(r"nsrc-ready-check-[a-z0-9-]+",name):
                    raise RuntimeError("Invalid owned container cleanup target")
                output = self.command("owned_container_check_"+str(index),["docker","container","inspect",name,"--format","{{.State.Running}}"],required=False)
                if self.data["commands"]["owned_container_check_"+str(index)]["returncode"] == 0:
                    self.command("owned_container_cleanup_"+str(index),["docker","container","rm","--force",name],required=False)
            if "protected_before" in self.data:
                self.data["protected_after"] = {tag:self.command("protected_after_"+str(i),["docker","image","inspect",tag,"--format","{{.Id}}"],required=False)
                                                for i,tag in enumerate(PROTECTED)}
                self.data["protected_tags_preserved"] = self.data["protected_before"] == self.data["protected_after"]
                after = self.command("builder_after",["docker","buildx","inspect"],required=False)
                def selected_name(text):
                    found = re.search(r"^Name:\s+(.+)$",text,re.M)
                    return found.group(1).strip() if found else None
                self.data["default_builder_preserved"] = selected_name(self.data["default_builder_before"]) == selected_name(after)
                if not self.data["protected_tags_preserved"] or not self.data["default_builder_preserved"]:
                    self.data["status"] = "failed"
                    self.data["failure"] = "Protected image tag or selected builder changed during validation"
            self.data["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            self.save()
            print(json.dumps({"status":self.data["status"],"report":str(self.folder/REPORT_NAME),"failure":self.data.get("failure")}),flush=True)
        return 0 if self.data["status"]=="success" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    release = parser.add_mutually_exclusive_group()
    release.add_argument("--final", action="store_true",
                         help="Build the separate final tag/report and protect the ready tag")
    release.add_argument("--sqlguards", action="store_true",
                         help="Build the separate SQL-guard tag/report and protect ready/final tags")
    args = parser.parse_args()
    if args.sqlguards:
        PROTECTED = (*PROTECTED,TAG,"nsrc-policy-v3-final:20261113")
        TAG = "nsrc-policy-v3-sqlguards:20261113"
        RUN = ROOT / "experiments/runs/2026-11-13-sqlguards"
        REPORT_NAME = "sqlguards_image_validation.json"
    elif args.final:
        PROTECTED = (*PROTECTED,TAG)
        TAG = "nsrc-policy-v3-final:20261113"
        RUN = ROOT / "experiments/runs/2026-11-13-final"
        REPORT_NAME = "final_image_validation.json"
    raise SystemExit(Validation().run())
