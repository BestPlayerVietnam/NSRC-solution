"""Isolated cold-build QA of the frozen fitted runtime; never calls an LLM.

The fresh docker-container builder is NOT selected with --use. No default
cache is pruned, no existing tag is overwritten, and cleanup targets only the
unique builder created by this invocation. Bootstrap is timed separately.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = ROOT / 'experiments/runs/2026-11-12-policy/cold_build'
PROTOCOL = ROOT / 'experiments/runs/2026-11-12-policy/test/protocol.json'
RUNTIME = (
    'run_agent.py', 'planner.py', 'budget_runtime.py', 'analytical_tools.py',
    'coding_tools.py', 'context_tools.py', 'feasibility.py', 'policy_features.py',
    'retry_policy.py', 'cost_model.py', 'forecast_guard.py', 'cost_artifact.json',
)
CONTEXT_FILES = ('Dockerfile', 'requirements-runtime.txt', *RUNTIME)
DEFAULT_FITTED_TAG = 'nsrc-policy-v3-fitted:20261112'
BUILD_TIMEOUT = 1000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def clean_environment():
    env = dict(os.environ)
    # In particular, never forward provider credentials as build args, context,
    # container environment, subprocess logs or report fields.
    for name in list(env):
        upper = name.upper()
        if upper.startswith(('LLM_', 'NEURALDEEP_', 'OPENAI_', 'ANTHROPIC_')):
            env.pop(name)
    return env


class Check:
    def __init__(self, fitted_tag, source_context=None, bootstrap_timeout=300):
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.token = stamp.lower() + '-' + uuid.uuid4().hex[:12]
        self.builder = 'nsrc-cold-' + self.token
        self.tag = 'nsrc-cold-validation:' + self.token
        self.fitted_tag = fitted_tag
        self.source_root = Path(source_context).resolve() if source_context else ROOT.resolve()
        self.bootstrap_timeout = bootstrap_timeout
        self.folder = RESULT_ROOT / self.token
        self.context = ROOT / '.tools' / ('cold-build-context-' + self.token)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.context.mkdir(parents=True, exist_ok=False)
        self.env = clean_environment()
        self.creation_attempted = False
        self.started = time.monotonic()
        self.data = {
            'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'builder': self.builder, 'builder_driver': 'docker-container',
            'qa_image_tag': self.tag, 'fitted_image_tag': fitted_tag,
            'context': str(self.context), 'result_directory': str(self.folder),
            'source_context': str(self.source_root),
            'bootstrap_timeout_seconds': self.bootstrap_timeout,
            'build_timeout_seconds': BUILD_TIMEOUT, 'api_calls': 0,
            'selected_default_builder_changed_by_command': False,
            'cache_prune_commands': [], 'commands': {}, 'status': 'initializing',
            'limitations': [
                'Fresh application BuildKit state; registry/CDN caches are outside local control.',
                'BuildKit daemon bootstrap is separate from the1000-second application build deadline.',
                'The Docker engine may reuse a partial or complete BuildKit daemon-image pull; this is not application-layer cache in the new builder.',
                'Two LLM pipelines may be active concurrently; one timed result is not a guarantee on other hosts/networks.',
            ],
        }
        self.save()

    def save(self):
        self.data['elapsed_total_seconds'] = round(time.monotonic() - self.started, 3)
        write_json(self.folder / 'result.json', self.data)

    def command(self, stage, arguments, timeout=120, required=True):
        log = self.folder / (stage + '.log')
        start = time.monotonic()
        record = {'argv': arguments, 'timeout_seconds': timeout, 'log': str(log),
                  'started_at': dt.datetime.now(dt.timezone.utc).isoformat()}
        self.data['commands'][stage] = record
        self.data['active_stage'] = stage
        self.save()
        print(json.dumps({'stage': stage, 'state': 'running', 'log': str(log)}), flush=True)
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        with log.open('wb') as output:
            process = subprocess.Popen(arguments, cwd=ROOT, env=self.env,
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                creationflags=flags)
            deadline = start + timeout
            record['owned_process_id'] = process.pid
            timed_out = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    record['timeout_triggered_after_seconds'] = round(time.monotonic() - start, 3)
                    # The PID comes from our own still-running Popen object.
                    # Windows Docker invokes a buildx child, so stop that tree too.
                    if process.poll() is None:
                        if os.name == 'nt':
                            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL, timeout=20, creationflags=flags)
                        else:
                            process.kill()
                    try: process.wait(timeout=20)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
                    break
                try:
                    process.wait(timeout=min(20, remaining))
                    break
                except subprocess.TimeoutExpired:
                    print(json.dumps({'stage':stage, 'elapsed_seconds':round(time.monotonic()-start,1),
                        'log_bytes':log.stat().st_size}), flush=True)
        record.update(returncode=process.returncode, timed_out=timed_out,
            elapsed_seconds=round(time.monotonic()-start,3))
        self.save()
        print(json.dumps({'stage':stage, 'returncode':record['returncode'],
            'timed_out':timed_out, 'elapsed_seconds':record['elapsed_seconds']}),flush=True)
        if required and (timed_out or process.returncode):
            raise RuntimeError('Command failed or timed out: ' + stage)
        return record

    def output(self, stage):
        return (self.folder / (stage + '.log')).read_text(encoding='utf-8', errors='replace').strip()

    def image_id(self, stage, tag, required=True):
        record=self.command(stage,['docker','image','inspect',tag,'--format','{{.Id}}'],required=required)
        return self.output(stage) if record['returncode']==0 else None

    def runtime_hashes(self, stage, image):
        mapping={name:'/app/'+name for name in RUNTIME}
        mapping['requirements-runtime.txt']='/app/requirements.txt'
        program='import hashlib,json,pathlib;paths='+repr(mapping)+';print(json.dumps({k:hashlib.sha256(pathlib.Path(v).read_bytes()).hexdigest() for k,v in paths.items()},sort_keys=True))'
        self.command(stage,['docker','run','--rm','--network','none','--read-only',
            '--cpus','1','--memory','256m','--pids-limit','32','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--entrypoint','python',image,'-B','-c',program],timeout=90)
        return json.loads(self.output(stage).splitlines()[-1])

    def prepare(self):
        if not self.source_root.is_relative_to(ROOT.resolve()) or not self.source_root.is_dir():
            raise RuntimeError('Source context must be an existing directory inside this workspace')
        self.command('docker_version',['docker','version'],timeout=60)
        self.command('buildx_version',['docker','buildx','version'],timeout=60)
        self.command('default_builder_before',['docker','buildx','inspect'],timeout=60)
        match=re.search(r'^Name:\s*(\S+)',self.output('default_builder_before'),re.M)
        if not match:raise RuntimeError('Cannot identify the current builder')
        self.data['default_builder_before']=match.group(1)
        self.data['protected_tags_before']={
            self.fitted_tag:self.image_id('fitted_before',self.fitted_tag),
            'nsrc-harness:latest':self.image_id('latest_before','nsrc-harness:latest',required=False),
        }
        frozen=json.loads(PROTOCOL.read_text(encoding='utf-8-sig'))
        expected={arm['image'] for arm in frozen['arms'].values()}
        fitted=self.data['protected_tags_before'][self.fitted_tag]
        if expected!={fitted}:raise RuntimeError('Fitted image tag differs from the frozen test protocol')
        self.data['frozen_protocol_sha256']=sha(PROTOCOL)
        self.data['context_sha256']={}
        for name in CONTEXT_FILES:
            source=self.source_root/name
            if source.is_symlink() or not source.resolve().is_relative_to(self.source_root):
                raise RuntimeError('Unsafe source path: '+name)
            if not source.is_file():raise FileNotFoundError(source)
            shutil.copyfile(source,self.context/name)
            self.data['context_sha256'][name]=sha(self.context/name)
        if {p.name for p in self.context.iterdir()}!=set(CONTEXT_FILES):
            raise RuntimeError('Unexpected build context entry')
        self.data['context_file_count']=len(CONTEXT_FILES)
        self.data['context_bytes']=sum(p.stat().st_size for p in self.context.iterdir())
        image_hashes=self.runtime_hashes('frozen_runtime_hashes',fitted)
        mismatches=[name for name,value in image_hashes.items() if self.data['context_sha256'][name]!=value]
        self.data['frozen_runtime_hashes']=image_hashes
        self.data['frozen_source_mismatches']=mismatches
        if mismatches:raise RuntimeError('Current sources do not match the frozen fitted image: '+','.join(mismatches))
        # Never adopt or remove a pre-existing builder, even under our prefix.
        existing=self.command('own_builder_precheck',['docker','buildx','inspect',self.builder],required=False)
        if existing['returncode']==0:raise RuntimeError('Unique builder unexpectedly already exists')
        if existing['timed_out']:raise RuntimeError('Unable to establish builder ownership')
        self.save()

    def build(self):
        self.data['status']='setting_up_builder';self.save()
        self.creation_attempted=True
        self.command('builder_create',['docker','buildx','create','--name',self.builder,
            '--driver','docker-container'],timeout=120)
        self.command('builder_bootstrap',['docker','buildx','inspect',self.builder,'--bootstrap'],timeout=self.bootstrap_timeout)
        self.data['status']='building';self.save()
        result=self.command('cold_build',['docker','buildx','build','--builder',self.builder,
            '--pull','--no-cache','--load','--progress','plain','--metadata-file',str(self.folder/'build_metadata.json'),
            '--tag',self.tag,'--file',str(self.context/'Dockerfile'),str(self.context)],
            timeout=BUILD_TIMEOUT,required=False)
        self.data['build_elapsed_seconds']=result['elapsed_seconds']
        self.data['build_within_1000_seconds']=not result['timed_out'] and result['elapsed_seconds']<=BUILD_TIMEOUT
        if result['timed_out']:
            self.data['status']='build_timeout';return
        if result['returncode']:
            self.data['status']='build_failed';return
        self.data['qa_image_id']=self.image_id('qa_image_id',self.tag)
        self.command('qa_image_size_inspect',['docker','image','inspect',self.tag,'--format','{{.Size}}'])
        self.command('qa_image_size_cli',['docker','images',self.tag,'--format','{{.Size}}'])
        self.data['image_size_inspect_bytes']=int(self.output('qa_image_size_inspect'))
        self.data['image_size_docker_images']=self.output('qa_image_size_cli')
        self.data['qa_runtime_hashes']=self.runtime_hashes('qa_runtime_hashes',self.data['qa_image_id'])
        self.data['qa_runtime_matches_frozen']=self.data['qa_runtime_hashes']==self.data['frozen_runtime_hashes']
        if not self.data['qa_runtime_matches_frozen']:raise RuntimeError('QA runtime differs from the frozen image')
        self.data['status']='success'

    def cleanup(self):
        if self.creation_attempted:
            if not re.fullmatch(r'nsrc-cold-\d{8}t\d{6}z-[0-9a-f]{12}',self.builder) or self.builder!='nsrc-cold-'+self.token:
                raise RuntimeError('Refusing cleanup of an unowned builder name')
            record=self.command('builder_cleanup',['docker','buildx','rm','--force',self.builder],timeout=120,required=False)
            self.data['builder_cleanup_success']=record['returncode']==0 and not record['timed_out']
        else:
            self.data['builder_cleanup_success']=None
        record=self.command('default_builder_after',['docker','buildx','inspect'],timeout=60,required=False)
        match=re.search(r'^Name:\s*(\S+)',self.output('default_builder_after'),re.M)
        self.data['default_builder_after']=match.group(1) if match and record['returncode']==0 else None
        before=self.data.get('default_builder_before');after=self.data['default_builder_after']
        self.data['default_builder_preserved']=(before==after) if before is not None and after is not None else None
        self.data['protected_tags_after']={}
        for i,tag in enumerate(self.data.get('protected_tags_before',{})):
            self.data['protected_tags_after'][tag]=self.image_id('protected_after_'+str(i),tag,required=False)
        self.data['protected_tags_preserved']=(self.data['protected_tags_before']==self.data['protected_tags_after']) if 'protected_tags_before' in self.data else None
        self.save()

    def run(self):
        try:
            self.prepare()
            self.build()
        except Exception as exc:
            self.data['failure']={'type':type(exc).__name__,'message':str(exc)}
            if self.data['status'] not in ('build_timeout','build_failed'):
                self.data['status']='setup_or_validation_failed'
        finally:
            try:self.cleanup()
            except Exception as exc:self.data['cleanup_failure']={'type':type(exc).__name__,'message':str(exc)}
            self.data['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat()
            self.save()
            print(json.dumps({key:self.data.get(key) for key in (
                'status','builder','builder_cleanup_success','qa_image_tag','build_elapsed_seconds',
                'build_within_1000_seconds','image_size_docker_images','default_builder_preserved',
                'protected_tags_preserved','result_directory','failure','cleanup_failure')},ensure_ascii=False),flush=True)
        return 0 if self.data['status']=='success' and self.data.get('builder_cleanup_success') else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fitted-image',default=DEFAULT_FITTED_TAG)
    parser.add_argument('--source-context',type=Path,help='Reuse a verified frozen context after the working tree has changed')
    parser.add_argument('--bootstrap-timeout',type=int,default=300,help='Separate BuildKit setup timeout; application build remains1000 seconds')
    args=parser.parse_args()
    if not 30<=args.bootstrap_timeout<=3600:parser.error('bootstrap timeout must be30..3600 seconds')
    raise SystemExit(Check(args.fitted_image,args.source_context,args.bootstrap_timeout).run())
