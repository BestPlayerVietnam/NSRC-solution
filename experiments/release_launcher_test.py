"""Offline release-template and Windows double-click launcher regressions."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import release_check


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('git'), 'Git is required for packaging checks')
class EnvironmentTemplateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        sources = ['requirements-runtime.txt', 'run_agent.py',
                   'cost_artifact.json', 'residual_cost_artifact.json']
        dockerfile = 'FROM python:3.11\n' + ''.join(
            'COPY ' + name + ' /app/' + name + '\n' for name in sources)
        (self.project / 'Dockerfile').write_text(dockerfile, encoding='utf-8')
        (self.project / '.dockerignore').write_text(
            '**\n!Dockerfile\n!.dockerignore\n' + ''.join('!' + name + '\n' for name in sources),
            encoding='utf-8')
        (self.project / 'requirements-runtime.txt').write_text('', encoding='utf-8')
        (self.project / 'run_agent.py').write_text('pass\n', encoding='utf-8')
        for name in ['cost_artifact.json', 'residual_cost_artifact.json']:
            (self.project / name).write_text('{}\n', encoding='utf-8')
        shutil.copyfile(ROOT / '.env.example', self.project / '.env.example')
        subprocess.run(['git', 'init', '--initial-branch=main'], cwd=self.project,
                       stdin=subprocess.DEVNULL, capture_output=True, check=True)

    def packaging(self):
        output = io.StringIO()
        with patch.object(release_check, 'ROOT', self.project), \
                redirect_stdout(output), redirect_stderr(output):
            sources = release_check.packaging_check()
        return sources, output.getvalue()

    def assert_rejected_without_value(self, secret):
        output = io.StringIO()
        with patch.object(release_check, 'ROOT', self.project), \
                redirect_stdout(output), redirect_stderr(output):
            with self.assertRaises(RuntimeError) as caught:
                release_check.packaging_check()
        text = output.getvalue() + str(caught.exception)
        self.assertIn('Potential credentials', text)
        self.assertFalse(secret in text, 'Credential value reached packaging output')

    def test_root_placeholder_template_is_git_eligible_and_accepted(self):
        sources, output = self.packaging()
        self.assertEqual(len(sources), 4)
        report = json.loads(output)
        self.assertEqual(report['packaging'], 'passed')
        self.assertEqual(report['git_eligible_files'], 7)
        self.assertEqual(report['secret_candidate_paths'], [])

    def test_template_secret_pattern_is_still_rejected_without_value(self):
        secret = 'sk-' + 'x' * 24
        (self.project / '.env.example').write_text(
            "LLM_API_KEY='" + secret + "'\n", encoding='utf-8')
        self.assert_rejected_without_value(secret)

    def test_private_environment_is_rejected_even_with_placeholder_value(self):
        secret = 'private-local-placeholder'
        for name in ['.env', 'private.env']:
            with self.subTest(name=name):
                path = self.project / name
                path.write_text("LLM_API_KEY='" + secret + "'\n", encoding='utf-8')
                try:
                    self.assert_rejected_without_value(secret)
                finally:
                    path.unlink()


@unittest.skipUnless(shutil.which('docker'), 'Docker CLI is required for Compose configuration')
class ComposeConfigurationTests(unittest.TestCase):
    def test_budget_environment_and_mount_contract_without_daemon(self):
        docker = shutil.which('docker')
        version = subprocess.run([docker, 'compose', 'version'], stdin=subprocess.DEVNULL,
                                 capture_output=True, timeout=30)
        if version.returncode:
            self.skipTest('Docker Compose plugin is unavailable')
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / 'Конфигурация с пробелами'
            tasks = project / 'Задачи с пробелами'
            tasks.mkdir(parents=True)
            compose = project / 'compose.yaml'
            shutil.copyfile(ROOT / 'compose.yaml', compose)
            self.assertIn('create_host_path: false', compose.read_text(encoding='utf-8'))
            environment = {key: value for key, value in os.environ.items()
                           if not key.upper().startswith(
                               ('LLM_', 'NEURALDEEP_', 'OPENAI_', 'ANTHROPIC_', 'COMPOSE_'))
                           and key.upper() not in {'BUDGET', 'TASKS_DIR'}}
            environment.update(LLM_BASE_URL='https://offline.invalid/v1',
                               LLM_API_KEY='compose-configuration-placeholder',
                               TASKS_DIR=tasks.as_posix())
            for label, value, expected in [('unset', None, '100000'),
                                           ('decimal', '5000.5', '5000.5'),
                                           ('empty', '', '')]:
                with self.subTest(budget=label):
                    configured = dict(environment)
                    if value is not None:
                        configured['BUDGET'] = value
                    result = subprocess.run(
                        [docker, 'compose', '--file', str(compose), 'config', '--format', 'json'],
                        cwd=project, env=configured, stdin=subprocess.DEVNULL,
                        capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, 'Compose configuration failed')
                    try:
                        service = json.loads(result.stdout)['services']['agent']
                    except (ValueError, KeyError, TypeError):
                        self.fail('Compose did not return the expected agent configuration')
                    self.assertEqual(service['environment']['BUDGET'], expected)
                    self.assertEqual(service['environment']['TRACE_FILE'],
                                     '/data/tasks/nsrc-trajectory.jsonl')
                    self.assertEqual(service['environment']['HOME'], '/tmp')
                    self.assertEqual(service['cpus'], 4)
                    self.assertEqual(int(service['mem_limit']), 16 * 1024 ** 3)
                    self.assertEqual(service['pids_limit'], 512)
                    self.assertIs(service['read_only'], True)
                    self.assertIn('/tmp:rw,size=2g', service['tmpfs'])
                    self.assertIn('ALL', service['cap_drop'])
                    mounts = [mount for mount in service['volumes']
                              if mount['target'] == '/data/tasks']
                    self.assertEqual(len(mounts), 1)
                    self.assertEqual(Path(mounts[0]['source']).resolve(), tasks.resolve())
                    self.assertEqual(mounts[0]['type'], 'bind')
                    # Compose omits this false flag from its resolved JSON.
                    self.assertIs(mounts[0].get('bind', {}).get('create_host_path', False), False)


@unittest.skipUnless(os.name == 'nt', 'Windows batch/PowerShell launcher checks')
class WindowsLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'Релиз с пробелами'
        (self.project / 'scripts').mkdir(parents=True)
        for name in ['start.cmd', 'compose.yaml', '.env.example', 'scripts/start_release.ps1']:
            shutil.copyfile(ROOT / name, self.project / name)
        self.tasks = self.project / 'Задачи с пробелами'
        task = self.tasks / 'task_001'
        task.mkdir(parents=True)
        (task / 'task.json').write_text('{}\n', encoding='utf-8')
        self.bin = self.root / 'mock docker'
        self.bin.mkdir()
        self.configuration = self.root / 'resolved configuration.json'
        self.run_marker = self.root / 'run marker.txt'
        self.call_log = self.root / 'docker calls.txt'
        self.secret = 'sk-' + 'z' * 24
        command = (
            '@echo off\n'
            'echo %*>>"%FAKE_DOCKER_CALL_LOG%"\n'
            'if "%~1"=="info" (\n'
            '  echo Harmless Docker information warning 1>&2\n'
            '  exit /b 0\n'
            ')\n'
            'if "%~1"=="compose" (\n'
            '  if "%~4"=="config" (\n'
            '    type "%FAKE_DOCKER_CONFIG%"\n'
            '    exit /b 0\n'
            '  )\n'
            '  if "%~4"=="run" (\n'
            '    echo Harmless Compose run warning 1>&2\n'
            '    echo %*>"%FAKE_DOCKER_RUN_MARKER%"\n'
            '    exit /b 7\n'
            '  )\n'
            ')\n'
            'exit /b 91\n'
        )
        (self.bin / 'docker.cmd').write_bytes(command.replace('\n', '\r\n').encode('ascii'))
        self.write_configuration(self.tasks)

    def write_configuration(self, tasks):
        configuration = {'services': {'agent': {
            'environment': {'LLM_API_KEY': self.secret, 'BUDGET': '5000.5'},
            'volumes': [{'type': 'bind', 'source': str(tasks), 'target': '/data/tasks'}],
        }}}
        self.configuration.write_text(json.dumps(configuration, ensure_ascii=False), encoding='utf-8')

    def launch(self, *, configured=True):
        if configured:
            (self.project / '.env').write_text("LLM_API_KEY='local-placeholder'\n", encoding='utf-8')
        environment = {key: value for key, value in os.environ.items()
                       if not key.upper().startswith(('LLM_', 'NEURALDEEP_', 'OPENAI_', 'ANTHROPIC_'))
                       and key.upper() not in {'BUDGET', 'TASKS_DIR'}}
        environment.update(
            PATH=str(self.bin) + os.pathsep + environment.get('PATH', ''),
            FAKE_DOCKER_CONFIG=str(self.configuration),
            FAKE_DOCKER_RUN_MARKER=str(self.run_marker),
            FAKE_DOCKER_CALL_LOG=str(self.call_log),
        )
        result = subprocess.run(['cmd.exe', '/d', '/c', 'start.cmd'], cwd=self.project,
                                env=environment, stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=30)
        text = (result.stdout + result.stderr).decode('utf-8', errors='replace')
        self.assertFalse(self.secret in text, 'Resolved API key reached launcher output')
        return result, text

    def test_batch_launch_preserves_compose_exit_code_in_unicode_project(self):
        result, _ = self.launch()
        self.assertEqual(result.returncode, 7)
        self.assertTrue(self.run_marker.is_file())
        command = self.run_marker.read_text(encoding='utf-8', errors='replace')
        self.assertIn('run --build --rm agent', command)

    def test_missing_task_directory_fails_before_compose_run(self):
        missing = self.project / 'Несуществующие задачи'
        self.write_configuration(missing)
        result, text = self.launch()
        self.assertEqual(result.returncode, 1)
        self.assertIn('TASKS_DIR must point to an existing directory', text)
        self.assertFalse(missing.exists())
        self.assertFalse(self.run_marker.exists())

    def test_first_launch_creates_template_and_exits_without_run(self):
        result, text = self.launch(configured=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Created .env.', text)
        self.assertEqual((self.project / '.env').read_bytes(),
                         (self.project / '.env.example').read_bytes())
        self.assertFalse(self.run_marker.exists())
        log = self.call_log.read_text(encoding='utf-8', errors='replace')
        self.assertNotIn('compose', log)


if __name__ == '__main__':
    unittest.main()
