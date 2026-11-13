"""Bounded, pre-execution features. No answers, tests execution or model calls."""
from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

from analytical_tools import CSVDatabase
from coding_tools import repair_snapshot, repair_context
from context_tools import retrieve_context

POLICY_ID = 'production-cpu-v3-fixed'


def extract_features(task: dict, workspace: Path) -> dict[str, float]:
    instruction = str(task.get('instruction', ''))
    kind = str(task.get('task_type', ''))
    f = {f'kind_{k}': float(kind == k) for k in ('math', 'coding', 'search', 'data_analysis')}
    f.update(instruction_chars=float(len(instruction)), context_chars=0.0,
             source_bytes=0.0, test_bytes=0.0, file_count=0.0, assert_count=0.0,
             numeric_count=float(len(re.findall(r'\b\d+(?:\.\d+)?\b', instruction))),
             operators=float(len(re.findall(r'[+*/%]|\b(?:per|each|average|ratio|percent|difference)\b', instruction, re.I))),
             contract_flag=float(bool(re.search(r'\b(?:stateless|deterministic)\b', instruction, re.I))),
             contradiction_flag=0.0, missing_evidence_flag=0.0, context_partial=0.0,
             legacy_raw_cost=0.0)
    if kind == 'coding':
        try:
            snap = repair_snapshot(workspace, instruction)
            f['source_bytes'] = float(sum(len(v or b'') for v in snap.values()))
            f['file_count'] = float(len(snap))
            f['context_chars'] = float(len(repair_context(workspace, snap, instruction=instruction)))
        except (OSError, ValueError, UnicodeError):
            f['context_partial'] = 1.0
        total = 0
        for p in sorted(workspace.rglob('test_*.py'))[:80]:
            if p.is_symlink():
                continue
            try:
                size = p.stat().st_size
                if total + size > 500_000:
                    f['context_partial'] = 1.0
                    break
                total += size
                text = p.read_text(encoding='utf-8')
                tree = ast.parse(text)
                f['assert_count'] += sum(isinstance(n, ast.Assert) or
                    (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr.startswith('assert')) for n in ast.walk(tree))
            except (OSError, UnicodeError, SyntaxError):
                f['context_partial'] = 1.0
        f['test_bytes'] = float(total)
        f['legacy_raw_cost'] = 65 + len(instruction)/40 + (f['source_bytes']+total)/40 + min(2500, f['source_bytes']/2.5) + min(150, f['source_bytes']/10)
    elif kind == 'search':
        context = retrieve_context(workspace, instruction)
        f['context_chars'] = float(len(context.text))
        f['file_count'] = float(len(context.snippets))
        f['context_partial'] = float(not context.complete)
        f['missing_evidence_flag'] = float(not context.snippets and context.complete)
        f['legacy_raw_cost'] = 25 + (len(instruction)+len(context.text))/40
    elif kind == 'data_analysis':
        try:
            db = CSVDatabase(workspace)
            try:
                f['context_chars'] = float(len(db.prompt()))
            finally:
                db.close()
        except (OSError, ValueError, UnicodeError, csv.Error):
            f['context_partial'] = 1.0
        f['file_count'] = float(sum(1 for p in workspace.rglob('*.csv') if not p.is_symlink()))
        f['legacy_raw_cost'] = 90 + len(instruction)/40 + 2.5
    else:
        f['legacy_raw_cost'] = 20 + len(instruction)/40
    from feasibility import analyze_task
    risk = analyze_task(task, workspace)
    f['contradiction_flag'] = float('conflicting_outputs' in risk.flags or risk.hard_impossible)
    return f


def fallback_cost(features: dict, model: str) -> tuple[float, float]:
    """Prompt-aware prior; empirical forecasts can replace it per policy/model."""
    prompt = (features.get('instruction_chars', 0) + features.get('context_chars', 0)) / 35 + 18
    if features.get('kind_coding'):
        output = 85 + min(1400, features.get('source_bytes', 0)/6)
        retry = 0 if features.get('contradiction_flag') else 0.2
    elif features.get('kind_data_analysis'):
        output, retry = 40, 0.12
    elif features.get('kind_search'):
        output, retry = 8, 0.05
    else:
        output, retry = 24, 0.06
    first = prompt + output * (5 if not model.endswith('-noreason') else 1)
    return first, first + retry * (prompt + 650)
