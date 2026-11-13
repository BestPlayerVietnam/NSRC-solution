"""Build a fresh 100-task PUBLIC heldout panel; never fit on this panel.

80 public-source problems and 20 authored, certified negatives. Reference code
executes only in networkless Docker. Downloading is development-only and opt-in.
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".tools" / "dataset-cache"
OUT = ROOT / "experiments" / "suite_public100"
SEED = 20261113207
DOMAINS = ("math", "coding", "search", "data_analysis")
OLD_SUITES = ("suite", "suite_fresh", "suite_production", "suite_policy")
ASSETS = {
    "gsm8k": ("gsm8k_train.jsonl", "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl", "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465", "MIT", "https://raw.githubusercontent.com/openai/grade-school-math/master/LICENSE"),
    "humaneval": ("public100/HumanEval.jsonl.gz", "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz", "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef", "MIT", "https://raw.githubusercontent.com/openai/human-eval/master/LICENSE"),
    "squad2_train": ("public100/squad_train_v2.json", "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v2.0.json", "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002", "CC-BY-SA-4.0", "https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt"),
    "wikisql": ("wikisql_data.tar.bz2", "https://raw.githubusercontent.com/salesforce/WikiSQL/master/data.tar.bz2", "755c728ab188e364575705c8641f3fafd86fb089cb8b08e8c03f01832aae0881", "BSD-3-Clause", "https://raw.githubusercontent.com/salesforce/WikiSQL/master/LICENSE"),
}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def digest(value):
    if isinstance(value, str): value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def norm(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def scalar(value):
    if isinstance(value, float) and value.is_integer(): return str(int(value))
    return str(value)


def safe(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or ":" in name: raise ValueError("Unsafe relative path")
    return path


def text(source):
    return textwrap.dedent(source).strip() + "\n"


def fetch(path, url, download):
    if path.is_file(): return
    if not download: raise FileNotFoundError(f"Missing {path}; rerun with --download")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exact official URLs only; no archive extraction and no downloaded execution.
    request = urllib.request.Request(url, headers={"User-Agent": "NSRC-public-evaluation-builder/1.0", "Accept-Encoding":"gzip"})
    print(json.dumps({'downloading':url,'destination':str(path)}),flush=True)
    started=time.monotonic();chunks=[];size=0;logged=0
    with urllib.request.urlopen(request, timeout=30) as response:
        encoding=response.headers.get('Content-Encoding','')
        print(json.dumps({'transfer_encoding':encoding,'content_length':response.headers.get('Content-Length')}),flush=True)
        while True:
            chunk=response.read1(64*1024)
            if not chunk:break
            chunks.append(chunk);size+=len(chunk)
            if size//1_000_000>logged:
                logged=size//1_000_000;print(json.dumps({'download_progress':path.name,'transfer_bytes':size}),flush=True)
            if size>100_000_000 or time.monotonic()-started>600:raise ValueError(f'Source download exceeds size/time limit at {size} bytes')
    body=b''.join(chunks)
    if encoding.casefold()=='gzip':body=gzip.decompress(body)
    if len(body)>100_000_000:raise ValueError('Decoded source size')
    with path.open("xb") as stream: stream.write(body)
    print(json.dumps({'downloaded':path.name,'bytes':len(body),'transfer_bytes':size}),flush=True)


def sources(download):
    result = {}
    for name, (filename, url, expected, license_name, license_url) in ASSETS.items():
        path = CACHE / filename
        fetch(path, url, download)
        actual = digest(path.read_bytes())
        if expected and actual != expected: raise ValueError("Source hash mismatch: " + name)
        license_path = CACHE / "public100" / (name + ".LICENSE.txt")
        fetch(license_path, license_url, download)
        result[name] = dict(url=url, source_file=filename, source_sha256=actual,
            bytes=path.stat().st_size, license=license_name, license_url=license_url,
            license_sha256=digest(license_path.read_bytes()), license_file="licenses/"+name+".txt")
    return result


def arithmetic(expression, variables=None):
    """Interpret a deliberately tiny arithmetic AST, never eval source code."""
    variables = variables or {}
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
        if isinstance(node, ast.Name) and node.id in variables: return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand); return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add): value = a + b
            elif isinstance(node.op, ast.Sub): value = a - b
            elif isinstance(node.op, ast.Mult): value = a * b
            elif isinstance(node.op, ast.Div): value = a / b
            elif isinstance(node.op, ast.FloorDiv): value = a // b
            elif isinstance(node.op, ast.Mod): value = a % b
            elif isinstance(node.op, ast.Pow) and abs(b) <= 12: value = a ** b
            else: raise ValueError("Unsupported arithmetic")
            if not math.isfinite(value) or abs(value) > 1e16: raise ValueError("Arithmetic range")
            return value
        raise ValueError("Unsupported arithmetic node")
    return walk(ast.parse(expression, mode="eval").body)


class Exclusions:
    def __init__(self):
        self.groups, self.instructions, self.passages, self.files = set(), set(), set(), set()
        self.snapshots = {}
        for name in OLD_SUITES:
            base = ROOT / "experiments" / name
            manifest_path = base / "manifest.json"
            if not manifest_path.exists(): raise FileNotFoundError("Required exclusion suite: " + name)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            metadata_path = base / "research_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
            self.snapshots[name] = {"manifest_sha256": digest(manifest_path.read_bytes()),
                "metadata_sha256": digest(metadata_path.read_bytes()) if metadata_path.exists() else None}
            for row in manifest["tasks"]:
                group = row.get("group") or metadata.get(row["id"], {}).get("group")
                if group: self.groups.add(group)
                folder = base / row["task_path"]
                task = json.loads((folder/"task.json").read_text(encoding="utf-8-sig"))
                self.instructions.add(norm(task["instruction"]))
                for path in (folder/"workspace").rglob("*"):
                    if path.is_file():
                        body = path.read_text(encoding="utf-8", errors="replace")
                        self.files.add(digest(norm(body)))
                        if path.suffix in (".md", ".txt"): self.passages.add(norm(body))


class Builder:
    def __init__(self, excluded):
        self.excluded = excluded
        self.items, self.groups, self.questions = [], set(), set()

    def add(self, domain, group, question, files, expected, *, source_name, source_id, **oracle):
        assert group not in self.excluded.groups and group not in self.groups, group
        assert norm(question) not in self.excluded.instructions and norm(question) not in self.questions
        ident = "public_" + digest(f"{SEED}:{group}")[:16]
        self.groups.add(group); self.questions.add(norm(question))
        self.items.append(dict(id=ident, domain=domain, group=group, instruction=question,
            files=files, expected=expected, source_name=source_name, source_id=source_id, oracle=oracle))


def math_cases(builder):
    candidates = []
    for row in map(json.loads, (CACHE/ASSETS["gsm8k"][0]).read_text(encoding="utf-8").splitlines()):
        question, solution = row["question"], row["answer"]
        group = "gsm8k:" + digest(question)
        if group in builder.excluded.groups or not 280 <= len(question) <= 2200: continue
        gold = solution.split("####")[-1].strip().replace(",", "")
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", gold): continue
        steps = re.findall(r"<<([^<>]+)>>", solution)
        if not 5 <= len(steps) <= 12: continue
        try:
            for step in steps:
                lhs, rhs = step.rsplit("=", 1)
                if not math.isclose(arithmetic(lhs), float(rhs.replace(",", "")), rel_tol=1e-8, abs_tol=1e-7): raise ValueError("Rounded source step")
            if not math.isclose(float(steps[-1].rsplit("=",1)[1]), float(gold), rel_tol=1e-8, abs_tol=1e-7): continue
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError): continue
        candidates.append((group, question, solution, gold, steps))
    # Prefer genuinely multi-step questions, with deterministic selection within bands.
    candidates.sort(key=lambda r:(0 if len(r[4]) >= 7 else 1, digest(f"{SEED}:{r[0]}")))
    chosen = candidates[:8] + [r for r in candidates if r not in candidates[:8]][:12]
    assert len(chosen) == 20, ("Insufficient multistep math",len(chosen))
    for group,q,solution,gold,steps in chosen:
        builder.add("math",group,q+"\nGive only the final numeric answer. Use the complete stated facts; return UNSOLVABLE only if no unique answer follows.",{},gold,
            source_name="gsm8k",source_id=group,validation_kind="math_source",reference_solution=solution,arithmetic_steps=steps)
    negatives = [
        ("unrecorded_intersection", "Two reading clubs contain42 and37 people. Exactly3 people from each club leave, all six are different and none belonged to both clubs. The overlap between the original memberships was not recorded. Every remaining person receives2 vouchers worth7 cents each. How many cents are distributed in total? The original overlap is an integer between0 and20; no additional facts determine it.", "(42+37-x-6)*2*7", [4,12]),
        ("unknown_conversion", "A laboratory buys8 reagent batches at125 euros each, receives a10 percent discount, then pays30 euros for packaging. It pays an additional local fee of60 crowns. The invoice exchange rate in crowns per euro is not recorded and is independent of all other facts. What is the total payment in crowns?", "(8*125*9/10+30)*x+60", [2,3]),
        ("unidentified_probability", "A trial produces a successful component with an unrecorded probability p, independently on each trial. A success earns18 credits and failure earns2 credits. Four trials are performed and their combined reward is multiplied by3, after which a fixed fee of8 credits is paid. What is the expected net reward? No observations or constraints determine p beyond0<=p<=1.", "4*(18*x+2*(1-x))*3-8", [0.25,0.75]),
        ("unknown_rotation_offset", "A machine displays one of the integers0 through16 on a circular counter. Its initial display was not logged. It advances8 positions per minute for6 minutes, then moves back13 positions. Its final display is multiplied by5 and increased by4. What number is produced? The initial display can be any integer0 through16.", "((x+8*6-13)%17)*5+4", [1,8]),
        ("missing_joint_composition", "A sealed box contains20 beads, each either red or blue. Exactly7 beads are removed without observing their colors. Each remaining red bead is assigned4 points and each remaining blue bead9 points, and the total is doubled before adding11. There is no information about the initial colors or removed colors. What is the final score?", "(4*x+9*(13-x))*2+11", [3,9]),
    ]
    for family,q,expr,values in negatives:
        # Normal prose spacing is deliberate, independent of the certification values.
        q = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", q)
        worlds = [{"x":x} for x in values]
        builder.add("math","public100_math:"+family,q+" Use only these facts. Return a numeric answer or UNSOLVABLE if it is not uniquely determined.",{},None,
            source_name="authored",source_id=family,validation_kind="math_negative",expression=expr,worlds=worlds,
            witness_answers=[arithmetic(expr,w) for w in worlds],reason="The two allowed completions of the unrecorded quantity produce different answers.")


ALLOWED_IMPORTS = {"math","typing","collections","itertools","functools","heapq","bisect","operator","re","string","statistics","fractions","decimal"}
PUBLIC_NAMES = {"candidate","abs","all","any","bool","dict","enumerate","float","frozenset","int","len","list","max","min","range","reversed","round","set","sorted","str","sum","tuple","zip"}


def humaneval_cases(builder):
    rows = [json.loads(line) for line in gzip.decompress((CACHE/ASSETS["humaneval"][0]).read_bytes()).decode().splitlines()]
    candidates = []
    for row in rows:
        group = "humaneval:"+row["task_id"]
        if group in builder.excluded.groups: continue
        complete = row["prompt"]+row["canonical_solution"]
        try:
            tree = ast.parse(complete); test_tree=ast.parse(row["test"])
        except SyntaxError: continue
        nodes=list(ast.walk(tree))
        complexity=sum(isinstance(n,(ast.For,ast.While,ast.If,ast.IfExp,ast.comprehension)) for n in nodes)
        if len(nodes)<85 or complexity<2 or len(row["prompt"])<220 or len(complete)>8500: continue
        if any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(z.split('.')[0] not in ALLOWED_IMPORTS for z in ([n.module or ''] if isinstance(n,ast.ImportFrom) else [a.name for a in n.names])) for n in nodes): continue
        if any(isinstance(n,ast.Name) and n.id in {"eval","exec","compile","open","input","__import__"} for n in nodes): continue
        asserts = [n for n in ast.walk(test_tree) if isinstance(n,ast.Assert)
            and any(isinstance(x,ast.Name) and x.id=="candidate" for x in ast.walk(n))
            and {x.id for x in ast.walk(n) if isinstance(x,ast.Name)} <= PUBLIC_NAMES]
        if not asserts: continue
        candidates.append((row,complete,len(nodes),complexity,asserts))
    candidates.sort(key=lambda r:digest(f"{SEED}:{r[0]['task_id']}"))
    assert len(candidates)>=20,("Insufficient nontrivial code",len(candidates))
    for i,(row,complete,nodes,complexity,asserts) in enumerate(candidates[:20]):
        nested = i%2==1
        target = "library/implementation.py" if nested else "mylib.py"
        module = "library.implementation" if nested else "mylib"
        entry = row["entry_point"]
        public = f"from {module} import {entry} as candidate\n\n"
        for j,node in enumerate(asserts[:2]): public += f"def test_public_example_{j}():\n    "+ast.unparse(node)+"\n\n"
        files = {target:row["prompt"]+"    raise NotImplementedError('Restore the documented function')\n",
            "tests/test_public.py" if nested else "test_mylib.py":public}
        if nested: files["library/__init__.py"]=""
        hidden = f"from {module} import {entry}\n\n"+row["test"]+f"\n\ndef test_original_hidden_checks():\n    check({entry})\n"
        question = f"Implement {entry} in {target}. Preserve its public signature. The COMPLETE specification follows; examples do not replace the general contract. Do not modify tests, monkeypatch built-ins or inspect hidden tests. Additional tests will check the implementation.\n\n"+row["prompt"]+"\nWhen the implementation is complete and public tests pass, write DONE to answer.txt."
        builder.add("coding","humaneval:"+row["task_id"],question,files,"DONE",source_name="humaneval",source_id=row["task_id"],
            validation_kind="coding",hidden_files={"tests/test_hidden_public100.py":hidden},reference_files={target:complete},
            public_profile="public_fail",source_ast_nodes=nodes,source_branch_count=complexity,source_prompt=row["prompt"])


def coding_negatives(builder):
    # All requirements constrain ordinary built-ins and the SAME returned object.
    specs = [
        ("finite_square", "Return a built-in integer in the inclusive interval0 through10 whose square is2.",
         "assert type(y) is int\nassert 0<=y<=10\nassert y*y==2", "square", [0,10,2]),
        ("finite_pigeonhole", "Return a built-in list of3 built-in integers. Every element must belong to{0,1}, and all three elements must be distinct.",
         "assert type(y) is list\nassert len(y)==3\nassert all(type(v) is int and v in (0,1) for v in y)\nassert len(set(y))==3", "pigeonhole", [3,2]),
        ("permutation_order", "Return a built-in tuple which is a permutation of the built-in integers0,1,2. Interpret it as a map p(i)=tuple[i]. Composing p four times must be the identity, while composing it twice must not be the identity.",
         "assert type(y) is tuple and len(y)==3\nassert all(type(v) is int for v in y)\nassert set(y)=={0,1,2}\nassert all(y[y[y[y[i]]]]==i for i in range(3))\nassert any(y[y[i]]!=i for i in range(3))", "permutation", [3,4,2]),
        ("byte_hex_length", "Return a built-in bytes object of length1 whose ordinary hex() representation has length1. The standard built-in bytes.hex method must remain unchanged.",
         "assert type(y) is bytes\nassert len(y)==1\nassert len(y.hex())==1", "byte_hex", [1,1]),
        ("ascii_case_pair", "Return a built-in str containing exactly one ASCII letter. Its ordinary lower() value must be 'a' and its ordinary upper() value must be 'B'.",
         "assert type(y) is str\nassert len(y)==1 and ('a'<=y<='z' or 'A'<=y<='Z')\nassert y.lower()=='a'\nassert y.upper()=='B'", "ascii_case", ["a","B"]),
    ]
    for family,contract,checks,kind,certificate in specs:
        contract = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", contract)
        body = "from mylib import build_value\n\ndef test_contract():\n    y=build_value()\n"+textwrap.indent(checks,"    ")+"\n"
        q = "Implement build_value() in mylib.py. "+contract+" All conditions apply to the same returned object, using ordinary Python built-in semantics. Modifying tests, built-ins, globals or the test framework is forbidden. Return UNSOLVABLE if no implementation can satisfy the contract; otherwise complete the file and write DONE."
        builder.add("coding","public100_coding:"+family,q,{"mylib.py":"def build_value():\n    return None\n","test_mylib.py":body},None,
            source_name="authored",source_id=family,validation_kind="coding_negative",hidden_files={"tests/test_hidden_public100.py":body},
            proof_type=kind,certificate=certificate,reason=contract,public_profile="public_fail")


def search_cases(builder):
    data=json.loads((CACHE/ASSETS["squad2_train"][0]).read_text(encoding="utf-8"))
    candidates=[]
    for article in data["data"]:
        group="squad2:"+article["title"].strip().casefold()
        if group in builder.excluded.groups: continue
        choices=[]
        for p in article["paragraphs"]:
            context=p["context"]
            if not 900<=len(context)<=3600 or norm(context) in builder.excluded.passages: continue
            for qa in p["qas"]:
                if qa.get("is_impossible") or len(qa["question"])<45: continue
                answers=qa.get("answers",[])
                answers=[a for a in answers if context[a['answer_start']:a['answer_start']+len(a['text'])]==a['text']]
                if not answers:continue
                answer=answers[0]
                if not 3<=len(answer["text"])<=80 or answer["answer_start"] < len(context)*0.35: continue
                if context[answer["answer_start"]:answer["answer_start"]+len(answer["text"])]!=answer["text"]: continue
                choices.append((qa,context,answer))
        if choices:
            choices.sort(key=lambda r:digest(f"{SEED}:{r[0]['id']}"))
            candidates.append((group,article,choices[0]))
    candidates.sort(key=lambda r:digest(f"{SEED}:{r[0]}"))
    assert len(candidates)>=25,("Insufficient fresh articles",len(candidates))
    for i,(group,article,(qa,context,answer)) in enumerate(candidates[:25]):
        key=digest(group)[:10]
        reference="EVIDENCE-"+key.upper()
        filename="kb/record_"+digest(key+"path")[:8]+".md"
        files={filename:f"Evidence reference: {reference}\nArticle: {article['title']}\n\n{context}\n"}
        distractors=[p["context"] for p in article["paragraphs"] if p["context"]!=context][:3]
        for j,body in enumerate(distractors): files[f"kb/record_{digest(key+str(j))[:8]}.md"]="Separate background passage:\n"+body+"\n"
        if i<20:
            q=f"Using only the complete passage labelled {reference} in the local kb files, answer this question: {qa['question']} Return only the short answer span copied from that passage. Other passages and outside knowledge are not evidence for this question; if this passage does not determine an answer, return UNSOLVABLE."
            builder.add("search",group,q,files,answer["text"].strip(),source_name="squad2_train",source_id=qa["id"],
                validation_kind="search_source",evidence_file=filename,source_context=context,answer_start=answer["answer_start"],source_answer=answer["text"],
                source_answers=[a for a in qa['answers'] if context[a['answer_start']:a['answer_start']+len(a['text'])]==a['text']],
                accepted_answers=sorted({a['text'].strip() for a in qa['answers'] if context[a['answer_start']:a['answer_start']+len(a['text'])]==a['text']}))
        else:
            # A missing independent catalog field is not inferred from world knowledge
            # or from the accompanying article. Two complete registries witness it.
            slots=["release reviewer","regional archive steward","translation verifier","shipment recipient","restoration signatory"]
            role=slots[i-20]
            ticket="TICKET-"+key
            directory={"evidence_reference":reference,"active_ticket":ticket,"entries":[{"ticket":"TICKET-"+str(j),"person":"Person "+str(j)} for j in range(5)],"field":role}
            files["kb/catalog.json"]=dumps(directory)
            q=f"Find the {role} assigned to the active ticket for evidence reference {reference}. Resolve the reference and active ticket using kb/catalog.json; person assignments are independent administrative records, not determined by the article text. The complete available catalog contains no implicit default assignments. Return the person's name, or UNSOLVABLE when these files do not uniquely identify one."
            worlds=[]
            for name in ("Dana Mercer","Robin Vale"):
                world=copy.deepcopy(directory);world["entries"].append({"ticket":ticket,"person":name});worlds.append(world)
            builder.add("search",group,q,files,None,source_name="squad2_train+authored",source_id=qa["id"],
                validation_kind="search_negative",catalog_file="kb/catalog.json",worlds=worlds,
                witness_answers=["Dana Mercer","Robin Vale"],reason="The active ticket has no person assignment; either witness can extend all observed catalog records unchanged.")


def csv_text(headers,rows):
    stream=io.StringIO(newline="");writer=csv.writer(stream,lineterminator="\n")
    writer.writerow(headers);writer.writerows(rows);return stream.getvalue()


def table_fingerprint(table):
    return digest(dumps({"header":[norm(x) for x in table["header"]],"types":table["types"],"rows":[[norm(x) for x in row] for row in table["rows"]]}))


def sql_result(body,types,query,params):
    rows=list(csv.reader(io.StringIO(body))); headers=rows[0]
    con=sqlite3.connect(":memory:")
    try:
        con.execute("CREATE TABLE data ("+",".join('"'+h.replace('"','""')+'" '+("REAL" if t=="real" else "TEXT COLLATE NOCASE") for h,t in zip(headers,types))+")")
        con.executemany("INSERT INTO data VALUES ("+",".join("?" for _ in headers)+")",[[float(x) if t=="real" else x for x,t in zip(row,types)] for row in rows[1:]])
        result=con.execute(query,params).fetchall()
        if len(result)!=1 or len(result[0])!=1 or result[0][0] is None: raise ValueError("Non-scalar SQL")
        if not isinstance(result[0][0],(int,float)) or not math.isfinite(result[0][0]): raise ValueError("Non-numeric SQL")
        return scalar(result[0][0])
    finally: con.close()


def python_sql_result(body,types,spec,params):
    """Independent evaluator of WikiSQL's small AST; no SQL or solver imports."""
    raw=list(csv.reader(io.StringIO(body)))[1:]
    rows=[[float(x) if kind=='real' else x.casefold() for x,kind in zip(record,types)] for record in raw]
    kept=[]
    for record in rows:
        ok=True
        for (column,operator,_),param in zip(spec['conds'],params):
            value=param if types[column]=='real' else str(param).casefold()
            actual=record[column]
            if not ({0:actual==value,1:actual>value,2:actual<value}[operator]):ok=False;break
        if ok:kept.append(record[spec['sel']])
    agg=spec['agg']
    if agg==0:
        if len(kept)!=1:raise ValueError('Expected one selected row')
        value=kept[0]
    elif agg==1:value=max(kept)
    elif agg==2:value=min(kept)
    elif agg==3:value=len(kept)
    elif agg==4:value=sum(kept)
    elif agg==5:value=sum(kept)/len(kept)
    else:raise ValueError('Unsupported aggregate')
    return scalar(value)


def data_cases(builder):
    with tarfile.open(CACHE/ASSETS["wikisql"][0]) as archive:
        members={m.name:m for m in archive.getmembers() if m.isfile()}
        def load(suffix):
            item=next(m for name,m in members.items() if name.endswith('/'+suffix))
            return [json.loads(line) for line in archive.extractfile(item)]
        old_tables={r["id"]:r for r in load("dev.tables.jsonl")}
        tables={r["id"]:r for r in load("test.tables.jsonl")}
        queries=load("test.jsonl")
    excluded_hashes={table_fingerprint(old_tables[group.split(':',1)[1]]) for group in builder.excluded.groups if group.startswith('wikisql:') and group.split(':',1)[1] in old_tables}
    candidates=[];seen=set()
    for row in sorted(queries,key=lambda r:digest(f"{SEED}:{r['table_id']}:{r['question']}")):
        table=tables[row["table_id"]];spec=row["sql"];group="wikisql:"+table["id"]
        signature=table_fingerprint(table)
        if group in builder.excluded.groups or signature in excluded_hashes or signature in seen: continue
        if not 12<=len(table['rows'])<=180 or not 3<=len(table['header'])<=12 or not 1<=len(spec['conds'])<=3: continue
        if spec['agg'] not in range(6) or any(c[1] not in range(3) for c in spec['conds']):continue
        if spec['agg']==0 and len(spec['conds'])<2:continue
        headers=[str(h) for h in table['header']]
        if len(set(h.casefold() for h in headers))!=len(headers) or any(not h or h!=h.strip() or '\n' in h for h in headers):continue
        try:
            rows=[[float(str(x).strip()) if t=='real' else str(x) for x,t in zip(record,table['types'])] for record in table['rows']]
            if any(not math.isfinite(x) for record in rows for x in record if isinstance(x,float)):continue
            # Keep TEXT unambiguous under common CSV type inference. Empty text
            # and entirely numeric-looking text columns could change COUNT/order.
            ambiguous=False
            for column,kind in enumerate(table['types']):
                if kind!='real':
                    cells=[str(record[column]) for record in rows]
                    if any(not x.strip() for x in cells) or all(re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',x.strip()) for x in cells):ambiguous=True
            if ambiguous:continue
            if spec['agg'] in (1,2,4,5) and table['types'][spec['sel']]!='real':continue
            quoted=lambda index:'"'+headers[index].replace('"','""')+'"'
            selected=quoted(spec['sel']);agg=['','MAX','MIN','COUNT','SUM','AVG'][spec['agg']]
            expression=agg+'('+selected+')' if agg else selected
            query='SELECT '+expression+' FROM data WHERE '+' AND '.join(quoted(c)+[' = ?',' > ?',' < ?'][op] for c,op,value in spec['conds'])
            params=[float(value) if table['types'][c]=='real' else str(value) for c,op,value in spec['conds']]
            body=csv_text(headers,rows);expected=sql_result(body,table['types'],query,params)
            independent=python_sql_result(body,table['types'],spec,params)
            if not math.isclose(float(expected),float(independent),rel_tol=1e-12,abs_tol=1e-10):continue
        except (ValueError,TypeError,OverflowError,sqlite3.Error):continue
        seen.add(signature)
        candidates.append((row,table,body,query,params,expected,signature))
    # At least half use two or three conditions; keep substantial real tables.
    candidates.sort(key=lambda r:(0 if len(r[0]['sql']['conds'])>=2 else 1,digest(f"{SEED}:{r[0]['table_id']}")))
    assert len(candidates)>=25,("Insufficient fresh numeric table problems",len(candidates))
    for i,(row,table,body,query,params,expected,signature) in enumerate(candidates[:25]):
        filename="records.csv" if i%2 else "observations.csv"
        q=f"Use the complete table in {filename}. {row['question']} Treat numeric columns as numeric (including numeric-looking quoted literals); text comparisons are case-insensitive. Column names are exactly the CSV header. Return the single requested numeric value."
        oracle=dict(validation_kind="sql_source",csv_file=filename,column_types=table['types'],query=query,params=params,
                    source_question=row['question'],source_sql_ast=row['sql'],source_table_fingerprint=signature,source_conditions=len(row['sql']['conds']),
                    transformations=['Source REAL cells and condition parameters converted to finite canonical floats.',
                      'Formatted numbers/currency excluded; ambiguous empty or wholly numeric-looking source TEXT columns excluded.',
                      'Original nonempty unique column names retained; no rows sampled or dropped.',
                      'Independent Python SQL-AST result cross-checked against typed SQLite.'])
        if i>=20:
            families=[("unrecorded_tax","The result is a pre-tax amount. A tax rate r was applied, but r was not recorded and can be any number between0 and0.25. What was the amount including tax?","base*(1+x)",[0.05,0.2]),
                ("unknown_additive_fee","A separate fixed handling fee was added to the result. The fee was not recorded and no other facts constrain it. What was the total including this fee?","base+x",[17,43]),
                ("unknown_currency","The result was converted using an unrecorded positive exchange rate in local units per source unit. What was the converted value?","base*x",[2,5]),
                ("unrecorded_allocation","The result was split equally among an unrecorded number of recipients, which could be2 or5. What amount did each recipient receive?","base/x",[2,5]),
                ("missing_credit","An independent unrecorded credit was deducted from the result. What was the adjusted result after the credit?","base-x",[3,13])]
            family,suffix,expr,values=families[i-20]
            # Nonzero is necessary for a multiplicative missing parameter to matter.
            if len({arithmetic(expr,dict(base=float(expected),x=x)) for x in values})<2:
                expr='base+x';suffix="An independent unrecorded handling fee was added to the result. What was the final amount?";values=[7,19]
            worlds=[dict(base=float(expected),x=x) for x in values]
            oracle.update(validation_kind="sql_negative",base_result=expected,expression=expr,worlds=worlds,
                witness_answers=[arithmetic(expr,w) for w in worlds],reason="The table determines the base result but does not constrain the independent adjustment; the two witnesses differ.")
            suffix=re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])"," ",suffix)
            q=q.removesuffix("Return the single requested numeric value.")+suffix+" Only supplied evidence is allowed; return UNSOLVABLE if the final number is not uniquely determined."
            expected=None
        builder.add("data_analysis","wikisql:"+table['id'],q,{filename:body},expected,source_name="wikisql" if i<20 else "wikisql+authored",source_id=table['id'],**oracle)


def write_suite(output,download):
    if output.exists() and any(output.iterdir()):raise FileExistsError("Refusing to overwrite a panel; use --validate-only")
    provenance=sources(download);excluded=Exclusions();builder=Builder(excluded)
    math_cases(builder);humaneval_cases(builder);coding_negatives(builder);search_cases(builder);data_cases(builder)
    assert len(builder.items)==100
    rewards=random.Random(SEED^0xA71C);records=[];oracle={};metadata={}
    output.mkdir(parents=True,exist_ok=True)
    for item in sorted(builder.items,key=lambda r:r['id']):
        ident=item['id'];folder=output/'tasks'/ident;workspace=folder/'workspace';workspace.mkdir(parents=True)
        task=dict(task_id=ident,task_type=item['domain'],reward=rewards.randint(1,5),instruction=item['instruction']+f"\nCase identifier: {ident}.")
        (folder/'task.json').write_text(dumps(task),encoding='utf-8')
        for name,body in item['files'].items():
            path=workspace/safe(name);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(body,encoding='utf-8')
        records.append(dict(id=ident,domain=item['domain'],reward=task['reward'],task_path='tasks/'+ident,
            task_sha256=digest(dumps(task)),files_sha256={name:digest(body) for name,body in item['files'].items()}))
        metadata[ident]=dict(split='test',group=item['group'],source_family=item['source_name'],source_id=item['source_id'])
        oracle[ident]=dict(expected=item['expected'],answerable=item['expected'] is not None,hidden_files={},reference_files={},reason=None)
        oracle[ident].update(item['oracle'])
    exclusion_report=dict(suites=excluded.snapshots,old_group_count=len(excluded.groups),new_group_count=len(builder.groups),
        overlapping_groups=sorted(excluded.groups & builder.groups),
        methods=['Exact source problem/article/table groups excluded across all four previous suites.',
          'Normalized instructions and source passages checked; WikiSQL normalized table aliases excluded.',
          'New HumanEval source replaces previous MBPP. Cross-benchmark semantic equivalence is not exhaustively decidable.'])
    manifest=dict(version=1,seed=SEED,label='public-source-heldout100',task_count=100,
        description='All100 are heldout test;25/domain,20 answerable and5 certified negatives. Do not fit on these data.',
        oracle_sha256=digest(dumps(oracle)),research_metadata_sha256=digest(dumps(metadata)),provenance_sha256=digest(dumps(provenance)),tasks=records)
    for filename,obj in [('manifest.json',manifest),('oracle.json',oracle),('research_metadata.json',metadata),('provenance.json',provenance),('deduplication.json',exclusion_report)]:
        (output/filename).write_text(dumps(obj),encoding='utf-8')
    for name,info in provenance.items():
        path=output/info['license_file'];path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(CACHE/'public100'/(name+'.LICENSE.txt'),path)
    (output/'README.md').write_text(text('''
        # Public100 heldout tournament panel
        100 entirely heldout tasks:25/domain,20 answerable plus5 certified negatives.
        Rewards1..5 are independently seeded; their scale is a synthetic assumption.
        No training, calibration or policy selection may consume this panel's outcomes.
        Give solvers ONLY tasks/<id>/task.json and workspace. Oracle, source provenance,
        research metadata, references and hidden tests remain outside their mount.

        Math preserves full multistep GSM8K questions; every annotated arithmetic step
        is checked. Coding preserves full HumanEval contracts and official hidden tests,
        with two public examples when available and nontrivial branch/loop complexity.
        Search preserves complete long SQuAD passages with explicit evidence scope.
        CSV uses complete real WikiSQL test tables and original question/SQL annotations.
        Negatives are authored with two-world or finite mathematical certificates.

        Public benchmarks may have appeared in model pretraining. New source groups
        prevent our own direct train/test reuse but DO NOT prove absence of benchmark
        memorization. These are adapted public problems, not representative hidden-final
        tasks, SWE-bench repositories or contamination-free evidence. Source annotations
        can contain natural-language ambiguity; arithmetic/SQL checks do not prove that
        every annotation captures the only reasonable interpretation of its wording.
        Record license attribution and modifications using provenance.json and licenses/.
        SQuAD excerpts/adaptations remain under CC-BY-SA4.0; other components retain
        their source licenses. Authored wrapper/proof code follows this repository.
    '''),encoding='utf-8')


def run_pytest(folder,paths):
    env={k:os.environ[k] for k in ('PATH','SYSTEMROOT','WINDIR','TEMP','TMP') if k in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONPATH=str(folder))
    result=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',*paths],cwd=folder,env=env,
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=25)
    return result.returncode==0,result.stdout+result.stderr


def validate(output,report):
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8-sig'))
    oracle=json.loads((output/'oracle.json').read_text(encoding='utf-8-sig'))
    metadata=json.loads((output/'research_metadata.json').read_text(encoding='utf-8-sig'))
    provenance=json.loads((output/'provenance.json').read_text(encoding='utf-8-sig'))
    assert len(manifest['tasks'])==len(oracle)==len(metadata)==100
    for name,obj in [('oracle',oracle),('research_metadata',metadata),('provenance',provenance)]:assert digest(dumps(obj))==manifest[name+'_sha256']
    for info in provenance.values():assert digest((output/info['license_file']).read_bytes())==info['license_sha256']
    assert not json.loads((output/'deduplication.json').read_text(encoding='utf-8'))['overlapping_groups']
    count=collections.Counter();instructions=[];cases=[]
    for row in manifest['tasks']:
        ident=row['id'];gold=oracle[ident];folder=output/row['task_path'];work=folder/'workspace'
        task=json.loads((folder/'task.json').read_text(encoding='utf-8-sig'));instructions.append(task['instruction'])
        assert metadata[ident]['split']=='test' and set(task)=={'task_id','task_type','reward','instruction'}
        assert not {'group','split','answerable','expected'} & set(row)
        assert digest(dumps(task))==row['task_sha256']
        actual={p.relative_to(work).as_posix() for p in work.rglob('*') if p.is_file()}
        assert actual==set(row['files_sha256'])
        files={name:(work/name).read_text(encoding='utf-8') for name in actual}
        assert all(digest(body)==row['files_sha256'][name] for name,body in files.items())
        count[row['domain'],gold['answerable']]+=1;kind=gold['validation_kind']
        if kind=='math_source':
            assert len(gold['arithmetic_steps'])>=5
            for step in gold['arithmetic_steps']:
                lhs,rhs=step.rsplit('=',1);assert math.isclose(arithmetic(lhs),float(rhs.replace(',','')),rel_tol=1e-8,abs_tol=1e-7)
            assert math.isclose(float(gold['arithmetic_steps'][-1].rsplit('=',1)[1]),float(gold['expected']),rel_tol=1e-8,abs_tol=1e-7)
        elif kind=='math_negative':
            values=[arithmetic(gold['expression'],w) for w in gold['worlds']]
            assert values==gold['witness_answers'] and len(set(values))==2
        elif kind in ('coding','coding_negative'):
            assert not set(gold['hidden_files']) & set(files)
            public=sorted(name for name in files if Path(name).name.startswith('test_'))
            with tempfile.TemporaryDirectory(prefix='public100-check-') as tmp:
                sandbox=Path(tmp)/'work';shutil.copytree(work,sandbox)
                initial,detail=run_pytest(sandbox,public);assert not initial,(ident,'Original unexpectedly passes',detail)
                for name,body in gold['hidden_files'].items():
                    path=sandbox/safe(name);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(body,encoding='utf-8')
                if gold['answerable']:
                    for name,body in gold['reference_files'].items():(sandbox/safe(name)).write_text(body,encoding='utf-8')
                    passed,detail=run_pytest(sandbox,public+sorted(gold['hidden_files']));assert passed,(ident,detail)
                else:
                    proof=gold['proof_type'];cert=gold['certificate']
                    if proof=='square':assert not any(x*x==cert[2] for x in range(cert[0],cert[1]+1))
                    elif proof=='pigeonhole':assert not any(len(set(x))==cert[0] for x in itertools.product(range(cert[1]),repeat=cert[0]))
                    elif proof=='permutation':assert not any(all(p[p[p[p[i]]]]==i for i in range(3)) and any(p[p[i]]!=i for i in range(3)) for p in itertools.permutations(range(3)))
                    elif proof=='byte_hex':assert not any(len(bytes([x]).hex())==cert[1] for x in range(256))
                    elif proof=='ascii_case':assert not any(x.lower()==cert[0] and x.upper()==cert[1] for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
                    else:raise AssertionError('Unknown proof')
        elif kind=='search_source':
            context=gold['source_context'];start=gold['answer_start'];answer=gold['source_answer']
            assert context in files[gold['evidence_file']] and context[start:start+len(answer)]==answer
            assert answer.strip()==gold['expected'] and len(context)>=900
            assert set(gold['accepted_answers'])=={a['text'].strip() for a in gold['source_answers']}
            assert all(context[a['answer_start']:a['answer_start']+len(a['text'])]==a['text'] for a in gold['source_answers'])
        elif kind=='search_negative':
            observed=json.loads(files[gold['catalog_file']]);ticket=observed['active_ticket']
            assert not [r for r in observed['entries'] if r['ticket']==ticket]
            answers=[]
            for world in gold['worlds']:
                assert world['entries'][:-1]==observed['entries']
                assert {k:v for k,v in world.items() if k!='entries'}=={k:v for k,v in observed.items() if k!='entries'}
                answer=[r['person'] for r in world['entries'] if r['ticket']==ticket];assert len(answer)==1;answers.append(answer[0])
            assert answers==gold['witness_answers'] and len(set(answers))==2
        elif kind in ('sql_source','sql_negative'):
            value=sql_result(files[gold['csv_file']],gold['column_types'],gold['query'],gold['params'])
            independent=python_sql_result(files[gold['csv_file']],gold['column_types'],gold['source_sql_ast'],gold['params'])
            assert math.isclose(float(value),float(independent),rel_tol=1e-12,abs_tol=1e-10)
            if kind=='sql_source':assert value==gold['expected']
            else:
                assert value==gold['base_result']
                answers=[arithmetic(gold['expression'],w) for w in gold['worlds']]
                assert all(w['base']==float(value) for w in gold['worlds']) and answers==gold['witness_answers'] and len(set(answers))==2
        else:raise AssertionError(kind)
        cases.append({'id':ident,'passed':True})
    assert len(set(instructions))==100 and all(i==j or a not in b for i,a in enumerate(instructions) for j,b in enumerate(instructions))
    assert len({m['group'] for m in metadata.values()})==100
    for domain in DOMAINS:assert count[domain,True]==20 and count[domain,False]==5,(domain,count)
    result=dict(passed=100,total=100,negative_count=20,api_calls=0,network='none',split_counts={'test':100},
        source_group_count=100,manifest_sha256=digest((output/'manifest.json').read_bytes()),cases=cases)
    report.write_text(dumps(result),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


def docker_validate(output,image):
    output=output.resolve()
    with tempfile.TemporaryDirectory(prefix='public100-validator-output-') as tmp:
        command=['docker','run','--rm','--network','none','--read-only','--tmpfs','/tmp:rw,size=512m','--cpus','2','--memory','2g','--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--mount',f'type=bind,source={output},target=/suite,readonly','--mount',f'type=bind,source={Path(tmp).resolve()},target=/out',
            '--mount',f'type=bind,source={Path(__file__).resolve()},target=/validator.py,readonly','--entrypoint','python',image,
            '-B','/validator.py','--validate-only','--inside-docker','--output','/suite','--report','/out/validation.json']
        subprocess.run(command,check=True,timeout=600)
        shutil.copyfile(Path(tmp)/'validation.json',output/'validation.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--inside-docker',action='store_true')
    parser.add_argument('--report',type=Path)
    parser.add_argument('--image',default='nsrc-cpu-candidate:20261112')
    args=parser.parse_args()
    if not args.validate_only:write_suite(args.output,args.download)
    if args.inside_docker:validate(args.output,args.report or args.output/'validation.json')
    else:docker_validate(args.output,args.image)


if __name__=='__main__':main()
