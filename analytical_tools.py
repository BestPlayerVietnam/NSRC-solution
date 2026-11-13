"""Small CPU-only calculation tools. No model weights or network access."""
from __future__ import annotations
import ast
import csv
import json
import math
import re
import sqlite3
import time
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path


def arithmetic_answer(text: str, exact_int: bool = False) -> str | None:
    """Interpret a bounded arithmetic AST, never eval/exec model output."""
    fenced=re.fullmatch(r'```(?:python|text)?\s*(.*?)\s*```',text.strip(),re.I|re.S)
    expression = re.sub(r'^CALC\s*:\s*', '', fenced.group(1) if fenced else text.strip(), flags=re.I)
    if len(expression) > 1800:
        return None
    try:
        tree = ast.parse(expression, mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 160:
            return None
        def bounded(x):
            if isinstance(x, Fraction) and (x.numerator.bit_length() > 4096 or x.denominator.bit_length() > 4096):
                raise ValueError('arithmetic result too large')
            return x
        def calc(n):
            if isinstance(n, ast.Constant) and type(n.value) in (int, float):
                return bounded(Fraction(str(n.value)))
            if isinstance(n, (ast.List, ast.Tuple)) and len(n.elts) <= 100:
                return [calc(x) for x in n.elts]
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
                x=calc(n.operand)
                return x if isinstance(n.op, ast.UAdd) else -x
            if isinstance(n, ast.BinOp):
                a,b=calc(n.left),calc(n.right)
                if not isinstance(a,Fraction) or not isinstance(b,Fraction):
                    raise ValueError('numeric operands required')
                if isinstance(n.op,ast.Add): x=a+b
                elif isinstance(n.op,ast.Sub): x=a-b
                elif isinstance(n.op,ast.Mult): x=a*b
                elif isinstance(n.op,ast.Div): x=a/b
                elif isinstance(n.op,ast.FloorDiv): x=Fraction(a//b)
                elif isinstance(n.op,ast.Mod): x=a%b
                elif isinstance(n.op,ast.Pow) and b.denominator==1 and abs(b)<=32: x=a**int(b)
                else: raise ValueError('operator not allowed')
                return bounded(x)
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and not n.keywords:
                args=[calc(x) for x in n.args];name=n.func.id
                if name=='abs' and len(args)==1:return abs(args[0])
                if name in ('floor','ceil') and len(args)==1:return Fraction(getattr(math,name)(args[0]))
                if name=='round' and len(args) in (1,2):
                    places=int(args[1]) if len(args)==2 else 0
                    if abs(places)>12 or (len(args)==2 and args[1].denominator!=1):raise ValueError('round digits')
                    return Fraction(round(args[0],places))
                if name in ('sum','min','max'):
                    seq=args[0] if len(args)==1 and isinstance(args[0],list) else args
                    if not seq or not all(isinstance(x,Fraction) for x in seq):raise ValueError('numeric sequence')
                    return bounded(Fraction({'sum':sum,'min':min,'max':max}[name](seq)))
            raise ValueError('unsupported expression')
        answer=calc(tree.body)
        if not isinstance(answer,Fraction):return None
        if answer.denominator==1:return str(answer.numerator)
        if exact_int:return None
        with localcontext() as ctx:
            ctx.prec=18
            return format(Decimal(answer.numerator)/Decimal(answer.denominator),'f').rstrip('0').rstrip('.')
    except (ValueError,TypeError,SyntaxError,OverflowError,ZeroDivisionError,RecursionError):
        return None


def quote_identifier(value: str) -> str:
    return '"'+value.replace('"','""')+'"'


def numeric_cell(value: str):
    value=value.strip()
    if value=='':return None
    # Preserve leading-zero identifiers rather than silently changing join keys.
    if re.fullmatch(r'[+-]?(?:0|[1-9]\d*)',value):
        n=int(value)
        return n if -(2**63)<=n<2**63 else value
    if re.fullmatch(r'[+-]?(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+)',value):
        n=float(value)
        return n if math.isfinite(n) else value
    return value


class CSVDatabase:
    """Load complete bounded tables; expose read-only scalar SELECTs and previews."""
    def __init__(self, workspace: Path, max_bytes: int=32*1024*1024, max_rows: int=200000):
        self.connection=sqlite3.connect(':memory:');self.tables=[]
        self.connection.create_collation('NOCASE',lambda a,b:(a.casefold()>b.casefold())-(a.casefold()<b.casefold()))
        root=workspace.resolve()
        files=[]
        for p in sorted(workspace.rglob('*.csv')):
            if any(part.startswith('.') for part in p.relative_to(workspace).parts):continue
            if p.is_symlink() or not p.resolve().is_relative_to(root):continue
            files.append(p)
        if not files or len(files)>12:
            self.close();raise ValueError('Need between 1 and 12 CSV tables')
        if sum(p.stat().st_size for p in files)>max_bytes:
            self.close();raise ValueError('CSV input exceeds the complete-table byte limit')
        used=set();total_rows=0
        try:
            for i,path in enumerate(files):
                base='data' if len(files)==1 else path.stem
                if not base or base.casefold().startswith('sqlite_') or base.casefold() in used:base=f'table_{i+1}'
                while base.casefold() in used:base+='_'+str(i+1)
                used.add(base.casefold())
                with path.open(newline='',encoding='utf-8-sig') as stream:
                    sample=stream.read(4096);stream.seek(0)
                    try:dialect=csv.Sniffer().sniff(sample,delimiters=',;\t|')
                    except csv.Error:dialect=csv.excel
                    reader=csv.reader(stream,dialect)
                    original=next(reader,[])
                    if not original or len(original)>150:raise ValueError('Empty or oversized CSV schema')
                    fields=[];seen=set()
                    for j,field in enumerate(original):
                        label=field.strip() or f'column_{j+1}'
                        while label.casefold() in seen:label+='_'+str(j+1)
                        fields.append(label);seen.add(label.casefold())
                    values=[];raw_values=[]
                    for row in reader:
                        if not row:continue
                        if len(row)!=len(fields):raise ValueError('CSV row width differs from header')
                        raw_values.append(row);values.append([numeric_cell(x) for x in row]);total_rows+=1
                        if total_rows>max_rows:raise ValueError('CSV row limit exceeded; no partial aggregation')
                types=[]
                for j in range(len(fields)):
                    vals=[r[j] for r in values if r[j] is not None]
                    types.append('NUMERIC' if vals and all(type(v) in (int,float) for v in vals) else 'TEXT')
                values=[[value if types[j]=='NUMERIC' else (raw_values[i][j] or None) for j,value in enumerate(row)] for i,row in enumerate(values)]
                self.connection.execute('CREATE TABLE '+quote_identifier(base)+' ('+','.join(quote_identifier(f)+' '+t+' COLLATE NOCASE' for f,t in zip(fields,types))+')')
                self.connection.executemany('INSERT INTO '+quote_identifier(base)+' VALUES ('+','.join('?' for _ in fields)+')',values)
                self.tables.append({'table':base,'file':path.relative_to(workspace).as_posix(),'columns':[{'name':f,'type':t} for f,t in zip(fields,types)],'rows':len(values),'preview':values[:3]})
            if len(files)==1:
                for alias in sorted({files[0].stem.casefold(),'sales','main'}):
                    if alias.casefold()!='data' and not alias.casefold().startswith('sqlite_'):
                        self.connection.execute('CREATE VIEW '+quote_identifier(alias)+' AS SELECT * FROM data')
            allowed={sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ,sqlite3.SQLITE_FUNCTION,sqlite3.SQLITE_RECURSIVE}
            def authorize(action,a,b,*_):
                if action==sqlite3.SQLITE_FUNCTION and str(b or a).casefold() in {'load_extension','readfile','writefile'}:return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
            self.connection.set_authorizer(authorize)
        except Exception:
            self.close();raise

    def prompt(self) -> str:
        text=json.dumps(self.tables,ensure_ascii=False,separators=(',',':'))
        if len(text)>14000:raise ValueError('CSV schema preview exceeds context limit')
        return text

    def execute(self,text: str, *, timeout: float = 2.0) -> tuple[str|None,str|None]:
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('SQL timeout must be finite and positive')
        sql=re.sub(r'^```(?:sql|sqlite)?\s*|\s*```$','',text.strip(),flags=re.I).strip()
        if len(sql)>5000 or not re.match(r'^(?:SELECT|WITH)\b',sql,re.I):return None,'Return one read-only SQLite SELECT or WITH query'
        end=time.monotonic()+min(2.0,timeout);steps=0
        def progress():
            nonlocal steps
            steps+=1
            return int(steps>10000 or time.monotonic()>end)
        self.connection.set_progress_handler(progress,1000)
        try:
            rows=self.connection.execute(sql).fetchmany(2)
            if len(rows)!=1 or len(rows[0])!=1:return None,'Query must return exactly one scalar value'
            value=rows[0][0]
            if value is None:return None,'Query returned NULL; verify column names, filters and required evidence'
            if isinstance(value,float):
                if not math.isfinite(value):return None,'Nonfinite result'
                if value.is_integer():value=int(value)
            text=str(value)
            if '\n' in text or len(text)>2000:return None,'Final answer must be one short line'
            return text,None
        except sqlite3.Error as exc:return None,'SQLite: '+str(exc)[:250]

    def close(self):
        self.connection.close()
