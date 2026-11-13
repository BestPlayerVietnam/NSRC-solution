"""Bounded CPU-only memory of tool repair events, never task answers.

The caller must supply actual tool feedback, not model text. A later successful
check certifies execution/public tests only, never hidden-answer correctness.
Advice is a fixed conditional template; no task text is copied into the bank.
"""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass, replace


MODELS = frozenset({
    'qwen3.6-35b-a3b', 'qwen3.6-35b-a3b-noreason',
    'qwen3.8-27b', 'qwen3.8-27b-noreason',
})
VERIFIERS = {'data_analysis': 'tool_execution', 'math': 'tool_execution',
             'coding': 'public_tests'}
MAX_CARDS = 32
MAX_PENDING = 64
MAX_HISTORY = 3
MAX_RENDER_CHARS = 1000
MAX_CALL_SEQ = 1_000_000


@dataclass(frozen=True)
class Lesson:
    id: str
    model: str
    domain: str
    version: int
    category: str
    applicability: tuple[str, ...]
    advice: str
    verification_level: str
    source_refs: tuple[str, ...]
    support_count: int = 1
    caution_count: int = 0


@dataclass(frozen=True)
class _Template:
    domain: str
    applicability: tuple[str, ...]
    hints: tuple[str, ...]
    advice: str


_SQL_HINTS = ('sql', 'sqlite', 'csv', 'table', 'column', 'schema', 'join',
              'таблиц', 'столб', 'колон')
_CODE_HINTS = ('edit', 'repair', 'fix', 'patch', 'implement', 'function',
               'module', 'исправ', 'реализ', 'функц', 'модул')
_TEMPLATES = {
    'sql_identifier': _Template('data_analysis', ('sqlite_identifier_lookup',), _SQL_HINTS,
        'For an unknown SQLite identifier, inspect the current schema and use its exact '
        'table/column names. Double-quote identifiers, including spaces; do not invent columns.'),
    'sql_integer_result': _Template('data_analysis', ('sqlite_exact_integer_output',),
        ('integer', 'exact_int', 'whole number', 'целое', 'целых'),
        'When the checker requires an integer, re-check the requested quantity and query. '
        'Do not truncate a fractional result merely to pass the format check.'),
    'sql_query_format': _Template('data_analysis', ('sqlite_readonly_scalar_query',), _SQL_HINTS,
        'Use one read-only SQLite SELECT or WITH query returning exactly one scalar. '
        'Successful execution checks query format, not whether it answers the question.'),
    'arithmetic_format': _Template('math', ('bounded_arithmetic_expression',),
        ('calc', 'compute', 'calculate', 'expression', 'arithmetic', 'вычисл', 'посчит', 'сколько'),
        'For the arithmetic checker, return CALC: followed by a supported numeric expression, '
        'without prose or assignments. Check operators, parentheses and division by zero.'),
    'coding_unique_edit': _Template('coding', ('exact_old_new_source_edits',), _CODE_HINTS,
        'For an old/new edit, copy an exact span from the current source that occurs once. '
        'Re-read the current source after earlier edits; do not guess omitted text.'),
    'coding_allowed_path': _Template('coding', ('allowed_source_paths_only',), _CODE_HINTS,
        'Use only the current allowed source paths. Tests, configuration and paths outside '
        'the repair set are read-only; fix implementation code instead.'),
    'coding_response_format': _Template('coding', ('json_files_or_exact_edits',), _CODE_HINTS,
        'Return one JSON files mapping or edits list, not both. Each edit needs path, old '
        'and new strings; each files entry needs a complete small allowed source.'),
    'coding_large_source': _Template('coding', ('large_source_exact_edits',),
        ('large', 'long', 'truncated', 'exact', 'edit', 'больш', 'обрез'),
        'For a large or truncated source, use exact old/new edits from visible current text. '
        'Do not replace the entire file or remove unseen code.'),
}

_EXACT_ERRORS = {
    'Return one read-only SQLite SELECT or WITH query': 'sql_query_format',
    'Query must return exactly one scalar value': 'sql_query_format',
    'Invalid arithmetic expression': 'arithmetic_format',
    'Edit old text must match exactly once': 'coding_unique_edit',
    'Edit path is not an allowed source': 'coding_allowed_path',
    'Source path outside allowed repair set': 'coding_allowed_path',
    'Unsafe source path': 'coding_allowed_path',
    'Return JSON files or exact edits with source paths': 'coding_response_format',
    'Use files or edits, not both': 'coding_response_format',
    'Use 1 to 16 edits': 'coding_response_format',
    'No bounded source update': 'coding_response_format',
    'Large sources require exact old/new edits': 'coding_large_source',
}


def _category(domain: str, feedback: object) -> str | None:
    if type(feedback) is not str or not 0 < len(feedback) <= 4096:
        return None
    category = _EXACT_ERRORS.get(feedback)
    if domain == 'data_analysis':
        if re.fullmatch(r'SQLite: no such (?:column|table): [^\r\n]+', feedback):
            category = 'sql_identifier'
        elif re.fullmatch(r'Expected an integer result, got: [^\r\n]+', feedback):
            category = 'sql_integer_result'
    if category and _TEMPLATES[category].domain == domain:
        return category
    return None


def _scope(model: object, domain: object) -> bool:
    return type(model) is str and model in MODELS and type(domain) is str and domain in VERIFIERS


def _digest(source_ref: object) -> str | None:
    if type(source_ref) is not str or not 0 < len(source_ref) <= 4096:
        return None
    return hashlib.sha256(source_ref.encode('utf-8', 'replace')).hexdigest()[:32]


def _sequence(value: object) -> bool:
    return type(value) is int and 0 <= value <= MAX_CALL_SEQ


def _source_mask(digest: str) -> int:
    # Bounded Bloom filter: false positives can omit a lesson, never leak a
    # lesson back into a task that contributed to it after old refs expire.
    return (1 << (int(digest[:8], 16) % 512)) | (1 << (int(digest[8:16], 16) % 512))


class ExperienceMemory:
    """One in-process bank; entries never cross model or domain boundaries."""

    def __init__(self, mode: str = 'revise'):
        if type(mode) is not str or mode not in {'off', 'append', 'revise'}:
            raise ValueError('memory mode must be off, append or revise')
        self.mode = mode
        self._cards: OrderedDict[str, tuple[Lesson, ...]] = OrderedDict()
        self._pending: OrderedDict[tuple[str, str, str], dict[str, tuple[int, int, int]]] = OrderedDict()
        self._sources: dict[str, int] = {}

    def record_failure(self, *, model: str, domain: str, source_ref: str,
                       call_seq: int, feedback: str, used_ids=()) -> str | None:
        """Record an allowlisted real tool error; no raw feedback is retained."""
        if self.mode == 'off' or not _scope(model, domain) or not _sequence(call_seq):
            return None
        digest = _digest(source_ref)
        category = _category(domain, feedback)
        if digest is None or category is None:
            return None
        key = (model, domain, digest)
        pending = self._pending.setdefault(key, {})
        old = pending.get(category)
        if old is None:
            pending[category] = (call_seq, call_seq, 1)
        elif call_seq > old[1]:
            pending[category] = (old[0], call_seq, min(2, old[2] + 1))
            card_id = self._card_id(model, domain, category)
            if self.mode == 'revise' and card_id in self._used(used_ids) and card_id in self._cards:
                previous = self._cards[card_id][-1]
                template = _TEMPLATES[category]
                revised = replace(previous, version=previous.version + 1,
                    caution_count=min(MAX_CALL_SEQ, previous.caution_count + 1),
                    applicability=template.applicability + ('matching_current_tool_error',),
                    advice='Only after the same tool error recurs: ' + template.advice,
                    source_refs=self._refs(previous.source_refs,
                        (f'{digest}:f:{old[0]}', f'{digest}:f:{call_seq}')))
                self._save(revised, digest)
        self._pending.move_to_end(key)
        while len(self._pending) > MAX_PENDING:
            self._pending.popitem(last=False)
        return category

    def retrieve(self, *, model: str, domain: str, instruction: str,
                 feedback: str = '', source_ref: str = '') -> tuple[Lesson, ...]:
        if self.mode == 'off' or not _scope(model, domain) or type(instruction) is not str:
            return ()
        digest = _digest(source_ref)
        mask = _source_mask(digest) if digest is not None else 0
        category = _category(domain, feedback)
        text = instruction[:16000].casefold()
        ranked = []
        for history in self._cards.values():
            card = history[-1]
            if (card.model, card.domain) != (model, domain):
                continue
            if mask and self._sources.get(card.id, 0) & mask == mask:
                continue
            same_error = category == card.category
            if card.caution_count and not same_error:
                continue
            hints = sum(hint in text for hint in _TEMPLATES[card.category].hints)
            if not same_error and not hints:
                continue
            score = 10 * same_error + min(3, hints) + min(3, card.support_count) * 0.1 - min(3, card.caution_count)
            ranked.append((score, card.id, card))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(item[2] for item in ranked[:2])

    def render(self, lessons: tuple[Lesson, ...]) -> str:
        """Render only owned current versions; forged or stale cards are ignored."""
        if self.mode == 'off' or not isinstance(lessons, (tuple, list)) or len(lessons) > 2:
            return ''
        lines = ['Past tool observations are conditional hints, not correctness or impossibility proofs.']
        for lesson in lessons:
            if not isinstance(lesson, Lesson) or type(lesson.id) is not str:
                continue
            history = self._cards.get(lesson.id)
            if not history or history[-1] != lesson:
                continue
            line = f'[{lesson.id} v{lesson.version}; {lesson.verification_level}] {lesson.advice}'
            if len('\n'.join(lines + [line])) <= MAX_RENDER_CHARS:
                lines.append(line)
        return '\n'.join(lines) if len(lines) > 1 else ''

    def finish_task(self, *, model: str, domain: str, source_ref: str, call_seq: int,
                    verification_level: str, used_ids=(), success: bool = True) -> tuple[Lesson, ...]:
        """Close pending events; successful execution is not an answer label.

        Only a later matching verifier promotes a lesson. An unsuccessful finish
        only clears pending events. Caution revisions happen at record_failure,
        where used_ids names the cards actually supplied to that failing call.
        """
        if self.mode == 'off' or not _scope(model, domain):
            return ()
        digest = _digest(source_ref)
        if digest is None:
            return ()
        pending = self._pending.pop((model, domain, digest), {})
        if not _sequence(call_seq) or type(success) is not bool:
            return ()
        if not success:
            return ()
        if verification_level != VERIFIERS[domain]:
            return ()
        used = self._used(used_ids)
        changed = []
        for category, (first, last, _) in pending.items():
            if call_seq <= last:
                continue
            template = _TEMPLATES[category]
            card_id = self._card_id(model, domain, category)
            history = self._cards.get(card_id, ())
            old = history[-1] if history else None
            refs = (f'{digest}:f:{first}', f'{digest}:v:{call_seq}')
            if old is None:
                current = Lesson(card_id, model, domain, 1, category, template.applicability,
                                 template.advice, verification_level, refs)
            elif self.mode == 'revise' and card_id in used:
                current = replace(old, version=old.version + 1,
                    support_count=min(MAX_CALL_SEQ, old.support_count + 1),
                    source_refs=self._refs(old.source_refs, refs))
            else:
                self._sources[card_id] = self._sources.get(card_id, 0) | _source_mask(digest)
                continue
            self._save(current, digest)
            changed.append(current)
        return tuple(changed)

    @staticmethod
    def _used(used_ids) -> set[str]:
        if not isinstance(used_ids, (tuple, list, set, frozenset)) or len(used_ids) > MAX_CARDS:
            return set()
        return {value for value in used_ids if type(value) is str and len(value) <= 40}

    @staticmethod
    def _card_id(model: str, domain: str, category: str) -> str:
        return hashlib.sha256(f'{model}/{domain}/{category}'.encode()).hexdigest()[:16]

    def _save(self, card: Lesson, digest: str) -> None:
        self._cards[card.id] = (*self._cards.get(card.id, ()), card)[-MAX_HISTORY:]
        self._cards.move_to_end(card.id)
        self._sources[card.id] = self._sources.get(card.id, 0) | _source_mask(digest)
        while len(self._cards) > MAX_CARDS:
            removed, _ = self._cards.popitem(last=False)
            self._sources.pop(removed, None)

    @staticmethod
    def _refs(old: tuple[str, ...], new: tuple[str, ...]) -> tuple[str, ...]:
        # Keep the original failure/success witnesses plus four recent refs.
        recent = tuple(ref for ref in dict.fromkeys((*old[2:], *new)) if ref not in old[:2])
        return old[:2] + recent[-4:]

    def history(self, lesson_id: str) -> tuple[Lesson, ...]:
        if type(lesson_id) is not str:
            return ()
        return self._cards.get(lesson_id, ())

    def summary(self) -> dict:
        return {'mode': self.mode, 'cards': len(self._cards), 'pending_tasks': len(self._pending),
                'history_ids': {key: [lesson.version for lesson in versions]
                                for key, versions in self._cards.items()}}
