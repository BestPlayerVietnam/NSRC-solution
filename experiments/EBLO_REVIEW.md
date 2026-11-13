# EBLO: воспроизводимый офлайн review

Воспроизведение: python -B experiments/eblo_counterexamples.py.
Отчёт: [experiments/eblo_counterexamples.json:1](<C:/Users/never/OneDrive/Рабочий стол/OJ/experiments/eblo_counterexamples.json:1>).

Подтверждены **4/4 ложных hard-flags на решаемых fixtures**.
Это целевые контрпримеры, не оценка частоты ошибок на конкурсе.
Решения проверила независимая stdlib-логика; fixture Python-файлы не исполнялись.
Импортирован только SHA-pinned AST-сканер EBLO и наш feasibility.py.
Child: -I -B, временный cwd, env без ключей, audit-запрет сети, подпроцессов
и записей. Это guard прочитанного кода, не OS sandbox. Вызовов LLM: **0**.

| Fixture | Проверенный результат | Hard-trigger EBLO | Наш hard skip |
|---|---|---|---|
| compatible_bool_numeric_equality | [1, 1] | contradictory_asserts | Нет |
| mutable_stateful_pop | {"returns": [10, 20], "remaining": []} | contradictory_asserts | Нет |
| empty_csv_count_zero | 0 | empty_csv | Нет |
| quoted_value_is_not_column | 7 | missing_requested_columns | Нет |

**Причины.** Текст RHS различает 1 и True, хотя Python equality совместима.
Сканер не учитывает изменение list между двумя f(items):
[tmp_repo-master/harness/task_scanner.py:246](<C:/Users/never/OneDrive/Рабочий стол/OJ/tmp_repo-master/harness/task_scanner.py:246>), hard-ветка [tmp_repo-master/harness/unfeasible_filter.py:54](<C:/Users/never/OneDrive/Рабочий стол/OJ/tmp_repo-master/harness/unfeasible_filter.py:54>).
Пустая таблица имеет COUNT=0, но отклоняется: [tmp_repo-master/harness/unfeasible_filter.py:68](<C:/Users/never/OneDrive/Рабочий стол/OJ/tmp_repo-master/harness/unfeasible_filter.py:68>).
Quoted token с underscore принимается за колонку, включая значение sku_42:
[tmp_repo-master/harness/task_scanner.py:315](<C:/Users/never/OneDrive/Рабочий стол/OJ/tmp_repo-master/harness/task_scanner.py:315>), [tmp_repo-master/harness/unfeasible_filter.py:61](<C:/Users/never/OneDrive/Рабочий стол/OJ/tmp_repo-master/harness/unfeasible_filter.py:61>).
Hard-ветки подтверждены разбором AST, filter/runtime не импортировался.
Сравнение с нашим кодом касается только analyze_task, не всего planner.

**Цель — лексикографическая.** Регламент требует максимум баллов,
при равенстве — минимум токенов: [TASK/Регламент соревнования.md:123](<C:/Users/never/OneDrive/Рабочий стол/OJ/TASK/Регламент соревнования.md:123>).
EBLO использует reward - 0.01 * tokens, называя это official:
[tmp_repo-master/eblo/competition_case/fitness.py:22](<C:/Users/never/OneDrive/Рабочий стол/OJ/tmp_repo-master/eblo/competition_case/fitness.py:22>).
100 баллов за 10000 токенов лучше 99 баллов бесплатно по регламенту
(оба в бюджете 100000), но scalar fitness EBLO предпочитает второй.
В наших экспериментах сравнивать (reward, -weighted_tokens) внутри одного
fixed-model pipeline с лимитом 100000. Число независимых экспериментов
не ограничивается этим лимитом. Переключение моделей разрешено исходным
регламентом, однако исключено в наших контролируемых сравнениях.

**Mock cap не моделирует truncation.**
[tmp_repo-master/eblo/competition_case/tournament_sim.py:217](<C:/Users/never/OneDrive/Рабочий стол/OJ/tmp_repo-master/eblo/competition_case/tournament_sim.py:217>)
уменьшает completion_tokens, но строка 226 возвращает полный oracle-code,
независимо от cap; reasoning добавляется поверх cap (строка 223).
Поиск политики может награждать маленький cap без реальной потери качества.
Тест использует accuracy_scale=1e9 и ground truth:
[tmp_repo-master/tests/test_full_100_sim.py:54](<C:/Users/never/OneDrive/Рабочий стол/OJ/tmp_repo-master/tests/test_full_100_sim.py:54>).
Это проверка инфраструктуры, не доказательство качества или стоимости модели.

**Учёт расхода.** Guard проверяет превышение после вызова, без резерва:
[tmp_repo-master/harness/budget_ledger.py:86](<C:/Users/never/OneDrive/Рабочий стол/OJ/tmp_repo-master/harness/budget_ledger.py:86>).
Если record_llm_call бросает исключение, следующий log не выполняется:
[tmp_repo-master/harness/jiuwen_agent.py:298](<C:/Users/never/OneDrive/Рабочий стол/OJ/tmp_repo-master/harness/jiuwen_agent.py:298>).
Сумма completion + reasoning требует подтверждения семантики usage —
reasoning может уже входить в completion. Название p90 не даёт калибровки:
[tmp_repo-master/harness/estimator.py:175](<C:/Users/never/OneDrive/Рабочий стол/OJ/tmp_repo-master/harness/estimator.py:175>) задаёт множитель.

Переносить стоит проверку первой дешёвой попытки, остановку повтора одинаковой
ошибки и отбор контекста. Hard-skip по эвристическим маркерам, count-only
stagnation и обрезку середины каждого сообщения переносить не следует.
Runtime-файлы не изменены; SHA до/после включены в JSON.
