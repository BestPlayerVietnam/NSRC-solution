"""Combine completed studies and export a compact, auditable result document."""
from pathlib import Path
import collections,json,math
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'experiments'/'runs'/'2026-11-12-v1';NEW=ROOT/'experiments'/'runs'/'2026-11-12-refinement'
MODELS=('qwen3.6-35b-a3b-noreason','qwen3.8-27b-noreason','qwen3.6-35b-a3b','qwen3.8-27b')
LABELS={'qwen3.6-35b-a3b-noreason':'3.6 plain','qwen3.8-27b-noreason':'3.8 plain','qwen3.6-35b-a3b':'3.6 reasoning','qwen3.8-27b':'3.8 reasoning'}
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s]
def main():
 first=read(OLD/'evaluation.json');fresh=read(NEW/'evaluation.json');ledger=read(OLD/'ledger.json');routing=read(OLD/'routing_evaluation.json')
 calls=[r for folder in (OLD,NEW) for r in rows(folder/'calls.jsonl')];episodes=[r for folder in (OLD,NEW) for r in rows(folder/'results.jsonl')]
 accounting={}
 for model in MODELS:
  state=ledger['models'][model];charged=state['charged_units']/10
  observed=sum(r['cost'] for r in calls if r.get('event')=='completed' and r['model']==model and r.get('accounting')=='provider_reconciled')
  assert charged<=100000
  accounting[model]={'charged':charged,'observed':round(observed,1),'held_or_uncertain':round(charged-observed,1),'remaining':round(100000-charged,1),
   'episodes':sum(r['model']==model for r in episodes),'completed_api_calls':sum(r.get('event')=='completed' and r['model']==model for r in calls)}
 totals={'accounting':accounting,'episodes':len(episodes),'completed_api_calls':sum(r.get('event')=='completed' for r in calls),'errors':sum(r.get('event')=='error' for r in calls),'unique_tasks':len(set(r['id'] for r in episodes))}
 (ROOT/'experiments'/'RESULTS_2026-11-12.json').write_text(json.dumps(totals,ensure_ascii=False,indent=2),encoding='utf-8')
 text=['# NSRC: реальные эксперименты 12 ноября 2026','',
 'Разрешённый бюджет разработки: **100 000 приведённых токенов на каждую модель**, до 400 000 всего. Один входной токен стоит 0,1; выходные токены вместе с reasoning — 1. Reasoning, уже включённый в completion, второй раз не прибавляется. Бюджет конкурсного тура остаётся отдельным: 100 000 на все задачи.','',
 f"Проведено {len(episodes)} попыток задача × модель × конфигурация, получено {totals['completed_api_calls']} ответов API. Эти числа включают ранний пилот и эксперименты с cap; это не столько независимых задач. В двух наборах — 200 разных задач, 20% с невозможным/недоопределённым требованием.",'',
 '## Расход','', '| Модель | Учтено, включая резервы | Подтверждено usage | Остаток |','| --- | ---: | ---: | ---: |']
 for model,a in accounting.items():text.append(f"| {LABELS[model]} | {a['charged']:.1f} | {a['observed']:.1f} | {a['remaining']:.1f} |")
 text+=['','Разница между учтённым и подтверждённым расходом — сохранённые резервы ошибок/прерванных запросов. Они не были возвращены в бюджет. Второй цикл использовал тот же ledger, без сброса и переноса остатков между моделями.','',
 '## Первый независимый тест','',
 'Исходный набор: 160 задач, 80 train / 40 calibration / 40 test. Основная общая панель из 20 test-задач выбрана заранее: 16 выполнимых и 4 невыполнимых. Все четыре модели её завершили. Группы исходных задач/статей/таблиц разделены. Параметры выбраны только на grouped CV train, прогнозы и код зафиксированы до отложенных вызовов.','',
 '| Модель | Решено из 16 | Верные отказы из 4 | MAE эвристики | MAE общего ridge | Покрытие верхней 90%-границы |','| --- | ---: | ---: | ---: | ---: | ---: |']
 for model,info in first['models'].items():
  section=info['splits']['test'];o=section['primary'];p=section['prediction']
  text.append(f"| {LABELS[model]} | {o['solved']} | {o['correct_abstentions']} | {p['heuristic_cost']['mae']:.1f} | {p['cost']['mae']:.1f} | {p['upper90']['coverage']:.0%} |")
 text+=['','Общий ridge улучшил прогноз для 3.6 reasoning, но проиграл эвристике для остальных трёх моделей. Поэтому эта версия не объявляется универсальным улучшением. Отказы не приносят конкурсных баллов. Стоимость неуспешной ограниченной попытки не является стоимостью гарантированного решения.','',
 '## Уточнение и свежий тест','',
 'После первого результата кандидаты подбирались отдельно по модели на прежних train-группах: эвристика, сглаженные средние, ridge и log-ridge с smearing-поправкой для среднего. Для двух plain-вариантов CV выбрала log-ridge; для 3.8 reasoning — отдельный ridge. Калибровка использовала прежнюю calibration-панель. Свежие 40 задач имеют новые группы, вопросы, статьи и таблицы; 8 выполнимых и 2 невыполнимых на домен. Восемь новых математических условий отдельно проверены без чтения oracle или ответов моделей.','',
 'Для 3.6 reasoning новый расход не планировался: её остаток слишком мал для сопоставимого прогона. Её предиктор из первого теста сохранён. CSV свежего набора имеет нормализованные переносы строк; это даёт иной preview и является явно отмеченным изменением представления входа. Все сравниваемые прогнозы ниже оценены на одних и тех же свежих ответах.','',
 '| Модель | Свежих наблюдений | MAE эвристики | MAE прежнего ridge | MAE уточнения | Покрытие верхней 90%-границы |','| --- | ---: | ---: | ---: | ---: | ---: |']
 for model,info in fresh['models'].items():
  text.append(f"| {LABELS[model]} | {info['refined_cost']['n']} | {info['heuristic_cost']['mae']:.1f} | {info['previous_cost']['mae']:.1f} | {info['refined_cost']['mae']:.1f} | {info['upper90_coverage']:.0%} |")
 text+=['','На этой свежей выборке уточнение снизило MAE относительно эвристики на 23,8% для 3.6 plain и на 17,5% для 3.8 plain. Для 3.8 reasoning MAE, наоборот, выросла на 79,5%; её эвристика остаётся более сильным проверенным ориентиром. Это наблюдаемые эффекты на 40 задачах, а не доказательство преимущества на любом распределении. Верхняя 90%-граница покрыла 87,5%, 95% и 95% случаев соответственно.','', '![Свежая проверка прогнозов](runs/2026-11-12-refinement/prediction_test.png)','',
 '## Выбор модели','',
 'Заранее зафиксированный offline-план для первой 20-task панели набрал 48 из 53 доступных баллов при 5502,9 токена. Постоянная 3.8 reasoning дала те же 48 баллов при 5382,2 токена: преимущества смешанной стратегии при таком свободном бюджете не показано.','',
 'Дополнительная сетка бюджетов — исследовательский анализ после теста, не новый независимый прогон. При плановом лимите 5000 выбор по среднему прогнозу дал 48 баллов на сохранённых траекториях за 3874 токена. При лимите 1000 фактический расход составил 1094,6: этот результат помечен как нарушение, баллы под таким лимитом не засчитаны. Это подтверждает необходимость жёсткого резерва перед каждым реальным вызовом. Offline replay не моделирует новый живой тур на 100 задачах, контекст между задачами или условную эскалацию.','',
 '## Изменения рабочего харнесса','',
 '- Отсутствие привычного имени файла или колонки больше не служит доказательством невыполнимости. Проверка противоречий учитывает явный stateless/deterministic контракт, keyword arguments и границы методов теста.','- Для единственной CSV добавлены SQLite views с естественными именами таблицы. Строковые литералы SQL не переписываются; регистр имён и системный префикс SQLite проверены. На сохранённых тренировочных ответах это исправление восстанавливало 10 из 16 ответов 3.6 plain; это диагностический replay, не повторный live-score.','- Предикторы проверены для экспериментального маршрута bounded-tool-math-v2. У рабочего run_agent.py другой контекст и лимиты, поэтому измеренные ошибки нельзя автоматически приписать его маршруту.','',
 '## Источники и ограничения','',
 '- [GSM8K](https://github.com/openai/grade-school-math), MIT; [MBPP sanitized](https://github.com/google-research/google-research/tree/master/mbpp), данные CC-BY-4.0; [SQuAD2](https://rajpurkar.github.io/SQuAD-explorer/), CC-BY-SA-4.0; [WikiSQL](https://github.com/salesforce/WikiSQL), BSD-3-Clause. Исходные SHA и преобразования сохранены.','- Публичные бенчмарки могли встречаться при обучении Qwen. MBPP-восстановление и QA по одному абзацу упрощают реальные многошаговые задачи.','- Исходные метки GSM8K оставлены неизменными. Отдельная слепая проверка отмечает 7 из 32 исходных условий как зависящие от соглашений или спорные; отказ на них не засчитывается автоматически как правильный.','- Небольшие контрольные выборки и сдвиг задач ограничивают выводы. Номинальная граница 90% не гарантирует покрытие каждого домена или бюджета всего тура.','- Все исходные результаты, ошибки и неудачные версии сохранены. Не было подбора правильного ответа по скрытым тестам.','',
 'Артефакты: `runs/2026-11-12-v1/`, `runs/2026-11-12-refinement/`. Воспроизведение и протокол — `experiments/README.md`. Бюджетный ledger не удалять.']
 (ROOT/'experiments'/'RESULTS_2026-11-12.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 targets=list(fresh['models']);x=np.arange(len(targets));width=.25
 fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
 series=[('heuristic_cost','Initial heuristic','#B8C1CD'),('previous_cost','Pooled ridge','#AD8A67'),('refined_cost','Model-specific predictor','#287D8E')]
 for offset,(key,label,color) in enumerate(series):
  vals=[fresh['models'][m][key]['mae'] for m in targets];bars=ax.bar(x+(offset-1)*width,vals,width,label=label,color=color)
  for bar,value in zip(bars,vals):ax.annotate(f'{value:.0f}',(bar.get_x()+bar.get_width()/2,value),ha='center',va='bottom',xytext=(0,3),textcoords='offset points',fontsize=9)
 ax.set_xticks(x,[LABELS[m]+f" (n={fresh['models'][m]['refined_cost']['n']})" for m in targets]);ax.set_ylabel('Mean absolute error (weighted tokens)');ax.set_title('Fresh held-out source groups',fontweight='bold');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 fig.supxlabel('All predictions were frozen before these calls. Smaller error is better; solver policy is unchanged.',fontsize=9)
 fig.savefig(NEW/'prediction_test.png',dpi=170,facecolor='white');fig.savefig(NEW/'prediction_test.svg',facecolor='white');plt.close(fig)
 print(json.dumps(totals,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
