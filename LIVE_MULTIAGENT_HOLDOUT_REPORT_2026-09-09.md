# Multiagent Max holdout и итог архитектурных изменений — 2026-09-09

## Итог

Новая candidate-конфигурация **не показала улучшения** и не должна заменять
baseline. В подтверждающем прогоне baseline прошёл 1 из 10 сценариев, candidate —
0 из 10. Candidate потерял единственный baseline-pass, сделал больше LLM-вызовов,
потратил больше токенов и времени и чаще уходил в reroute.

Прогон был выполнен только в multiagent-режиме. И агент, и обязательный
semantic judge использовали `GigaChat-2-Max`. GigaChat Ultra не запускалась:
последний подтверждённый остаток 7 931 317 токенов ниже жёсткого минимального
резерва 15 000 000.

## Методика

- 10 сценариев были зафиксированы до первого результата и не пересекались с
  E1–E5, на которых разрабатывались изменения.
- Каждый сценарий выполнен один раз в baseline и один раз в candidate;
  нечётные пары шли AB, чётные — BA.
- Обе руки использовали `GigaChat-2-Max`, temperature 0 и один неизменный
  synthetic SQLite fixture.
- Успех требовал одновременно пройти детерминированный oracle и semantic judge
  на `GigaChat-2-Max`.
- До и после каждого из 20 HTTP `/chat`-обменов проверялся SHA256 базы:
  `d0770b10cb57d62ffb4fe63dea53dc9b8e63306550560f3c7fa48abc908d940b`.
- Skip, HTTP 500, отсутствие judge telemetry или изменение базы делали бы
  прогон неполным. Таких событий не было.
- Все 530 агентных LLM-вызовов и все judge-вызовы действительно выполнились на
  `GigaChat-2-Max`; provider telemetry содержит ненулевой расход токенов для
  каждого semantic verdict.
- Между preregistration и завершением прогона код, prompts, сценарии и пороги
  не менялись.

Baseline отключал capability reroute и typed SQL-risk aspects. Candidate включал
оба механизма; split selector оставался выключен в обеих руках. Поэтому это
проверка связки изменений, а не оценка отдельного flag.

## Сводные результаты

| Метрика | Baseline | Candidate |
|---|---:|---:|
| Combined hard + semantic | 1/10 | 0/10 |
| Technical pass/fail/error/skip | 1/9/0/0 | 0/10/0/0 |
| Semantic failures | 5 | 5 |
| HTTP 500 | 0 | 0 |
| Presentation warnings | 0 | 0 |
| Reroutes | 8 | 9 |
| Tool errors | 0 | 0 |
| Reader calls | 50 | 65 |
| Agent LLM calls | 243 | 287 |
| Agent tokens | 374 431 | 460 042 |
| Agent time, seconds | 473.760 | 571.053 |
| HTTP time, seconds | 473.823 | 571.113 |

Candidate относительно baseline использовал на 22,9% больше agent tokens,
на 18,1% больше LLM-вызовов, на 30% больше readers и на 20,5% больше agent
time.

Judge-метрики учитывались отдельно от эффективности агента:

| Метрика judge | Baseline | Candidate |
|---|---:|---:|
| Attempts / completed / retry errors | 18 / 18 / 0 | 19 / 19 / 0 |
| Input / output / total tokens | 12 452 / 986 / 13 438 | 11 650 / 1 027 / 12 677 |
| Модель | GigaChat-2-Max | GigaChat-2-Max |

## Результаты по сценариям

Каждая ячейка содержит `technical / semantic`.

| Сценарий | Baseline | Candidate |
|---|---|---|
| Exact SQLite count | failed / failed | failed / failed |
| Однозначная ссылка из истории | failed / failed | failed / failed |
| Неоднозначная ссылка из истории | failed / passed | failed / passed |
| Недоверие к assistant-only предположению | failed / failed | failed / failed |
| Последнее пользовательское правило | failed / failed | failed / failed |
| Полный scrollable SQL result | failed / passed | failed / failed |
| Четыре точные S2T-пары | failed / passed | failed / passed |
| Составная SQLite-сводка | failed / passed | failed / passed |
| Совместимость типов | passed / passed | failed / passed |
| Объяснение transformation SQL | failed / failed | failed / passed |

## Наблюдения

1. Главный общий сбой — выбор и исполнение инструментов для точного SQLite.
   Router часто не давал worker доступ к `run_sql`, после чего planner завершался
   с `no_results` либо заменял точный запрос более узким S2T/catalog tool.
2. История разрешалась нестабильно. Однозначный и latest-wins reference иногда
   доходили до supervisor корректно, но downstream не получал точный count.
   Неоднозначная ссылка и assistant-only предположение ошибочно делегировались
   вместо прямого уточнения.
3. Candidate смог получить корректные данные в нескольких задачах, но нарушал
   технический контракт: делал лишний reader, выбирал не тот display evidence
   или добавлял `ON`/`WHERE` к значениям, которые требовались без ключевых слов.
4. Единственная полная победа baseline — совместимость `INTEGER → BIGINT` одним
   exact pair-reader. Candidate дал правильный содержательный вывод, но запустил
   второй избыточный reader, поэтому потерял baseline-pass.
5. Semantic judge и deterministic oracle дополняют друг друга. Judge принимал
   несколько ответов с неправильным маршрутом или display, а hard oracle это
   обнаруживал. В transformation-сценарии judge baseline, наоборот, отклонил
   `aux_np` как неподтверждённый идентификатор, хотя он присутствовал в
   сохранённом SQL; это отдельный сигнал для калибровки judge, но raw verdict не
   переписывался.

## Решение

Preregistered verdict: **`not_improved`**.

Candidate не проходит ни критерий качества, ни guardrail отсутствия регрессий,
ни критерий эффективности. Flags candidate не следует включать как новый
default по результатам этого прогона. Следующая итерация должна отдельно
исправлять общий generic-SQL routing, доверие supervisor к истории, передачу
точных display-results и устранение избыточных workers; текущие holdout-примеры
нельзя превращать в тренировочный набор для последующего заявления об
независимом улучшении.

## Финальная проверка репозитория

- Полный offline suite: **755 passed, 75 skipped**; пропущены только opt-in live
  сценарии.
- Holdout dry-run повторно подтвердил фиксированные 10 сценариев, 20 обменов,
  обе multiagent arms, Max/Max judge и тот же SHA256 без HTTP/LLM-вызовов.
- `git diff --check` прошёл.
- Все файлы в `samples/` совпадают и с `HEAD`, и с исходной веткой
  `origin/multiagent-worker`.
- `.env` игнорируется Git, не отслеживается и имеет права `0600`.

## Артефакты и воспроизводимость

- Confirmatory report: `.test_runs/holdout-max-confirmatory/20260909_210753/20260909_210753_comparison.md`.
- Preregistration: `.test_runs/holdout-max-confirmatory/20260909_210753/20260909_210753_preregistered.md`.
- Первый запуск `.test_runs/holdout-max/20260909_205457` был признан pilot и не
  включён в результат: semantic failure тогда ошибочно оформлялся как teardown
  error. Harness был исправлен и закоммичен до нового полного preregistered run.

`.test_runs/` намеренно игнорируется Git; JUnit и полные transcripts остаются
локальными, а этот файл хранит компактный проверяемый итог.

## Выполненные изменения проекта

- Старые notebooks, generated JUnit и устаревшие отчёты удалены либо перенесены
  в `docs/history/`; Excel samples не изменялись.
- Live-тесты разделены на смысловые группы, добавлены history, validation и
  entity-resolution сценарии.
- Реализованы Raw/Resolved validation contracts, общий entity resolver,
  dependency-based readers и deterministic compiler с Phase 0–3 и 13 checks.
- Добавлены no-silent-fallback, typed WorkerOutcome, run-scoped saved results,
  CandidateSet batch guard, origin/requirement guards плана и bounded telemetry.
- SQL-risk flow получил typed aspects и deterministic value-change,
  write-semantics и cardinality safeguards.
- Ultra runner защищён fail-closed floor 15 млн токенов, включая отдельный Ultra
  judge; Max holdout имеет обязательную judge telemetry и неизменяемый DB hash.
- `.env` хранится локально вне Git с правами `0600`; sample-файлы сохранены.
