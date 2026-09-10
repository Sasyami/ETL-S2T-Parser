# SQL-risk anti-overfit и paraphrase live-проверка — 2026-09-10

## Итог

Подгонка по раскрытым формулировкам удалена из runtime path. Семантический
выбор теперь делает operation-router одним structured enum, а последующие
стадии работают только с typed contract, буквальным scope и фактическими
tool-results. В production нет regex/keyword/denylist-классификации intent и
нет fixture identifiers.

На шести существенно переформулированных запросах candidate с
`typed_plan` прошёл `6/6` hard checks и `6/6` обязательных Max semantic
verdicts. Это development/regression evidence, а не независимый holdout и не
основание включать режим по умолчанию.

## Что удалено

- Full-string match раскрытых cardinality/nullable live-запросов.
- NL keyword/denylist intent classifiers в cardinality и constraint анализе.
- Повторная классификация intent в compilers и plan validation.
- `agents/operation_intent.py` и `agents/plan_requirements.py`.
- Fixture-shaped SQL-пример `s.id AS id, COALESCE(...)` из operation skill.
- Текстовые regex-oracles live-тестов для JOIN/fan-out/uniqueness/nullable,
  которые давали false-red на корректных формулировках.

Сохранены только синтаксические safety checks: точная техническая пара с
Unicode-стрелкой, явно размеченный `file_id`, enum/aspect consistency, точные
tool name/args, полнота SavedResultStore и SQLGlot AST. Они не определяют
пользовательский intent.

## Новая граница ответственности

1. Operation-router выбирает один из `agentic`, `conditional_cardinality`,
   `nullable_constraint` через native structured output.
2. Только два узких специальных mode допускают deterministic typed plan.
3. Scope parser извлекает literal identifiers, но не читает смысл фразы.
4. Cardinality compiler анализирует полный exact S2T-result и JOIN AST;
   `WHERE`/`COALESCE` не являются evidence уникальности или fan-out.
5. Nullable compiler требует exact S2T и role-preserving metadata.
6. Любой несогласованный, широкий или неоднозначный contract остаётся
   `agentic`; неизвестный mode не угадывается.

## Offline-проверка

- Full suite: `1049 passed, 81 skipped, 3 warnings`.
- Focused structured-contract suite: `207 passed`.
- Live collection: `80` scenarios.
- `py_compile` и `git diff --check`: clean.
- Production scan: нет `src_np`, `tgt_np`, `aux_np`, `9101`, раскрытых
  запросов или прежнего fixture SQL-примера.

## Live-дизайн

Run: `.test_runs/scope-evidence-dev/20260910_184216/`.

- 3 cardinality и 3 nullable формулировки: RU, EN и переставленный порядок.
- 6 paired scenarios, 12 последовательных `/chat` exchanges.
- Только multiagent.
- Agent: `GigaChat-2-Max`; обязательный judge: `GigaChat-2-Max`.
- Порядок `AB, BA, AB, BA, AB, BA` зафиксирован до первого вызова.
- Baseline: scope experiment off; candidate: `typed_plan`.
- Каждый arm: fresh local no-hardlinks clone и отдельная SQLite-копия.
- Neo4j и Langfuse отключены.

## Результаты

| Метрика | Baseline | Candidate |
|---|---:|---:|
| Hard pass | 5/6 | 6/6 |
| Semantic Max pass | 6/6 | 6/6 |
| HTTP 500 | 0 | 0 |
| Tool errors | 0 | 0 |
| Agent tokens | 194 217 | 75 665 |
| Agent LLM calls | 101 | 44 |
| Agent latency, sum | 203,425 s | 88,797 s |
| Reader calls | 19 | 9 |

Candidate снизил tokens на `61,0%`, LLM calls на `56,4%`, latency на
`56,3%`, reader calls на `52,6%`. Эти cost-метрики диагностические: набор уже
раскрыт и не является подтверждающим экспериментом.

Во всех шести candidate arms operation-router выбрал ожидаемый mode. Три
cardinality ответа имели fact
`conditional_duplicate_risk / join_fanout /
full_join_key_uniqueness_unknown`, точный `JOIN aux_np AS d ON d.id = s.id`,
один exact S2T reader и связанный used/display evidence. Три nullable ответа
имели `source_not_null=0`, `target_not_null=1`,
`conditional_rejection_risk / nullable_source_to_not_null_target`, ровно два
exact readers и связанные evidence IDs. Ни одна candidate-рука не вызывала
downstream-plan или upstream-answer LLM.

Единственный hard fail был у baseline English-cardinality: после reroute Max
дважды прочитал ту же exact S2T-пару и добавил лишние metadata reads. Ответ и
semantic verdict были приняты. После run правило `exactly one reader` оставлено
hard только для deterministic candidate, а для model-owned baseline повторное
чтение стало диагностикой эффективности; этот test-only сдвиг не смешивается с
зафиксированными результатами run.

## Ручной semantic audit и слабость модели

Candidate `6/6` соответствует данным и SQL вручную. У baseline English-
cardinality Max назвал `WHERE s.ok=TRUE` механизмом фильтрации внутри ответа на
cardinality и использовал неудачное двойное отрицание о дубликатах. Итоговая
условная оценка осталась верной, поэтому generic Max-judge поставил pass, но
это пограничный false-positive по строгому разделению JOIN и WHERE.

Таким образом, слабость Max есть в двух местах:

- operation-router остаётся вероятностным: в предыдущем run одна исходная
  cardinality-формулировка осталась `agentic`, хотя все шесть новых
  переформулировок здесь выбрали special mode;
- Max-judge не всегда различает JOIN fan-out, filtering и uniqueness evidence.

Код намеренно не компенсирует это NL-эвристиками. Для настоящего подтверждения
нужен заранее зарегистрированный disjoint holdout с другими identifiers, SQL
и формулировками и более предметным semantic oracle. Результат ничего не
утверждает про Ultra.

## Rollback/integrity

- Base commit run: `0f90578e4cbe303014f48398a3ccafaffd0dbe5b`.
- Все `12/12` rollback certificates: `rollback_complete=true`.
- Все disposable DB SHA до/после равны.
- Все child tracked trees clean; временные roots удалены.
- Source DB SHA: `d0770b10…d940b`; SQLite integrity `ok`, journal `delete`,
  sidecars отсутствуют.
- Plugin, `.env`, samples и runner hashes совпали с preregistration.
- Judge telemetry полна во всех 12 arms; model только Max, errors `0`.

Артефакты в `.test_runs/` игнорируются Git. Этот отчёт сохраняет постоянный
итог, но не превращает development run в confirmation.
