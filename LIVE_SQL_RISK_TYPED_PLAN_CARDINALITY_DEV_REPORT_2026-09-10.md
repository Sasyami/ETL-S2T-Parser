# Multiagent Max: typed evidence plan и cardinality compiler — 2026-09-10

## Итог

HTTP 500 в предыдущем evidence-scope эксперименте были вызваны не SQLite,
readers или transport GigaChat. Они возникали на двух свободно-модельных
границах coordinator:

1. repaired downstream plan второй раз терял literal source/target scope или
   нарушал atomic SQL-risk plan contract; исключение превращалось в общий
   HTTP 500 до запуска workers;
2. после корректных exact readers upstream-answer repair не вернул ровно один
   native `submit_upstream_answer` и исчерпал лимит в 8192 output tokens;
   исключение также превращалось в HTTP 500.

Первый prompt-mediated evidence contract поэтому не дорабатывался. Вместо
цепочки `typed scope → prompt → свободный LLM-plan → regex-validation` создана
узкая code-owned ветка `typed_plan`:

- coordinator сам строит один typed worker-plan для поддерживаемого exact
  scope и не вызывает downstream-plan LLM;
- evidence requirements передаются worker как typed `required_evidence`, а не
  только как prompt-текст, и проверяются по точным tool args и полной
  materialized relation;
- nullable constraint и cardinality компилируются из принятого полного
  evidence и возвращаются без upstream decision/answer LLM;
- неподдерживаемые, неоднозначные или расширенные запросы fail-closed остаются
  на прежнем agentic path;
- default остаётся выключенным.

Новая архитектура устранила наблюдавшиеся HTTP 500 и на раскрытом
cardinality-regression во всех трёх повторах дала правильный детерминированный
ответ. Однако preregistered promotion gate **не пройден полностью**: Max-judge
ошибочно принял все три неверных baseline-ответа, поэтому формальный
`strict_semantic_gain` равен нулю. Результат остаётся development evidence, а
не основанием включать `typed_plan` по умолчанию.

GigaChat Ultra не запускалась. Все live arms — только `multiagent`, агент и
обязательный semantic judge — `GigaChat-2-Max`.

## Почему возникал HTTP 500

### Candidate cardinality и constraint: ошибка до workers

В первом Max/Max A/B `20260910_011022` candidate дважды сформировал
невалидный downstream plan:

- для cardinality initial plan придумал лишний `file_id`, а repaired plan
  снова потерял `src_np`/`tgt_np` и нарушил one-task conditional-cardinality
  contract;
- для constraint оба plan-ответа потеряли literal
  `src_np.id → tgt_np.id` и не создали одну самодостаточную task для exact
  mapping и role-preserving metadata.

Coordinator разрешает один repair. После второй ошибки
`validate_worker_plan_origin` / `validate_sql_risk_plan_requirements`
возвращали `CoordinatorResponseError`; общий Flask boundary преобразовывал его
в HTTP 500. Ни один data-tool в этих arms не запускался.

### Baseline constraint: ошибка после правильных readers

Baseline дошёл до workers и получил нужные факты:

```text
source_not_null=0
target_not_null=1
```

Были выполнены точные `read_s2t_source_to_target` и
`get_source_target_column_pair`. Ошибка возникла позже: первый upstream answer
нарушил native-call contract, а repair сгенерировал 8192 output tokens без
ровно одного корректного `submit_upstream_answer`. После исчерпания
единственного repair coordinator вернул HTTP 500.

Таким образом, расширять operation prompt было недостаточно: правильные данные
уже читались, а сбои находились в свободном планировании и структурированном
ответе.

## Новая архитектура

### Typed plan

Env-переключатель:

```text
OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT=typed_plan
```

Режим включается только для закрытого безопасного подмножества:

- одного conditional-cardinality запроса с одной literal table pair;
- одного nullable-only constraint-rejection запроса с exact field pair и
  единственным явным `file_id`.

Дополнительные запросы — DDL, test SQL, типы, catalog, рекомендации,
фактические counts/NULL, другие constraint mechanisms или второй scope — не
попадают в deterministic lane. Неоднозначные dotted identifiers также не
угадываются.

Для eligible task coordinator:

1. синтезирует один канонический `WorkerPlan`;
2. отмечает его `plan_source=deterministic_sql_risk_scope_v2`;
3. передаёт exact evidence slots только первому worker;
4. проверяет tool name, exact args, descriptor/source provenance,
   `source_total == row_count`, отсутствие source truncation, схему и полный
   сохранённый набор строк;
5. при неполном evidence повторяет тот же typed plan один раз, затем отвечает
   `not assessed` с HTTP 200, а не падает или делает LLM fallback.

Worker router/planner/observer и finish-worker остаются multiagent. Удалена
только стохастическая генерация data-selection plan для уже однозначного
scope.

### Deterministic nullable constraint

Полная exact S2T relation и exact source/target column metadata компилируются в
bounded `ConstraintRejectionFact`. Риск выражается только для известного
перехода `source_not_null=0 → target_not_null=1` и остаётся условным: target
projection должна фактически выдать NULL. Missing, unknown, conflicting или
неполные данные дают `not_assessed`/`conflicting`, но не догадку и не HTTP 500.

Ответ формируется кодом с
`answer_source=deterministic_constraint_rejection`; downstream и upstream LLM
для terminal fact не вызываются.

### Deterministic cardinality

После ручного аудита первого v2-прогона добавлен отдельный compiler полного
exact mapping. Он анализирует SQL через SQLGlot и принимает только
консервативно поддерживаемый flat outer SELECT.

Для текущего regression evidence compiler устанавливает:

```text
scope: src_np → tgt_np
factual_join: JOIN aux_np AS d ON d.id = s.id
mechanism: join_fanout
condition: full_join_key_uniqueness_unknown
actual_duplicates: not proven
```

`WHERE` и `COALESCE` принципиально не входят ни в механизм cardinality, ни в
доказательство уникальности. CTE, nested SELECT, set operations,
DISTINCT/GROUP/HAVING/QUALIFY/LIMIT/OFFSET, aggregate projection, malformed или
неразрешимый JOIN дают консервативный `not_assessed`.

Ответ имеет `answer_source=deterministic_cardinality`; exact mapping evidence
одновременно записывается как used и display evidence. Downstream-plan,
upstream-decision и upstream-answer LLM не вызываются.

## Изменения

### `2ed5768` — typed SQL-risk evidence lane

- `agents/sql_risk_typed_plan.py` — закрытый eligibility contract и
  deterministic worker plan;
- `agents/constraint_rejection_analysis.py` — exact nullable compiler и
  renderer;
- `agents/coordinator.py` — typed plan, evidence closure и terminal answer;
- runtime, provenance, metrics и live regressions.

### `198d7f5` — exact cardinality compiler

- `agents/cardinality_analysis.py` — full saved-relation validation, SQLGlot
  JOIN facts, bounded payload и renderer;
- coordinator terminal integration до обоих upstream LLM stages;
- hard live oracle, запрещающий подменять JOIN-механизм `WHERE`/`COALESCE`;
- metrics и edge-case tests для joins, ambiguity и provenance.

## Offline-проверка

После второй архитектурной части:

- полный offline suite: **1130 passed, 75 skipped**;
- live suite: **74** сценария собираются;
- focused cardinality/coordinator/typed/constraint/worker/metrics suites
  прошли;
- `compileall` и `git diff --check` прошли;
- `samples/` не изменён.

Покрыты complete/empty/wrong-scope/blank-rule relations, source truncation,
opaque evidence IDs, identifier bounds, exact provenance, malformed и outer
JOIN forms, unsupported SQL barriers, no-upstream terminal paths и неизменный
fallback для ineligible tasks.

## Первый v2 Max/Max A/B: пять раскрытых сценариев

Run: `.test_runs/scope-evidence-dev/20260910_133357`.
Freeze commit: `2ed5768c4b43c1782985d7c4e5ccd5522eb1c11c`.

| Метрика | Baseline | `typed_plan` |
|---|---:|---:|
| Hard pass | 4/5 | 5/5 |
| Semantic Max pass | 5/5 | 5/5 |
| Combined hard + semantic | 4/5 | 5/5 |
| HTTP 500 / tool / judge errors | 0 | 0 |
| Agent tokens | 106 090 | 75 108 |
| Agent LLM calls | 52 | 42 |
| Agent seconds | 224,928 | 72,213 |
| Reader calls | 7 | 6 |
| Reroutes | 0 | 0 |

Формальный gate этого development-run прошёл. Для constraint сокращение было
архитектурно причинным: candidate исключил downstream LLM, лишний
`read_previous_result` и upstream LLM, вернув точный nullable fact. Масштаб
наблюдаемой экономии нельзя обобщать: baseline один раз потратил 8192 output
tokens на malformed upstream answer.

Ручной аудит не позволил считать этот run доказательством semantic improvement.
Оба cardinality-ответа — baseline и candidate — называли `COALESCE`
подтверждённым механизмом дубликатов и `WHERE s.ok=TRUE` условием уникальности.
Фактическим механизмом был JOIN, а uniqueness metadata не читалась. Max-judge
ошибочно принял оба ответа. Именно это наблюдение стало основанием для
deterministic cardinality compiler, а не для продвижения исходного candidate.

## Повторный cardinality regression: три пары

Run: `.test_runs/scope-evidence-dev/20260910_143855`.
Freeze commit: `198d7f5fdc5802c05e4297eb0dade65a566b899f`.

До первого вызова были зафиксированы один раскрытый regression-сценарий, три
повтора, порядок `AB / BA / AB`, Max agent, Max judge и все acceptance gates.
Все результаты учитывались; raw judge verdicts не редактировались.

| Метрика | Baseline | `typed_plan` + compiler |
|---|---:|---:|
| Pytest hard pass | 0/3 | 3/3 |
| Ручной SQL-semantic audit | 0/3 | 3/3 |
| Max semantic judge | 3/3 | 3/3 |
| Combined hard + Max judge | 0/3 | 3/3 |
| Agent tokens | 53 827 | 27 338 |
| Agent LLM calls | 30 | 18 |
| Agent seconds | 53,771 | 25,408 |
| Judge tokens | 6 059 | 5 394 |
| Reader calls | 3 | 3 |
| Reroutes / HTTP 500 / tool / judge errors | 0 | 0 |

Candidate во всех трёх повторах:

- выполнил один exact `read_s2t_source_to_target(src_np, tgt_np)`;
- получил полный двухстрочный saved result;
- вывел `JOIN aux_np AS d ON d.id = s.id`;
- явно оставил уникальность join keys неизвестной;
- не утверждал наличие фактических дубликатов;
- не использовал `WHERE` или `COALESCE` как cardinality evidence;
- сохранил один и тот же evidence ID в fact, used evidence и display evidence;
- не вызвал downstream-plan и оба upstream LLM stages.

Baseline во всех трёх повторах прочитал правильный mapping и назвал JOIN, но
затем объявил `WHERE s.ok=TRUE` условием уникальности; в двух формулировках
также заявил неподтверждённый высокий риск. Literal directed scope был потерян.
Pytest останавливался уже на первом hard violation — потерянном scope, поэтому
более поздние cardinality assertions в этих baseline arms не исполнялись. Их
semantic результат **0/3 → 3/3** установлен отдельным ручным разбором сохранённых
raw answers; он не подменяет записанные Max-judge verdicts.

## Preregistered gate повторного run

| Условие | Результат |
|---|---|
| Candidate combined = 3/3 | pass |
| Строгий рост combined | pass: 0/3 → 3/3 |
| Candidate semantic Max = 3/3 | pass |
| Строгий рост semantic Max | **fail: 3/3 → 3/3** |
| Baseline-pass → candidate-fail | pass: 0 |
| Zero skip/judge error/HTTP500/tool error | pass |
| Candidate tokens ≤ 110% baseline | pass: ratio 0,508 |
| Reroutes не растут | pass: 0 → 0 |

Итог gate: **не пройден полностью** из-за `strict_semantic_gain`.

Это не означает, что baseline был семантически правильным. Max-judge во всех
трёх случаях дал false-positive: generic rubric оценил наличие заголовков и
идентификаторов, но не проверил SQL-инвариант «WHERE не доказывает
уникальность». Агент и judge одной Max-family воспроизвели одну и ту же
предметную ошибку. Нельзя ретроспективно заменить эти verdicts ручными и
объявить prereg gate пройденным.

## Rollback и целостность

Для повторного run:

- journal завершён, 6/6 arms записаны;
- порядок исполнения точно `AB / BA / AB`;
- 6/6 rollback certificates имеют `rollback_complete=true`;
- все clones были clean, все temp roots удалены;
- отдельная SQLite-копия каждого arm сохранила SHA до/после;
- исходная SQLite сохранила SHA256
  `d0770b10cb57d62ffb4fe63dea53dc9b8e63306550560f3c7fa48abc908d940b`,
  `integrity_check=ok`, journal mode `delete`, sidecar-файлов нет;
- frozen plugin SHA256:
  `6b70aae0360ef841eb26c6de1de9d94cc9b66d902d3e7870fd41c0e0a0ca8bea`;
- env, samples и runner hashes совпали с preregistration;
- HEAD после run остался `198d7f5`, worktree и `samples/` чисты;
- residual benchmark processes и `etl_scope_*` directories отсутствуют.

Неоткатываемы только потраченные provider tokens и provider-side logs.
Langfuse был отключён.

`return_code=0` в certificates означает завершение benchmark с
`--allow-failures`, а не quality pass. Quality восстановлена из JUnit,
transcripts и semantic markers. Runner также не пишет отдельный итоговый
machine-readable gate verdict, поэтому таблица выше пересчитана независимо.

Cardinality compiler намеренно консервативен и не является полным SQL
optimizer. Он не constant-fold-ит составные условия вроде
`ON key = key AND FALSE` и финальный `WHERE FALSE`; implicit comma join,
`NATURAL INNER JOIN` и `INSERT … SELECT` пока дают `not_assessed`. Это не
затрагивает проверенный flat-SELECT regression, но ограничивает область
semantic sufficiency.

## Решение

1. Считать исходные HTTP 500 объяснёнными и закрытыми для узких eligible
   cardinality/nullable запросов новой архитектурой.
2. Сохранить deterministic cardinality/constraint compilers и их regressions:
   они устраняют конкретные свободно-модельные failure surfaces.
3. Не включать `typed_plan` по умолчанию: повторный prereg gate формально не
   пройден, а оба live набора уже раскрыты и использованы для разработки.
4. Не дорабатывать решение на этих же ответах. Следующий шаг — заранее
   зарегистрированный disjoint holdout с другими identifiers, SQL и wording.
5. Для будущей semantic оценки передавать judge structured expected fact или
   отдельный cardinality invariant; generic Max-vs-Max judge оказался
   недостаточным для различения JOIN, WHERE и uniqueness evidence.

## Локальные артефакты

- Первый v2 run:
  `.test_runs/scope-evidence-dev/20260910_133357/`.
- Повторный cardinality run:
  `.test_runs/scope-evidence-dev/20260910_143855/`.
- Preregistration повторного run:
  `.test_runs/scope-evidence-dev/20260910_143855/preregistration.json`.
- Journal:
  `.test_runs/scope-evidence-dev/20260910_143855/journal.json`.
- Transcripts/JUnit/comparison reports:
  `.test_runs/scope-evidence-dev/20260910_143855/pairs/`.
- Rollback certificates:
  `.test_runs/scope-evidence-dev/20260910_143855/certificates/`.

`.test_runs/` намеренно игнорируется Git; этот документ хранит постоянный
аудируемый итог.
