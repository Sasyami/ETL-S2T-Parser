# Multiagent Max: exact SQL-risk scope/evidence — 2026-09-10

## Итог

Новый opt-in контракт точного scope/evidence **не показал улучшения** на пяти
ранее раскрытых проблемных SQL-risk сценариях и не должен включаться по
умолчанию.

В замороженном A/B-прогоне baseline и candidate получили одинаковые `3/5`
combined pass, а semantic pass у candidate снизился с `4/5` до `3/5`.
Candidate завершился HTTP 500 в двух сценариях против одного у baseline.
Заранее зафиксированный verdict: **`not improved / reject`**.

Отдельное общее исправление deterministic-ответа `value_changes` сработало:
ответ больше не публикует внутренний SQL alias без display-evidence. Обе руки
прошли hard oracle и обязательный semantic judge на `GigaChat-2-Max`. Этот
эффект нельзя приписывать opt-in scope-контракту, потому что renderer был
одинаков в baseline и candidate.

GigaChat Ultra не запускалась. Агент и semantic judge во всех десяти arms —
`GigaChat-2-Max`, режим — только `multiagent`.

## Что было найдено в исходных baseline failures

Аудит предыдущего 20-cell прогона `20260910_021405` разделил четыре разных
класса ошибок:

1. `row_filtering`: в 3/4 baseline ответах анализ и exact S2T-read были
   корректны, но hard oracle отклонял ответ без одного из literal endpoints.
   Semantic judge принимал 4/4.
2. `cardinality`: 4/4 baseline не прошли hard oracle из-за потерянного target
   endpoint; при этом exact mapping, фактический JOIN и условность
   неподтверждённой уникальности были получены, semantic judge принимал 4/4.
3. `constraint_rejection`: в 2/4 baseline router выбирал только metadata и мог
   завершить worker без обязательного exact S2T mapping. Это реальный gap
   обязательных evidence, а не только формат ответа.
4. `value_changes`: 4/4 baseline hard+semantic failures были детерминированы.
   Вычисленный факт был правильным, но renderer публиковал `s.id`, тогда как
   display evidence отсутствовал; Max identifier-audit отвергал ответ.

`write_semantics` уже проходил и был включён только как regression sentinel.

## Реализованный эксперимент

Freeze commit: `89844d0bba04e46c4246061817565a61cef83a88`
(`feat: enforce exact SQL risk evidence scope`).

Новый env toggle:

```text
OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT=1
```

При unset/`0` прежний runtime-path сохраняется. При `1` консервативный typed
контракт:

- извлекает только одну literal technical-пару `source → target` из исходной
  задачи и не разрешает/не угадывает имена;
- добавляет `read_s2t_source_to_target` в обязательную worker palette;
- для однозначного field-level `constraint_rejection` с explicit `file_id`
  дополнительно требует `get_source_target_column_pair`;
- не позволяет observer завершить worker, пока обязательные exact artifacts не
  приняты;
- проверяет полноту materialized saved result, не смешивая clipping model
  preview с source truncation;
- присоединяет контракт только к первому подходящему plan step, чтобы не
  размножать одинаковые readers;
- на последнем coordinator cycle возвращает честный `not assessed`, если
  required evidence всё ещё отсутствует;
- один раз и идемпотентно добавляет точный scope в публичный model answer.

Неоднозначные dotted endpoints fail-closed остаются на baseline-path. ASCII
`->` не принимается как S2T scope, чтобы не спутать его с SQL/JSON operator.
Двухчастный `value_changes` также не активирует scope-контракт: его исходный
дефект находился в deterministic renderer.

Общий renderer-fix удаляет raw target-expression из публичного
`value_changes`-ответа, сохраняя выражение в structured facts/metrics.

## Offline-проверки

- полный offline suite: **989 passed, 75 skipped**;
- целевые contract/value/worker/coordinator тесты после live-run:
  **179 passed**;
- live suite: 74 сценария успешно собираются;
- `compileall` и `git diff --check` прошли;
- `samples/` не изменён.

Unit/runtime regressions покрывают default-off, неизвестный env, консервативный
scope parser, schema-qualified/неоднозначные endpoints, обязательную palette,
observer repair, сохранённую полноту результата, final-cycle failure, один
контракт при нескольких plan steps, endpoint envelope и alias-safe renderer.

## Замороженный Max/Max A/B

Это development-прогон по уже раскрытым сценариям, не confirmatory holdout.
До первого HTTP-вызова были зафиксированы пять сценариев, порядок AB/BA и
acceptance gates:

1. `test_live_agent_checks_row_loss_risk` — AB;
2. `test_live_agent_checks_duplicate_risk_in_target` — BA;
3. `test_live_agent_checks_nulls_in_required_target_fields` — AB;
4. `test_live_agent_checks_value_change_risk` — BA;
5. `test_live_agent_checks_write_semantics_risk` — AB.

Baseline отличался от candidate только значением scope/evidence toggle. В обеих
руках были зафиксированы typed SQL-risk aspects, capability reroute on, split
selector off, current operation protocol, temperature 0, Max agent и Max
judge. Каждый success требовал одновременно pytest hard oracle и semantic
`passed`.

| Сценарий | Baseline hard / semantic | Candidate hard / semantic | Наблюдение |
|---|---|---|---|
| Row loss | pass / pass | pass / pass | обе руки: один exact mapping, правильный `WHERE s.ok = TRUE` |
| Duplicate risk | fail / pass | fail / fail | baseline дал корректный условный JOIN-анализ, но пропустил `tgt_np`; candidate остановился HTTP 500 до workers |
| Required nulls | fail / fail | fail / fail | baseline прочитал оба exact artifacts, но сломал structured upstream answer; candidate остановился HTTP 500 на repaired plan |
| Value changes | pass / pass | pass / pass | общий alias-safe deterministic renderer; не эффект toggle |
| Write semantics | pass / pass | pass / pass | regression sentinel сохранён |

## Сводные метрики

| Метрика | Baseline | Candidate |
|---|---:|---:|
| Combined hard + semantic | 3/5 | 3/5 |
| Pytest hard pass | 3/5 | 3/5 |
| Semantic pass | 4/5 | 3/5 |
| HTTP 500 | 1 | 2 |
| Agent tokens | 78 933 | 33 759 |
| Agent LLM calls | 48 | 34 |
| Reader/tool calls | 6 | 3 |
| Agent time, seconds | 193,406 | 46,834 |
| Reroutes | 0 | 0 |
| Tool errors | 0 | 0 |
| Judge attempts/errors/tokens | 9 / 0 / 4 445 | 9 / 0 / 3 398 |

Меньшие tokens/calls/time candidate не являются выигрышем: два candidate
сценария аварийно завершились до workers. Сравнивать стоимость корректных и
ранних HTTP 500 как эффективность нельзя.

## Preregistered gate

| Условие | Результат |
|---|---|
| Candidate combined не ниже 4/5 | **fail: 3/5** |
| Строгий рост combined | **fail: 3/5 → 3/5** |
| Нет baseline-pass → candidate-fail | pass: 0 |
| Semantic не ухудшается | **fail: 4/5 → 3/5** |
| Zero skip/judge error/HTTP500/tool error | **fail: candidate HTTP500=2** |
| Candidate tokens не выше 110% | формально pass, но неинтерпретируемо из-за ранних failures |
| Reroutes не растут | pass: 0 → 0 |

Итог: **не продвигать scope/evidence toggle**.

## Причины live failures

- `cardinality` candidate: первый downstream plan придумал `file_id=1` и
  потерял literal endpoints; repaired plan снова потерял `src_np`/`tgt_np` и
  нарушил one-task conditional-cardinality contract. Существующий общий
  plan-validator вернул HTTP 500 до readers. Baseline прочитал точный mapping и
  дал семантически правильный conditional verdict, но не повторил `tgt_np`,
  поэтому hard oracle остался красным.
- `constraint_rejection` candidate: оба downstream plan потеряли literal
  `src_np.id → tgt_np.id` и не создали одну самодостаточную task для mapping +
  metadata; тот же общий validator остановил выполнение до workers. Baseline
  действительно сделал ровно два требуемых reads и получил корректные факты
  `source_not_null=0`, `target_not_null=1`, однако третий upstream structured
  call сгенерировал 8192 output tokens без корректного
  `submit_upstream_answer`, и coordinator вернул HTTP 500.
- `value_changes`: новый публичный ответ больше не содержит внутренний alias и
  честно говорит о прямой проекции. Baseline и candidate получили одинаковый
  deterministic verdict и оба прошли Max judge.

Во всех arms, которые дошли до worker, существующий SQL-risk operation skill
уже выбрал нужный exact mapping reader, а для baseline constraint — и exact
metadata reader. Оставшиеся baseline failures находятся позже: в answer
contract и устойчивости structured upstream output. Поэтому добавлять ещё
prompt-only правил в operation skill вряд ли полезно.

Mandatory evidence closure остаётся проверяемой runtime-границей, но candidate
операционно дал два ранних plan-stage HTTP 500 и не решил исходные failures.
Оба 500 были сформированы общим plan-validator; opt-in prompt мог повлиять на
ответ Max, однако один стохастический pair не доказывает причинность. Для
решения о продвижении этого и не требуется: наблюдаемая candidate-регрессия и
провал заранее заданных gates достаточны для отклонения.

## Rollback и целостность

- 10/10 arms имеют `rollback_complete=true`;
- каждый arm выполнялся в fresh `git clone --local --no-hardlinks` на exact
  freeze commit и на отдельной копии SQLite;
- все десять temp roots, clones, plugin/env copies и arm DB удалены;
- disposable DB SHA до/после каждого arm совпал;
- исходная SQLite сохранила SHA256
  `d0770b10cb57d62ffb4fe63dea53dc9b8e63306550560f3c7fa48abc908d940b`,
  `PRAGMA integrity_check=ok`, journal mode `delete`, sidecar-файлов нет;
- synthetic plugin сохранил SHA256
  `6b70aae0360ef841eb26c6de1de9d94cc9b66d902d3e7870fd41c0e0a0ca8bea`;
- HEAD после запуска остался `89844d0`, tracked worktree и `samples/` чисты;
- `.env` не копировался в artifacts, временные копии удалены, исходный файл
  сохранил режим `0600`.

Неоткатываемы только потраченные provider tokens и provider-side logs.
Langfuse был принудительно выключен.

## Ограничения

- Пять сценариев уже использовались для разработки; результат нельзя выдавать
  за независимую подтверждающую оценку.
- Один повтор на arm не отделяет эффект toggle от дисперсии Max и cache/order.
- Контракт доказывает полный exact saved result, но upstream LLM всё ещё видит
  ограниченный preview. Для длинного mapping нужна deterministic агрегация над
  saved relation, а не утверждение о semantic sufficiency полного набора.
- Exact metadata call доказывает факт вызова, но не гарантирует, что в каталоге
  действительно присутствуют обе роли; это должен отражать semantic outcome.
- `return_code=0` в rollback certificates означает завершение benchmark с
  `--allow-failures`, а не quality pass; verdict рассчитан по JUnit, transcript
  и semantic markers.

## Решение

1. Оставить `OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT` выключенным по
   умолчанию и не продвигать candidate.
2. Сохранить alias-safe `value_changes` renderer и его unit/live regressions:
   он устраняет самостоятельный deterministic defect.
3. Не подбирать новый prompt на этих же пяти результатах. Следующую гипотезу
   проверять только на новом заранее зарегистрированном наборе с другими
   identifiers/SQL/формулировками.
4. Если mandatory evidence closure развивать дальше, не добавлять больше
   свободного prompt-текста: лучше материализовать required evidence slots в
   typed plan schema или построить минимальный deterministic plan для строго
   однозначного scope.

## Локальные артефакты

- Preregistration:
  `.test_runs/scope-evidence-dev/20260910_011022/preregistration.json`.
- Journal:
  `.test_runs/scope-evidence-dev/20260910_011022/journal.json`.
- Transcripts, JUnit и comparison reports:
  `.test_runs/scope-evidence-dev/20260910_011022/pairs/`.
- Rollback certificates:
  `.test_runs/scope-evidence-dev/20260910_011022/certificates/`.

`.test_runs/` намеренно игнорируется Git; этот файл хранит постоянный итог.
