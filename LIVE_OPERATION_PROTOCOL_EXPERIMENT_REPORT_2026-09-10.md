# Multiagent Max: 20 экспериментов operation-протоколов — 2026-09-10

## Итог

Ни одно из четырёх семейств operation-протоколов не показало улучшения по
заранее зафиксированным критериям. Все candidate-варианты должны остаться
экспериментальными; текущий `default` менять нельзя.

Подтверждающий прогон `20260910_021405` завершил все 20 пар и 40 HTTP
`/chat`-обменов. Обе руки работали только в `multiagent`; агент и обязательный
semantic judge использовали `GigaChat-2-Max`. GigaChat Ultra не запускалась.

Candidate получил 9 combined pass против 7 у baseline, но:

- semantic pass снизился с 16/20 до 15/20;
- все три candidate-only победы были исправлением hard-contract при уже
  положительной semantic-оценке baseline;
- один baseline-pass превратился в candidate HTTP 500 и semantic failure;
- agent tokens выросли на 76,3%, LLM calls — на 17,8%, время — на 95,1%;
- каждое семейство нарушило хотя бы один preregistered guardrail.

Поэтому итог runner-а — **`complete / not_improved`**. Код возврата `1`
означает полный отрицательный эксперимент, а не инфраструктурный сбой.

## Что проверялось

Матрица была зафиксирована до первого результата:

- пять аспектов: `row_filtering`, `cardinality`, `constraint_rejection`,
  `value_changes`, `write_semantics`;
- четыре общих семейства: `minimal_artifact`, `evidence_ledger`,
  `epistemic_state_machine`, `decision_table`;
- каждый из 20 candidate-протоколов сравнивался со своим `default` baseline на
  том же сценарии;
- нечётные пары выполнялись AB, чётные — BA;
- common flags были одинаковы: capability reroute включён, split selector
  выключен, typed SQL-risk aspects включены;
- temperature — 0, модель агента и judge — `GigaChat-2-Max`;
- успех одной руки требовал одновременно hard oracle и semantic judge;
- automatic promotion был выключен.

Семейство могло быть принято только при строгом росте combined pass без
регрессий и при candidate tokens не выше 110% baseline. При 5/5 в обеих руках
допускалась альтернативная победа по эффективности: не менее 10% экономии
токенов без роста числа calls. Дополнительно требовались нулевые skip,
judge errors и HTTP 500, а reroutes и tool errors не могли вырасти.

## Сводные метрики

| Метрика | Baseline | Candidate | Изменение |
|---|---:|---:|---:|
| Combined hard + semantic | 7/20 | 9/20 | +2 |
| Technical pass | 7/20 | 9/20 | +2 |
| Semantic pass | 16/20 | 15/20 | −1 |
| Agent tokens | 195 547 | 344 818 | +76,3% |
| Agent LLM calls | 185 | 218 | +17,8% |
| Agent time, seconds | 295,493 | 576,639 | +95,1% |
| Reader/tool calls | 22 | 33 | +50,0% |
| Recorded reroutes | 0 | 0 | без изменения |
| Tool errors | 0 | 0 | без изменения |
| HTTP 500 | 0 | 1 | хуже |
| Judge attempts/completed/errors | 36/36/0 | 35/35/0 | — |
| Judge tokens | 20 916 | 19 654 | −6,0% |

Меньшее число judge calls/tokens candidate не является преимуществом: число
structured judge stages зависит от идентификаторов в ответе, а один candidate
`/chat` был безусловно отклонён из-за HTTP 500. Ошибка всё равно была передана
Max-judge и сохранена как semantic failure.

Парные расхождения: три candidate-only pass и один baseline-only pass. Это
слишком мало для устойчивого вывода даже без cost guardrails: описательный
односторонний exact McNemar p-value равен 0,3125.

## Preregistered verdict по семействам

| Семейство | Combined B→C | Semantic B→C | Tokens B→C | Calls B→C | Verdict |
|---|---:|---:|---:|---:|---|
| `minimal_artifact` | 2→3 | 4→4 | 58 754→66 408 (+13,0%) | 46→48 | `not_improved`: превышен 110% token guard |
| `evidence_ledger` | 1→2 | 4→4 | 43 645→99 299 (+127,5%) | 46→57 | `not_improved`: превышен token guard |
| `epistemic_state_machine` | 3→2 | 4→3 | 51 960→96 310 (+85,4%) | 47→58 | `not_improved`: regression и HTTP 500 |
| `decision_table` | 1→2 | 4→4 | 41 188→82 801 (+101,0%) | 46→55 | `not_improved`: превышен token guard |

Ни одно семейство не имело 5/5 в обеих руках, поэтому альтернативный
efficiency gate неприменим.

## Все 20 пар

`T/S` означает `technical / semantic`.

| № | Aspect / family | Baseline T/S | Candidate T/S | Verdict |
|---:|---|---|---|---|
| 1 | row filtering / minimal artifact | failed / passed | passed / passed | `improved` |
| 2 | row filtering / evidence ledger | failed / passed | passed / passed | `improved` |
| 3 | row filtering / epistemic state machine | passed / passed | passed / passed | `tie_pass` |
| 4 | row filtering / decision table | failed / passed | failed / passed | `tie_fail` |
| 5 | cardinality / minimal artifact | failed / passed | failed / passed | `tie_fail` |
| 6 | cardinality / evidence ledger | failed / passed | failed / passed | `tie_fail` |
| 7 | cardinality / epistemic state machine | failed / passed | failed / passed | `tie_fail` |
| 8 | cardinality / decision table | failed / passed | failed / passed | `tie_fail` |
| 9 | constraint rejection / minimal artifact | passed / passed | passed / passed | `tie_pass` |
| 10 | constraint rejection / evidence ledger | failed / passed | failed / passed | `tie_fail` |
| 11 | constraint rejection / epistemic state machine | passed / passed | failed / failed | `regressed` |
| 12 | constraint rejection / decision table | failed / passed | passed / passed | `improved` |
| 13 | value changes / minimal artifact | failed / failed | failed / failed | `tie_fail` |
| 14 | value changes / evidence ledger | failed / failed | failed / failed | `tie_fail` |
| 15 | value changes / epistemic state machine | failed / failed | failed / failed | `tie_fail` |
| 16 | value changes / decision table | failed / failed | failed / failed | `tie_fail` |
| 17 | write semantics / minimal artifact | passed / passed | passed / passed | `tie_pass` |
| 18 | write semantics / evidence ledger | passed / passed | passed / passed | `tie_pass` |
| 19 | write semantics / epistemic state machine | passed / passed | passed / passed | `tie_pass` |
| 20 | write semantics / decision table | passed / passed | passed / passed | `tie_pass` |

Итого: 3 `improved`, 1 `regressed`, 10 `tie_fail`, 6 `tie_pass`.

## Наблюдения по аспектам

1. `row_filtering`: два candidate-протокола чаще сохраняли в публичном ответе
   оба точных endpoint. Однако semantic judge принимал все восемь ответов, а
   baseline hard-fail обычно был только пропуском `src_np` или `tgt_np`.
   `decision_table` добавил лишний worker/`read_previous_result` и не прошёл
   exact-one-reader oracle. Результаты также нестабильны относительно pilot.
2. `cardinality`: все восемь ответов прошли semantic judge, но ни один не прошёл
   hard oracle из-за пропуска одного endpoint. Candidate иногда точнее отделял
   JOIN multiplicity от неподтверждённой уникальности `aux_np.id`, но измеримого
   выигрыша не получил.
3. `constraint_rejection`: `minimal_artifact` прошёл в обеих руках;
   `decision_table` дал candidate-only pass, но стоил 31 173 токена и 154 секунды
   против 11 840 и 18 секунд baseline. `epistemic_state_machine` потерял exact
   S2T-read при повторном планировании, сделал пять metadata reads и завершился
   HTTP 500 — это реальная регрессия протокола/планирования, а не ошибка
   provider, tool или judge.
4. `value_changes`: все восемь рук вернули одинаковый детерминированный ответ с
   `answer_source=deterministic_value_changes`. Identifier-audit Max-judge
   отклонил alias `s.id` как неподтверждённый, потому что deterministic formatter
   не передал display evidence. Это общий дефект границы formatter/judge, а не
   различие протоколов. Даже если считать эти клетки pass в обеих руках, ни одно
   семейство всё равно не проходит token/regression guardrails.
5. `write_semantics`: все восемь рук дали одинаковый корректный
   deterministic verdict и прошли. Candidate только увеличил стоимость.

Восемь клеток `value_changes`/`write_semantics` не проверяют модельный final
upstream answer: он формируется детерминированным кодом. Эти клетки оценивают
plan/planner/observer и лишь часть upstream decision.

## Прерванный pilot

Первый frozen run `20260910_015343` на base commit `ea66d3d` был остановлен на
девятом эксперименте. Были завершены и откатаны 8 пар; baseline имел 3/8
combined pass, candidate — 2/8. Девятый baseline вернул HTTP 500, а старый live
harness не запускал judge для non-200, из-за чего корректно сработал
infrastructure fail-closed.

Pilot не объединялся с подтверждающей статистикой. Между запусками изменился
только измерительный harness: commit `a48b879` заставил Max-judge оценивать
non-200 ответы, сохранив HTTP как безусловный technical+semantic failure и
оставив terminal `judge_error` инфраструктурной ошибкой. Матрица, сценарии,
acceptance gates и все 20 protocol SHA не менялись.

Pilot важен как свидетельство дисперсии Max даже при temperature 0: результаты
первых четырёх `row_filtering` пар заметно отличались от confirmatory run.
Поэтому отдельные candidate-only выигрыши нельзя считать устойчивым эффектом.

## Rollback и целостность

- 20/20 rollback certificates имеют `rollback_complete=true`;
- все fresh clones и 40 disposable arm DB удалены, recovery не требуется;
- остаточных `.runtime-*`, SQLite WAL/SHM/journal файлов нет;
- 40 result attestations подтверждают только `multiagent`, agent/judge
  `GigaChat-2-Max` и ожидаемый protocol SHA;
- сохранены 40 transcripts, 40 JUnit XML, 40 result JSON и 20 pair reports;
- исходная SQLite до и после имеет SHA256
  `d0770b10cb57d62ffb4fe63dea53dc9b8e63306550560f3c7fa48abc908d940b`,
  `PRAGMA integrity_check=ok`;
- synthetic plugin SHA256 остался
  `6b70aae0360ef841eb26c6de1de9d94cc9b66d902d3e7870fd41c0e0a0ca8bea`;
- пять файлов `samples/` сохранили manifest SHA256
  `e41db8add86859a2de793e6bccfb1a92b938d18f4f45a10ad218c296b3856947`;
- repository snapshot до и после полностью совпал; `.env` остался ignored с
  правами `0600`.

Неоткатываемы только уже потраченные provider tokens и provider-side логи.
Внешний Langfuse для экспериментов был принудительно выключен.

## Проверки кода

- Перед confirmatory run: **933 passed, 75 skipped**, пропущены только opt-in
  live tests.
- Harness regression проверяет non-200 + successful judge и non-200 + terminal
  `judge_error`; ошибка judge не маскируется HTTP-статусом.
- Dry-run на clean committed HEAD повторно подтвердил ровно 20 вариантов и 40
  Max-only multiagent exchanges без HTTP/LLM-вызовов.
- Freeze commits: `ea66d3d` — protocols/isolated runner; `a48b879` —
  non-200 semantic-judge harness.

## Решение

Оставить `default/current` operation protocol без изменений. Не продвигать ни
одно из четырёх семейств и не комбинировать удачные клетки постфактум.

Отдельно, вне этого набора, можно исправить deterministic value-change display
boundary и калибровку identifier-audit для SQL aliases. Гипотезы об endpoint
restatement в `row_filtering` и более чётком conditional verdict в
`cardinality` допустимо проверять только на новом заранее зарегистрированном
holdout, а не повторным подбором по этим же пяти сценариям.

## Артефакты

- Confirmatory report:
  `.test_runs/operation-protocol-experiments/20260910_021405/20260910_021405_comparison.md`.
- Confirmatory preregistration:
  `.test_runs/operation-protocol-experiments/20260910_021405/20260910_021405_preregistered.md`.
- Confirmatory journal:
  `.test_runs/operation-protocol-experiments/20260910_021405/20260910_021405_journal.json`.
- Aborted pilot report:
  `.test_runs/operation-protocol-experiments/20260910_015343/20260910_015343_comparison.md`.

`.test_runs/` намеренно игнорируется Git; полный локальный evidence остаётся
там, а этот файл хранит компактный проверяемый итог.
