# Отчёт по 39 live-сценариям — 2026-09-03

## Контекст

- Код: `5757bd9` (`fix(agent): stabilize generic S2T planning`).
- Агентная модель: `GigaChat-3-Ultra`.
- Режим: `multiagent`, реальные последовательные HTTP-запросы к `/chat`.
- Включён отдельный LLM-as-judge.
- Проверялись 39 сценариев, оставшихся после исключения семи основных S2T-сценариев.
- Исходный прогон: `20260903_132106`.
- Повтор трёх пропущенных Neo4j-сценариев после запуска Neo4j: `20260903_140132`.

## Автоматический результат

Первый запуск выбрал ровно 39 сценариев:

- 30 — PASS с учётом LLM-as-judge;
- 6 — FAIL по LLM-as-judge;
- 3 — SKIP из-за недоступного Neo4j;
- HTTP 500 — 0;
- presentation warnings — 9;
- efficiency warnings — 21;
- 899 LLM-вызовов;
- 204 data-tool вызова;
- 3 281 890 токенов;
- 2 260,051 секунды агентного времени.

После запуска Neo4j три пропущенных сценария были выполнены отдельно:

- 3/3 — PASS;
- semantic failures — 0;
- judge errors — 0;
- warnings — 0;
- 27 LLM-вызовов;
- 3 data-tool вызова;
- 69 065 токенов;
- 80,222 секунды агентного времени.

## Ручная семантическая оценка

Из шести автоматических FAIL один оказался ложным срабатыванием evidence-аудита judge. После ручной проверки ответов, трасс и SQLite итог объединённого покрытия 39 сценариев:

- 34/39 — семантически корректны;
- 5/39 — реальные содержательные провалы;
- 0 — оставшихся инфраструктурных пропусков.

## Полный список сценариев

| Сценарий | Итог | Примечание |
|---|---|---|
| `analyzes_s2t_validation_risks` | PASS | Один efficiency warning |
| `answers_simple_conversation_without_display_results` | PASS | — |
| `catalog_01_finds_target_field_source` | PASS | — |
| `catalog_02_finds_source_field_targets` | FAIL | Неверный и неполный список downstream targets |
| `catalog_03_lists_table_mapping` | PASS | — |
| `catalog_04_explains_calculated_field` | PASS | — |
| `catalog_05_finds_business_metric_source` | PASS | — |
| `catalog_06_semantic_close_date_search` | PASS вручную | Judge ошибочно отклонил подтверждённый идентификатор |
| `catalog_07_finds_business_filter_rule` | PASS | — |
| `catalog_08_searches_client_id_synonyms` | FAIL | Пустой ответ при наличии S2T-строк |
| `catalog_09_maps_russian_term_to_technical_field` | FAIL | Не получено запрошенное S2T-правило |
| `catalog_10_builds_full_lineage` | PASS | Два efficiency warning |
| `catalog_11_lists_intermediate_tables` | PASS | — |
| `catalog_12_compares_two_field_origins` | PASS | — |
| `catalog_13_finds_join_condition` | PASS | — |
| `catalog_14_finds_filtering` | PASS | — |
| `catalog_15_finds_constant_or_default` | PASS | — |
| `catalog_16_finds_case_transformation` | PASS | — |
| `catalog_17_finds_aggregation` | PASS | — |
| `catalog_18_investigates_wrong_value` | PASS | Два efficiency warning |
| `catalog_19_investigates_null` | PASS | — |
| `catalog_20_finds_data_loss_points` | FAIL | Вместо фактического lineage выдан общий чек-лист |
| `catalog_21_traces_value_change` | PASS | — |
| `catalog_22_finds_multiple_sources` | PASS | — |
| `catalog_23_performs_impact_analysis` | PASS | — |
| `catalog_24_compares_two_mart_rules` | PASS | — |
| `catalog_25_finds_conflicting_s2t` | PASS | — |
| `passes_sqlite_result_into_full_neo4j_path` | PASS после повтора | Первоначально Neo4j был недоступен |
| `preserves_exact_s2t_pairs_in_answer_and_full_result` | PASS | Один efficiency warning |
| `resolves_history_reference_into_task` | PASS | Три efficiency warning |
| `returns_complete_three_edge_neo4j_path` | PASS после повтора | Первоначально Neo4j был недоступен |
| `returns_exact_global_sqlite_count` | PASS | Два presentation и два efficiency warning |
| `returns_exact_neo4j_path_and_full_result` | PASS после повтора | Первоначально Neo4j был недоступен |
| `runs_dependent_workers_sequentially` | PASS | Два presentation и три efficiency warning |
| `runs_three_dependent_sqlite_workers` | PASS | Два presentation и четыре efficiency warning |
| `selects_full_sql_result_for_scrollable_ui` | FAIL | SQL не выполнен, display отсутствует |
| `writes_independent_s2t_test_protocol` | PASS | — |
| `writes_multi_source_s2t_validation_protocol` | PASS | — |
| `writes_multi_target_s2t_validation_protocol` | PASS | — |

## Подробный разбор шести автоматических FAIL

### 1. `selects_full_sql_result_for_scrollable_ui` — реальный FAIL

Запрос требовал дословно выполнить в SQLite:

```sql
SELECT file_id, filename FROM files ORDER BY file_id
```

и показать полный результат отдельно.

Downstream дважды сформировал правильную задачу. Ошибка появилась в worker-router:

1. На специализированной стадии `run_sql` отсутствовал в доступной палитре.
2. Router сначала вернул пустой список tools.
3. После observer-reroute модель выбрала `run_sql`, но валидатор отклонил его как недоступный на этой стадии.
4. Ограниченный fallback также не содержал `run_sql`.
5. Worker вызывал только `analyze_known_facts`, не получая данных.

Финальный ответ сообщил об отсутствии evidence; display отсутствовал. Это ошибка поэтапной доступности tools и завершения worker без второго содержательного reroute, а не ошибка SQL или SQLite.

### 2. `catalog_02_finds_source_field_targets` — реальный FAIL

Запрос требовал все downstream S2T для точной пары:

- `source_table=s_grnplm_as_t_didsd_700_db_stg.a_000025_t_loanscontract`;
- `source_field=c_closedate`.

Downstream сохранил оба фильтра, но router выбрал широкий `read_s2t_by_source_table`, у которого нет фильтра `source_field`. Tool вернул 85 строк всей source-таблицы. Observer ошибочно признал широкий результат полным, после чего upstream вручную отфильтровал preview неверно.

Фактический ответ:

- вернул один правильный target `b700000025_agr_cred::subquery::v_agr_cred1`;
- придумал отсутствующий target `s_grnplm_as_t_didsd_700_db_tmd.t_loanscontract`;
- пропустил два существующих target `b700000025_agr_grntee::subquery::v_agr_grntee1` и `b700000025_agr_grntee::subquery::v_agr_grntee2`.

Причина — потеря точного фильтра между task и выбранным tool, слишком мягкая проверка observer и ненадёжная фильтрация большого evidence в upstream.

### 3. `catalog_06_semantic_close_date_search` — ложный FAIL judge

Агент нашёл и использовал подтверждённую S2T-пару:

`b3080000460002_escrow_legalgk_agreement_actual_dto → t_agr_escrow.close_dt`

с правилом `closedate`, а также перечислил другие релевантные технические поля `close_dt`.

Judge отклонил ответ, назвав `t_agr_escrow.close_dt` неподтверждённым. При этом точная пара присутствует в принятом worker evidence, используется upstream и подтверждается SQLite. Следовательно, содержательно ответ проходит, а ошибка находится в пользовательском evidence-аудите judge: он не увидел точную строку в переданном ему display-content.

Сценарий остаётся неэффективным: 35 LLM-вызовов, 9 tool-вызовов и 161 834 токена. Это efficiency issue, но не semantic failure.

### 4. `catalog_08_searches_client_id_synonyms` — реальный FAIL

Downstream правильно запросил поиск по вариантам `client_id`, `cust_id`, `client_entityid_uid`, `baseclientid`, но worker вызвал:

```text
read_s2t_by_target_table(target_table="s2t_transformations")
```

То есть имя внутренней SQLite-таблицы было ошибочно использовано как значение ETL `target_table`. Пустой результат observer признал завершением, а второй цикл повторил ту же ошибку.

Ответ заявил, что совпадений нет. В SQLite существуют четыре исходные строки, представляющие две уникальные пары:

- `b3050000420007_product.client_entityid_uid → t_agr_dep_cust.cust_id`;
- `b3050000420002_baseagreement.client_entityid_uid → t_agr_frame_cust.cust_id`.

Judge поставил FAIL по формальной причине «неподтверждённый `s2t_transformations`». Это неточная формулировка judge, но итоговый FAIL верен.

### 5. `catalog_09_maps_russian_term_to_technical_field` — реальный FAIL

Семантический поиск source/target-каталогов прошёл успешно и вернул несколько кандидатов `DEL_DT/del_dt`. Следующий worker произвольно выбрал только пару:

`B700000025_AGR_COLLAT → t_agr_collat`

Для этой пары прямого S2T нет. Worker повторял то же чтение и не проверил остальные кандидаты пакетным S2T-поиском. Финальный ответ назвал технические поля, но прямо сообщил, что соответствующее S2T-правило не представлено.

Запрос явно требовал и поле, и правило, поэтому задача не завершена. Причина — отсутствие устойчивой операции над набором результатов предыдущих workers.

### 6. `catalog_20_finds_data_loss_points` — реальный FAIL

Запрос использовал неполное имя source-таблицы `a_000025_t_loanscontract`. Downstream сразу сформировал exact-задачи для этого имени вместо предварительного разрешения полного идентификатора. Exact S2T и Neo4j readers вернули пустые результаты.

Последующие workers получили зависимые задачи вида «для каждой найденной таблицы», хотя путь не был найден, и многократно перечитывали один пустой previous result. На последнем coordinator-цикле data decision обязан вернуть `pass`; upstream сформировал общий диагностический текст без принятого evidence.

В ответ попали неподтверждённые предположения о `stg/bronze/silver`, `APPEND/UPSERT/MERGE`, ограничениях и внутренних tools, но не фактические S2T, промежуточные таблицы, JOIN и FILTER из проекта.

Judge ошибочно назвал основной причиной неподтверждённый `read_s2t_by_source_table`, приняв имя tool за физический идентификатор. Итоговый FAIL при этом верен.

## Выводы по judge

Evidence-аудит полезно ловит выдуманные таблицы и колонки, но сейчас имеет три системных ограничения:

1. Может не увидеть идентификатор в выбранном display-content и отклонить подтверждённый ответ.
2. Считает внутренние имена SQLite-таблиц физическими идентификаторами ETL.
3. Считает имена agent tools физическими идентификаторами.

Поэтому автоматический verdict следует хранить отдельно от ручной семантической классификации. В этом прогоне judge правильно определил пять проблемных ответов, но для двух из них указал неверную непосредственную причину, а один корректный ответ отклонил полностью.

## Архитектурные причины

Пять реальных провалов сводятся к следующим общим причинам:

- общие инструменты становятся доступны только после нескольких reroute, но worker может завершиться раньше;
- task с точными фильтрами допускает выбор более широкого reader;
- observer проверяет наличие результата, но не всегда точное соответствие его аргументов task;
- набор кандидатов из previous results превращается в один произвольно выбранный кандидат;
- неполный идентификатор передаётся exact-reader без отдельного разрешения;
- последний upstream-цикл обязан отвечать даже при пустом evidence и переходит к догадкам.

## Наблюдаемость и стоимость

Наиболее дорогие успешные сценарии также показывают архитектурную избыточность:

- `catalog_10_builds_full_lineage`: 89 LLM-вызовов, 440 615 токенов;
- `catalog_18_investigates_wrong_value`: 91 LLM-вызов, 416 879 токенов;
- `runs_three_dependent_sqlite_workers`: 56 LLM-вызовов, 306 932 токена.

Даже при правильном финальном ответе повторные router/planner/observer циклы и многократное чтение previous results требуют отдельной оптимизации.

## Счётчик строк загрузки

Одновременно добавлен отдельный счётчик фактически разобранных строк Excel:

- `data_row_count` — число строк данных каждого листа;
- `total_data_row_count` — сумма по книге;
- пропущенный лист возвращает `data_row_count=0`;
- скрытые строки включаются только при `include_hidden_rows=true`;
- одинаковые строки считаются отдельно;
- число не зависит от количества заполненных ячеек в строке;
- итог передаётся в HTTP-ответе, финальном progress-event и показывается в интерфейсе рядом с количеством S2T и пустых target columns.

Покрыты единичные parser-тесты, upload/progress-тесты и реальная многолистовая книга с несколькими колонками, скрытой строкой, дубликатом и пустым листом.

## Артефакты

- Полный transcript: `.test_runs/remaining_39_once_20260903/LIVE_AGENT_BENCHMARK_gigachat_GigaChat-3-Ultra_20260903_132106_multiagent.md`.
- Автоматическое сравнение: `.test_runs/remaining_39_once_20260903/LIVE_AGENT_BENCHMARK_gigachat_GigaChat-3-Ultra_20260903_132106_comparison.md`.
- JUnit: `.test_runs/remaining_39_once_20260903/LIVE_AGENT_BENCHMARK_gigachat_GigaChat-3-Ultra_20260903_132106_multiagent.xml`.
- Повтор Neo4j: `.test_runs/remaining_39_neo4j_retry_20260903/LIVE_AGENT_BENCHMARK_gigachat_GigaChat-3-Ultra_20260903_140132_comparison.md`.

Каталоги `.test_runs/` являются локальными артефактами и в коммит не включаются; данный файл сохраняет итоговую воспроизводимую сводку.
