# AGENTS.md

## Назначение

ETL S2T Parser разбирает Excel-файлы с ETL/S2T-описаниями, сохраняет исходные данные и каталоги в SQLite, строит Neo4j-lineage и отвечает на вопросы через read-only инструментального агента.

## Текущее состояние — 2026-09-10

- Рабочая ветка может отличаться. Перед изменениями проверять `git status`;
  незакоммиченные пользовательские изменения и каталог `artifacts/` не удалять
  и не перезаписывать.
- `/chat` по умолчанию использует multiagent; `CHAT_AGENT_MODE=single_agent` оставлен как baseline.
- Актуальный поток: `supervisor → downstream plan → workers → upstream decision → upstream answer`.
- Downstream создаёт полный план из 1–8 задач чтения; coordinator допускает максимум два цикла. Workers идут последовательно и могут лениво читать принятые результаты предыдущих workers.
- Router одновременно выбирает tools, retrieval-skills и schemas; planner вызывает выбранные tools; observer проверяет каждый data-tool result и возвращает только `complete`, `continue` или `reroute`.
- Upstream получает исходную задачу и принятые evidence, решает `pass/reroute`, затем анализирует данные, формирует ответ и выбирает display-results.
- Полные tool-results живут только в run-scoped хранилище; последующим workers передаются короткие `result_id`/schema references. SQLite проекта не изменяется.
- Специализированный `validation_protocol` использует поток `RawTestProtocolContract → shared entity resolution → ResolvedTestProtocolContract → dependency-based exact readers → deterministic compiler`. Ошибки extraction/resolution возвращают структурированные состояния и не вызывают silent fallback в agentic.
- Validation compiler поддерживает режимы `explicit`, `standard`, `exhaustive`, Phase 0 preflight и 13 SQL checks в Phase 1–3. Файл необязателен: без него доступны S2T-only checks, а зависимые от catalog checks помечаются `partial`/`unavailable`.
- `agents/entity_resolution.py` общий для validation и agentic flows: already-canonical identifier проходит exact bypass, а typo/partial/semantic mention разрешается с сохранением source/target role. Ambiguity не угадывается.
- Live-набор разделён на `smoke`, `history`, `display`, `handoff`, `graph`, `validation`, `resolution`, `catalog`; внешнюю SQLite-базу можно явно задать через `LIVE_AGENT_DB_PATH`.
- `scripts/run_multiagent_experiments.py` запускает bounded E1–E5. GigaChat Ultra runs (включая отдельно настроенный Ultra judge) защищены fail-closed проверкой баланса с floor 15 000 000 токенов и консервативным резервом на фактические HTTP `/chat`-обмены и judge retries.
- `scripts/run_multiagent_holdout.py` запускает фиксированный независимый A/B из 10 сценариев и 20 последовательных `/chat`-обменов: обе конфигурации только multiagent, агент и обязательный semantic judge — GigaChat-2-Max, порядок AB/BA контрбалансирован, SQLite SHA256 проверяется до и после каждого обмена.
- Confirmatory holdout `20260909_210753` завершён полностью без skip/HTTP 500: baseline `1/10`, candidate `0/10`, verdict `not_improved`. Candidate использовал 460 042 agent tokens против 374 431 и потерял единственный baseline-pass; конфигурацию не продвигать. Подробности — `LIVE_MULTIAGENT_HOLDOUT_REPORT_2026-09-09.md`.
- `scripts/run_operation_protocol_experiments.py` запускает 20 prompt-only SQL-risk protocol variants (5 aspects × 4 generic families) как 40 paired Max/Max-judge multiagent exchanges. Каждый pair изолирован fresh no-hardlinks clone и disposable SQLite-копиями, имеет preregistration/journal/rollback certificate и не меняет default автоматически.
- `OPERATION_SQL_RISK_SCOPE_EVIDENCE_EXPERIMENT=1` сохраняет первый prompt-mediated development-контракт. Значение `typed_plan` включает узкую code-owned ветку для однозначных conditional-cardinality и nullable-constraint запросов: deterministic worker-plan, exact saved-result evidence и deterministic renderers. Cardinality compiler извлекает фактический JOIN внешнего SELECT и не использует `WHERE`/`COALESCE` как механизм или evidence уникальности; nullable compiler читает exact mapping и role-preserving metadata. Более широкие/неоднозначные запросы остаются на default agentic path без scope-аттестации; default выключен.
- Max/Max development run `20260910_133357` после typed redesign дал combined `4/5 → 5/5`, semantic `5/5 → 5/5`, ноль HTTP 500 и `-29,2%` agent tokens, но manual audit обнаружил ошибочное смешение `WHERE` и uniqueness в обеих руках. После deterministic cardinality compiler три пары `20260910_143855` дали hard/combined `0/3 → 3/3`, ноль ошибок и `-49,2%` tokens; полный prereg gate всё равно не пройден, потому Max-judge false-positive оставил semantic `3/3 → 3/3`. Это revealed development evidence, не default promotion; подробности — `LIVE_SQL_RISK_TYPED_PLAN_CARDINALITY_DEV_REPORT_2026-09-10.md`.
- Confirmatory operation-protocol run `20260910_021405` завершил и откатил 20/20 пар. Ни одно семейство не прошло preregistered gate: combined baseline/candidate `7/20 → 9/20`, semantic `16/20 → 15/20`, agent tokens `195 547 → 344 818`; `epistemic_state_machine` дал regression с HTTP 500. Default не менять. Подробности — `LIVE_OPERATION_PROTOCOL_EXPERIMENT_REPORT_2026-09-10.md`.
- Расширения handoff-схемы `source_total`, 3 и 8 sample rows проверены на GigaChat-2-Max и отклонены: они увеличивали prompt, но не решали последовательный перебор кандидатов. Handoff остаётся компактным; planner читает полный результат через `read_previous_result`.
- `search_s2t_transformations` принимает совместимый одиночный `needle` и batch `needles` до 50 технических имён. Для набора из прошлого результата planner должен сделать один batch-вызов; исходные S2T-дубликаты сохраняются.
- Downstream prompt содержит компактные возможности чтения и краткие описания публичных таблиц. Эксперимент с сильно сокращёнными правилами и полными списками колонок откатан: на GigaChat-2-Max он заменил семантический поиск лексическим S2T-поиском.
- Текущий downstream system prompt занимает 3502 символа: 2180 правил, 520 описания возможностей и 802 справка по таблицам.
- Корректный план «семантический каталог → технические имена → точный S2T» получался в live-прогонах GigaChat-3-Ultra `20260826_124348`, Lazy Dependencies `20260827_0430` и GigaChat-2-Max `20260827_084904`. Последний — текущий целевой вариант из двух workers.
- Эксперимент GigaChat-2-Max `20260827_090853` с сокращённым prompt и полными списками колонок был семантически хуже: план начинался с лексического S2T-поиска по русскому бизнес-тексту. Изменения откатаны.
- После batch-изменения 192 релевантных unit-теста прошли. Изолированный GigaChat-2-Max прогон с фиксированным semantic-result из 10 колонок завершился двумя data-вызовами (`read_previous_result` и один batch S2T search) при компактной схеме. Полный HTTP-прогон `20260827_100855` провалился раньше handoff: downstream снова выбрал лексический поиск по бизнес-терминам вместо семантического каталога; это отдельная неустойчивость плана.
- LLM-as-judge отделён от агентной модели: для GigaChat по умолчанию используется `GigaChat-2-Pro`, независимо от `GIGACHAT_MODEL`; переопределение — `GIGACHAT_JUDGE_MODEL`. Для запросов по сохранённым данным отдельный короткий structured-вызов сначала извлекает неподтверждённые физические идентификаторы, затем основной judge оценивает маршрут и полноту.

## Основные компоненты

- `app.py` — Flask API, загрузка файлов, просмотр данных и `/chat`.
- `processing/excel.py`, `sheet_skills/` — механический Excel-разбор и доменное извлечение каталогов/S2T.
- `storage/database.py`, `storage/s2t.py` — публичная SQLite-схема и транзакционная запись.
- `services/graph_sync.py`, `graph_storage/` — производная Neo4j-проекция lineage; исходные факты остаются в SQLite.
- `agents/supervisor.py` — выделяет самодостаточную task и устойчивый context; не хранит неявный активный файл.
- `agents/coordinator.py` — downstream-план, последовательный запуск workers, два upstream-этапа и reroute.
- `agents/worker.py`, `agents/chat_graph.py` — router/planner/tool/observer/finish-worker цикл.
- `agents/tools/routing.py` — компактные каталоги и structured selection tools/skills/schemas.
- `agents/tools/saved_results.py` — временные полные результаты, `read_previous_result` и read-only SQL над сохранённой relation `result`.
- `agents/tools/s2t.py` — точные S2T-фильтры и ролево-нейтральный batch-поиск по техническим именам из предыдущих результатов.
- `agents/entity_resolution.py` — общий exact/normalized/partial/fuzzy/semantic resolver с явными resolved/ambiguous/unresolved outcomes.
- `agents/test_protocol_resolution.py` — разрешение пользовательского `RawTestProtocolContract` в канонический `ResolvedTestProtocolContract`.
- `agents/validation_protocol.py` — exact S2T и dependency-based catalog readers; legacy S2T-analysis readers остаются отдельно.
- `agents/transformation_ast.py`, `agents/test_protocol.py` — SQLGlot-нормализация и declarative phased compiler validation protocol.
- `agents/run_metrics.py`, `scripts/run_live_agent_benchmark.py` — пассивная трассировка и последовательные real-HTTP live-сценарии.
- `scripts/run_multiagent_experiments.py`, `scripts/run_multiagent_holdout.py`, `scripts/run_operation_protocol_experiments.py`, `scripts/gigachat_budget.py` — E1–E5 matrix, preregistered Max-only multiagent A/B, isolated operation-protocol matrix и fail-closed token guard для Ultra.

## Поток агентного запроса

1. `supervisor` один раз решает, нужен ли доступ к данным. Он передаёт coordinator текущую самодостаточную `task` и отдельный устойчивый `context` до 4000 символов; история, tools и план в context не копируются.
2. `downstream` возвращает обязательный native call `submit_worker_plan` с 1–8 `PlanStep(task)`. В plan нет tools/skills/schemas и отдельных задач анализа, сравнения или оформления.
3. Workers выполняются последовательно. Coordinator добавляет каждой последующей task короткие references всех принятых результатов предыдущих workers; содержимое читается явно через `read_previous_result` либо анализируется через `query_saved_result`. Когда строки задают несколько однотипных входов, planner использует batch-аргумент соответствующего tool, а не расходует цикл на последовательный перебор. Prompt требует не использовать handoff, если следующая task исполнима прямо из исходного запроса.
4. Внутри worker router одним structured вызовом независимо выбирает списки tools, retrieval-skills и schemas. Они могут быть пустыми; выбранные skills/schemas подгружаются лениво. После двух невалидных ответов используется ограниченная общая read-only палитра; при reroute новая палитра дополняет прежнюю.
5. Planner вызывает data-tool. После каждого результата observer проверяет соответствие именно worker-task и возвращает `complete`, `continue` или `reroute`; статуса `blocked` нет. Невалидный structured observer-output повторяется до пяти раз на том же payload без повторного data-tool.
6. `finish_worker` завершает worker и отдаёт summary, факты и только принятые evidence. Подтверждённые факты ссылаются на принятый `evidence_id`.
7. После всех workers upstream получает только `original_task` и evidence: `evidence_id`, tool name, точные args, preview, `truncated` и булевый признак `displayable`. Внутренний `display_ref`, worker observations, summaries и runtime refs туда не передаются.
8. `submit_upstream_data_decision` выбирает `pass` или `reroute`. Reroute сбрасывает результаты текущего цикла и передаёт следующему downstream только краткий `problem`; максимум два полных цикла. После `pass` отдельный `submit_upstream_answer` анализирует evidence, формирует ответ и выбирает display evidence.

## Поток validation protocol

1. Operation router выбирает `validation_protocol`; LLM извлекает только literal mentions, requested checks/mode и optional explicit key в `RawTestProtocolContract`.
2. Origin validation запрещает добавлять в raw contract идентификаторы, которых нет в исходной task. Exact verifier пропускает подтверждённые canonical names без approximate resolution; остальные mentions обрабатывает общий resolver с сохранением file/source/target role.
3. Ambiguous или unresolved entity возвращает `ambiguous_entity`/`unresolved_entity`, а повторно невалидный extraction — структурированный failure. В этих случаях `silent_fallback=false`; agentic pipeline автоматически не запускается.
4. `ResolvedTestProtocolContract` содержит canonical loads, optional `file_id`, checks, mode, explicit key и resolution metadata. Файл не обязателен.
5. `read_test_protocol_inputs` всегда подтверждает направленные source→target S2T-пары, а target/source catalogs читает только по зависимостям выбранных checks. Для единственного `row_count` полный target read и catalogs не нужны.
6. Transformation SQL нормализуется через SQLGlot. Phase 0 проверяет mapping coverage, catalog/projection consistency, required fields, requested sources, parseability и ambiguity.
7. Declarative `CHECKS` registry компилирует Phase 1 smoke, Phase 2 reconciliation и Phase 3 diagnostics. Поддержаны `row_count`, `key_uniqueness`, `required_null_rate`, `transformation_correctness`, `key_reconciliation`, `missing_rows`, `extra_rows`, `field_mismatch`, `schema_compatibility`, `expected_required_nulls`, `aggregate_reconciliation`, `duplicate_expected`, `duplicate_actual`.
8. `explicit` использует только запрошенные checks, `standard` — четыре базовых, `exhaustive` — все 13. Explicit comparison key имеет приоритет над target PK; без обоих key-based checks становятся unavailable. SQL templates используют разные `{{SOURCE_SCOPE_PREDICATE}}` и `{{TARGET_SCOPE_PREDICATE}}`.
9. Compiler возвращает machine-readable check statuses `ready|partial|unavailable`, protocol status `ready|partial_protocol|unavailable`, issues, preflight и phase summaries; SQL во внешней Greenplum не исполняется.

## Результаты tools и наблюдаемость

- Каждый принятый полный tool-result сохраняется на время coordinator-запуска под непрозрачным `result_id`; `display_ref` хранится отдельно от текстового preview.
- Табличный результат дополнительно получает `result_ref`, список колонок и `truncated`; `query_saved_result` исполняет read-only SQL только над выбранной relation `result`.
- Хранилище удаляется после coordinator и не пишет во внешнюю `excel_data.db`.
- `agents/run_metrics.py` пишет этапы `supervisor`, `downstream_plan`, `router`, `worker_planner`, `observer`, `finish_worker`, `upstream`, bounded-решение supervisor, планы, маршруты, observations, display-tools, entity-resolution events, validation contract/status/phases/reader calls, время и provider tokens. Полные tool-results в метрики не копируются.
- `agents/observability.py` содержит необязательную Langfuse-интеграцию; `logs/agent.log` — ротационный UTF-8 лог.

## SQLite-данные

Публичные таблицы: `files`, `file_sheet_headers`, `source_tables`, `target_tables`, `source_columns`, `target_columns`, `additional_objects`, `pxf_to_a`, `s2t_transformations`, `data`. Для обычных вопросов про «таблицы», DDL и схему показывать именно их, а не служебную реализацию SQLite.

- `files`: `file_id`, имя файла, модель, время загрузки, summary, description и embedding.
- `file_sheet_headers`: файл, лист, статус пропуска, координаты заголовка, структура и `headers_json`.
- `source_tables`/`target_tables`: `id`, `file_id`, `sheet_name`, `row_num`, `table_name`, `description`, embedding. Одинаковые строки сохраняются отдельно.
- `source_columns`/`target_columns`: происхождение, `table_name`, `column_name`, `data_type`, `primary_key`, `not_null`, `description`, embedding. Embedding строится из размеченных имени и описания, а при пустом описании — только из имени.
- `s2t_transformations`: происхождение, `target_field`, `source_field`, `target_table`, `source_table`, `transformation_rule`, nullable `source_layer`/`target_layer`; включает сырой S2T и lineage из Additional objects.
- `additional_objects`: происхождение, имя и полный SQL; `pxf_to_a`: происхождение и external/materialized/replica/SOD.
- `data`: полные значения Excel с `file_id`, именем листа, строкой и `column_id`; SQL и длинные значения не обрезаются.
- Aliases относятся только к физическим заголовкам Excel и хранятся в `config/column_mapping.json`, не в ETL-каталогах.

## Excel/S2T extraction

- `processing/excel.py` механически определяет preview/header и сохраняет сырые строки без доменной очистки.
- Перед полезными колонками sheet-group resolver делает exact/fuzzy по `config/sheet_groups.json`, затем один LLM-вызов только для несопоставленных листов и сохраняет подтверждённые aliases.
- `usefull_col_extraction` и каталоги колонок используют один resolver ролей: exact/fuzzy по `config/column_mapping.json`, затем максимум один LLM-вызов на лист, если сматчились не все настроенные роли. Невалидный LLM-ответ не создаёт частичный каталог.
- В LLM передаются только `sheet_name`, настроенные mapping fields, плоское `column_name` и samples. Не передавать `file_id`, `column_id`, индекс, полный header path, valid/critical/nullable роли и эвристики matching.
- Специализированные source/target column-листы имеют приоритет; отсутствующие колонки и атрибуты добираются из сырых S2T-строк в `data` с учётом стороны одинаковых заголовков.
- `not_null`, `primary_key` и типы сохраняются как нормализованные значения; неизвестное остаётся `NULL`, а не угадывается. Конфликтующие описания отражаются в отчёте и не превращаются в aliases ETL-колонки.
- `source_layer`/`target_layer` вычисляются детерминированно по группе листа через `config/table_layers.json`, не по именам таблиц и не через LLM.

## Обязательные правила

- Чат read-only по умолчанию. Загрузка, refresh и очистка требуют явного действия пользователя.
- Не придумывать `file_id`, листы, таблицы, колонки, S2T-строки и роли source/target; получать их из tools/SQL/evidence.
- Не хранить неявный активный `file_id`. Глобальную `s2t_transformations` никогда не ограничивать файловым `file_id`.
- В validation pipeline не передавать typo/partial/semantic mentions напрямую в exact readers: сначала общий resolver. Already-canonical identifier проверять exact bypass; ambiguity сохранять и не угадывать.
- Не смешивать raw пользовательские mentions и canonical identifiers: LLM формирует `RawTestProtocolContract`, readers/compiler получают только `ResolvedTestProtocolContract`.
- Не делать silent fallback `validation_protocol → agentic` после ошибки extraction или entity resolution. Возвращать structured `unresolved_entity`, `ambiguous_entity`, `missing_parameter`, `unsupported_check`, `partial_protocol` или `unavailable_check`.
- Не требовать файл для S2T-only validation checks. Catalog читать только по declarative dependencies; отсутствие file scope делает недоступными только зависимые checks.
- Validation SQL не исполнять. Явный comparison key приоритетнее catalog PK; source и target scope не объединять в один predicate.
- Логические ETL-таблицы вида `t_*` искать в S2T, а не через SQLite `PRAGMA`.
- Точную пару `source_table.source_field → target_table.target_field` сначала читать через точные ролевые S2T-фильтры; Neo4j использовать для lineage, а не вместо S2T.
- Для колонок: точный поиск — каталог, одна явно данная буквальная подстрока — каталоговый search, бизнес-смысл/назначение/описание при неизвестном имени — semantic search; при неизвестной роли искать source и target. Не заменять смысловой запрос набором подстрок, синонимов или переводов.
- Для Additional objects использовать точный/подстрочный read-only поиск с полным SQL. `trace_transformation_path` предназначен для связанного S2T-пути.
- Для точной известной S2T-роли/таблицы использовать точный list-инструмент; для фрагмента или неизвестной роли — search; произвольный `run_sql` оставлять нестандартным срезам, не покрытым готовыми tools.
- Полная SQL-строка анализируется SQL-инструментами; сохранённая `table.column` сначала ищется в S2T/Neo4j. Не выдавать текст transformation rule за исполняемый SQL.
- Не дедуплицировать одинаковые строки исходного Excel. Отсутствующий `target_table` в S2T — ошибка до транзакции.
- Колонковые листы сопоставлять exact/fuzzy по config, затем максимум одним LLM-вызовом на неполный лист; не отправлять LLM внутренние ID и эвристические служебные поля. Подтверждённые новые названия заголовков сохранять как aliases.
- Если на колонковом листе нет `table_name`, разрешать его только по однозначному совпадению `column_name` на сыром S2T-листе; иначе сохранять без придуманной таблицы и сообщать в отчёте.
- Additional objects после сохранения разбирать SQLGlot с Greenplum-диалектом, включая CTE и вложенные SELECT. Для промежуточных scope хранить `NULL → NULL`; только связи в конечную таблицу получают `NULL → B`. Ошибка одного объекта не останавливает остальные.
- Новые agent tools оформлять через `@tool(parse_docstring=True)` с русским docstring и типами, регистрировать в `agents/tools/registry.py` и покрывать тестами.
- Использовать существующие паттерны, UTF-8 и настроенный `LLM_PROVIDER` (`gigachat`, `openrouter`, `ollama`); не добавлять молчаливые non-LLM записи при невалидном ответе модели.

## Проверка

- UI/API: `tests/test_app.py`.
- Tools и агентная логика: `tests/test_agent_tools.py`, `tests/test_agent.py`, `tests/test_worker.py`, `tests/test_coordinator.py`.
- Хранение и S2T: `tests/test_database.py`, `tests/test_s2t_transformations.py`.
- Unit: `pytest tests/ -q`; покрытие: `pytest tests/ --cov=. --cov-config=.coveragerc`.
- Live quality проверяется отдельно через `scripts/run_live_agent_benchmark.py`: real-HTTP сценарии идут последовательно, сохраняют transcript/JUnit/comparison report; `--group` с именем `smoke`, `history`, `display`, `handoff`, `graph`, `validation`, `resolution` или `catalog` запускает только выбранную смысловую группу. `LIVE_AGENT_DB_PATH` задаёт существующую внешнюю SQLite-базу; без него используется workspace `excel_data.db`. Mock-тесты не оценивают качество модели.
- Bounded E1–E5 запускаются `scripts/run_multiagent_experiments.py`; `--experiment` можно повторять, без него выполняется вся матрица. Основные flags: `--provider`, `--model`, `--llm-judge`, `--pytest-arg`, `--output-dir`, `--dry-run`, `--ultra-token-floor`, `--ultra-reserve-per-scenario`. E1 переключает `WORKER_CAPABILITY_REROUTE_EXPERIMENT`/`WORKER_SPLIT_TOOL_CALL_EXPERIMENT`, E2 — `OPERATION_SQL_RISK_ASPECTS_EXPERIMENT`; E3–E5 проверяют текущие compiler/readers/resolver.
- Независимый holdout запускается `scripts/run_multiagent_holdout.py`: ровно 10 фиксированных сценариев вне E1–E5, обе руки только `multiagent`, agent/judge `GigaChat-2-Max`, обязательные hard+semantic проверки, counterbalanced AB/BA и неизменный SHA256 live DB. `--dry-run` проверяет preregistration и fixture без сетевых вызовов.
- Operation-protocol matrix запускается `scripts/run_operation_protocol_experiments.py`: ровно 20 fixed variants (`<aspect>__<family>`) и 40 paired exchanges, только multiagent и Max/Max semantic judge. До live-run обязателен clean committed HEAD; каждый pair работает в отдельном clone и на копиях SQLite, а default остаётся неизменным. Восемь `value_changes`/`write_semantics` cells из-за deterministic final-answer guard не являются full five-stage comparison.
- Для GigaChat Ultra benchmark и experiment runner обязаны fail-closed проверить баланс до и после run. Жёсткий floor 15 000 000 и базовый резерв 250 000 токенов на фактический HTTP `/chat`-обмен нельзя понизить CLI/env, только повысить; Ultra judge резервируется отдельно с учётом двух стадий и retries. Невозможность подтвердить balance или прогноз ниже floor блокирует дальнейший запуск. Benchmark допускает env overrides `GIGACHAT_ULTRA_TOKEN_FLOOR` и `GIGACHAT_ULTRA_RESERVE_PER_SCENARIO` в этих границах.
- `--llm-judge` оценивает текущий запрос, role-aware историю, публичный answer и ограниченные display-results. Пользовательские сообщения истории являются условиями, неподтверждённый текст assistant — нет. Новые физические идентификаторы в основанном на сохранённых данных SQL требуют подтверждения display либо явного placeholder; маршрут `A → B` должен достигать точного `B`. Не использовать judge как замену ручному разбору плана и evidence.
- Запуск UI: `uv run python app.py`.
