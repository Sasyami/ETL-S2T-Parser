# AGENTS.md

## О проекте

ETL S2T Parser - приложение для разбора Excel-файлов с S2T-маппингами: определяет структуру заголовков, сохраняет метаданные и примеры строк в SQLite, строит бизнес-саммари через настроенный LLM provider, сопоставляет листы с целевой S2T-схемой и отвечает на вопросы через инструментального агента.

## Основные команды

- Запуск Flask UI: `uv run python app.py`
- Тесты: `pytest tests/ -q`
- Тесты с покрытием: `pytest tests/ --cov=. --cov-config=.coveragerc`

## Карта архитектуры

- `app.py` - тонкий Flask API: маршруты загрузки, progress, summary, S2T transformations и chat; `/chat` по умолчанию запускает multiagent supervisor, а `CHAT_AGENT_MODE=single_agent` включает немультиагентный baseline для сравнительных live-прогонов.
- `processing/excel.py` - механический разбор загруженного Excel: preview, существующее определение заголовков, построение колонок и чтение сырых строк без доменной очистки.
- `agents/agent.py` - определение заголовков Excel и входная точка исходного немультиагентного chat agent; baseline использует тот же tool router и общий planner/tool/observer/responder LangGraph.
- `agents/header_classifier.py` - выбор строки заголовка переданной CatBoost-моделью по 22 строковым признакам.
- `agents/tools/routing.py` - отдельный LLM tool-router: модель получает раздельные компактные каталоги tools, skills и schemas и через `with_structured_output(ToolRoute, method="function_calling")` возвращает строго валидируемые списки; все три списка выбираются одновременно и независимо, каждый из них может быть пустым независимо от остальных, а выбранные runtime skills и фактические схемы/маппинги загружаются лениво без группового или эвристического fallback.
- `agents/chat_graph.py` - общий многошаговый LangGraph с нативным tool calling: observer возвращает structured `Observation` с отдельными `goal_satisfied`, `mismatches`, summary, фактами и ограничениями; невалидный structured observer один раз повторяется на том же сохранённом tool result без повторного data-tool вызова, а второй невалидный ответ завершает worker явной ошибкой. Legacy chat после planner использует responder, а изолированный worker завершается самим planner через `finish_worker(answer)` либо его обычный финальный текст без отдельного LLM-responder. В worker-режиме каждый planner-вызов получает исходную task, только последний `AIMessage.tool_calls`/`ToolMessage` обмен и одну накопительную observer-выжимку; при `goal_satisfied=false` planner отдельно получает все `mismatches`; worker не выбирает результаты для UI.
- `agents/worker.py` - экспериментальный изолированный read-only worker: получает только самодостаточную подзадачу, внутри одновременно выбирает tools/skills/schemas, лениво загружает выбранные схемы и запускает planner/tool/observer-цикл; если router вернул `tools=[]`, worker подставляет внутренний `analyze_known_facts`, который не читает данные, а переносит готовый анализ в обычный ToolMessage для observer, поэтому каждый worker-цикл содержит хотя бы один tool. В LLM-контексте инструментального цикла остаётся единое ограниченное текстовое preview tool result, полные успешные результаты хранятся отдельно, а публичный worker-ответ содержит краткий LLM-ответ, компактную ordered-историю циклов с tool calls, теми же ограниченными preview и structured observations, временные непрозрачные UI result refs и схемы run-scoped сохранённых SQLite-результатов без полных строк.
- `agents/tools/saved_results.py` - временная SQLite-база одного coordinator-запуска: табличные результаты SQLite-tools материализуются ровно в фактически возвращённом объёме с `result_ref`, схемой и признаком `truncated`; `query_saved_result` выполняет произвольный read-only SQL только над выбранной relation `result`, а её схема динамически передаётся вместе с tool следующему worker. База удаляется после завершения coordinator и не изменяет `excel_data.db`.
- `agents/coordinator.py` - coordinator разделён по направлению потока. Downstream-путь `downstream_plan -> downstream_materialize -> worker` проводит исходную цель сверху вниз: строит семантический план и материализует самодостаточные worker tasks. Upstream-путь `worker -> upstream_evidence -> upstream_answer` проводит результаты снизу вверх: сначала формирует structured `confirmed_facts`/`unresolved_requirements` без пользовательского оформления, затем отдельный узкий prompt оформляет окончательный ответ по исходной task и проверенному evidence. Полные result refs для `presentation=full_results` выбираются детерминированно, без LLM. Полные tool results в state и LLM-контекст coordinator не передаются.
- `agents/supervisor.py` - верхний supervisor LangGraph `supervisor -> coordinator -> END`: supervisor один раз решает, нужен ли доступ к данным, и при необходимости передаёт coordinator конкретную самодостаточную task и отдельный context. Task содержит текущее исполнимое поручение и все нужные ему разовые факты с уже разрешёнными ссылками; context содержит только устойчивые правила, определения, предпочтения, инварианты и общие ограничения, установленные в диалоге, без пересказа task и истории. Context ограничен 4000 символами и не содержит tools, skills или план. Неявного выбранного/активного файла в состоянии агента нет; конкретный файл определяется только из явного запроса или однозначной ссылки в истории и встраивается в task. Зависимые продолжения выполняются внутри coordinator-плана; агрегированный coordinator-ответ завершает запрос без повторного supervisor-аудита, а выбранные display refs разрешаются только на границе ответа UI.
- `agents/run_metrics.py` - включаемая только для live-проверок пассивная трассировка одного supervisor-запуска: фактические LLM/tool-вызовы, workers, coordinator plan, выбранные display tools, длительность и provider token usage; полные tool results в метрики не копируются, завершённый снимок забирается тестом по `session_id`.
- `scripts/run_live_agent_benchmark.py` - последовательный запуск одинаковых real-HTTP live-сценариев для `multiagent`/`single_agent` на выбранном LLM provider с отдельными transcript/JUnit и итоговым сравнительным Markdown-отчётом; запросы не пакетируются и не выполняются параллельно.
- `agents/summarizer_agent.py` - извлечение семантического каталога и один LLM-вызов для summary/description.
- `sheet_skills/s2t.py` - sheet skill `usefull_col_extraction`: inspect, deterministic/LLM matching, валидация строк и построение записей S2T.
- `sheet_skills/table_catalog.py` - sheet skill каталогов: построчно сохраняет `table_name` и `description` из групп `source_tables`/`target_tables`.
- `sheet_skills/column_catalog.py` - строит `source_columns`/`target_columns`: заголовки специализированных листов сопоставляются общим механизмом exact/fuzzy по `column_mapping.json`, затем одним LLM-вызовом для листа с неполным набором ролей и сохранением подтверждённых новых aliases заголовков; специализированные листы имеют приоритет, недостающие колонки и атрибуты добираются из сырых строк S2T в `data` с учётом расположения одинаковых заголовков в source/target-блоках; конфликтующие описания одной `table_name`/`column_name` отражаются в отчёте, но не превращаются в aliases ETL-колонки.
- `sheet_skills/structured_metadata.py` - sheet skill для `additional_objects` и `pxf_to_a`; `sheet_skills/additional_objects.py` преобразует SQL дополнительных объектов в строки общей ETL-таблицы, общая механика сопоставления полей находится в `sheet_skills/configured_rows.py`.
- `storage/s2t.py` - транзакционная запись, очистка, чтение, поиск/агрегация, проверка S2T-записей и детерминированный backfill ETL-слоёв.
- `storage/database.py` - актуальная публичная SQLite-схема: `files`, `file_sheet_headers`, `source_tables`, `target_tables`, `source_columns`, `target_columns`, `additional_objects`, `pxf_to_a`, `s2t_transformations`, `data`.
- `graph_storage/` - изолированные настройки и lifecycle Neo4j driver.
- `services/graph_sync.py` - пересобирает Neo4j-проекцию lineage одного файла: `ETLColumn`/`TRANSFORMS_TO` и `ETLTable`/`TABLE_TRANSFORMS_TO`; исходные факты остаются в SQLite.
- `config/` - загрузчики и JSON-конфигурации sheet groups, column mappings и useful-column extraction.
- `agents/tools/` - тематические модули с декорированными `@tool` и отдельный read-only/write registry.
- `agents/observability.py` - необязательная интеграция Langfuse.
- `services/logging_setup.py` - единая настройка console logging и ротационного UTF-8 файла `logs/agent.log`.
- `agents/prompts/` - runtime skills и контекст chat-agent.
- `templates/chat_app.html` - основной chat-first веб-интерфейс; корневой маршрут `/` и совместимый `/chat_app` показывают одну страницу с чатом, загрузкой и просмотром данных.

## Правила для доработок

- Сначала проверяй существующие паттерны проекта; не вводи новый фреймворк без явной пользы.
- Храни факты о файлах, листах, колонках и маппингах в SQLite через существующий слой `storage/database.py`.
- Не придумывай `file_id`, имена листов, колонок, ETL source/target tables или S2T rows: получай их через инструменты или SQL.
- Для логических ETL/S2T-таблиц вида `t_*` не используй `PRAGMA` как для физических SQLite-таблиц; ищи их в `s2t_transformations.target_table` и связанных строках трансформаций.
- Перед добавлением нового инструмента агента используй `@tool(parse_docstring=True)`, русский docstring и типы; затем явно добавляй готовый `BaseTool` в `agents/tools/registry.py` и обновляй тесты.
- Runtime LLM-provider для агентной части и S2T matching настраивается через `LLM_PROVIDER`; по умолчанию используется `gigachat`. Альтернативы: OpenAI-compatible `openrouter` или локальный `ollama`. Не добавляй молчаливые non-LLM fallback-записи: если LLM не дал валидный план, возвращай ошибку.
- Unit-тесты могут мокать транспорт LLM для проверки обвязки, но это не считается проверкой качества модели. Реальные проверки модели выноси в отдельные integration/smoke тесты с явным запуском.
- Сохраняй русские промпты и документацию в UTF-8. Не переписывай русские строки только из-за mojibake в консоли PowerShell.
- Summarizer делает один LLM-вызов по семантическому каталогу: извлекает описания таблиц, представлений, атрибутов и полей из всех листов с данными (включая пропущенные, если строки сохранены) и передаёт их в LLM без сырых S2T-строк, SQL и метаданных об исключённых листах.

## Правила для агентной части

- Единый диалог должен быть read-only по умолчанию: отвечать про файлы, листы, заголовки, summary и S2T transformations.
- Мутации вроде загрузки файла, повторного S2T refresh или очистки `s2t_transformations` должны идти только через явное действие пользователя или подтверждение.
- Перед `usefull_col_extraction` запускай sheet-group subagent: exact/fuzzy по `config/sheet_groups.json`, затем LLM только для несматченных листов, затем запись новых алиасов в текущий `config/sheet_groups.json`. Шаг извлечения полезных колонок не должен сам решать группу листа по имени.
- Запись `s2t_transformations` выполняется через target `s2t_transformations` в `usefull_col_extraction.json`: subagent выбирает колонки в два шага - сначала exact/fuzzy по `column_mapping.json`, затем настроенный OpenAI-compatible LLM только для листов, где сматчились не все настроенные роли.
- `usefull_col_extraction` должен отправлять в настроенный LLM максимум один запрос на один неполностью сматченный лист и просить компактный ответ `column_roles`: каждой колонке по плоскому имени `column_name` сопоставляется ключ `mapping_field` из настроенной группы `column_mapping_json` или `null`.
- Для targets `source_columns`/`target_columns` применяй тот же column-role resolver: если exact/fuzzy не закрыл хотя бы одну настроенную роль, отправляй не больше одного LLM-запроса на лист, требуй полный валидный `column_roles`, сохраняй подтверждённые aliases и не записывай частичный каталог при ошибке ответа.
- Не отправляй LLM внутренние ID (`file_id`, `column_id`), `column_index`, полный `header_path`, valid/critical/nullable роли, `role_to_column_mapping_field` и эвристические подсказки для matching; в prompt достаточно `sheet_name`, настроенного `column_mapping_json`, плоского `column_name` и samples.
- Exact/fuzzy matching можно использовать внутри Python для evidence и первичного сопоставления. В prompt это не отправлять; если deterministic pass нашел все v1-роли, LLM для этого листа не вызывается.
- Если `usefull_col_extraction` подтвердил header с ролью не exact-ом, добавляй фактическое название header в `column_mapping.json` для соответствующего поля, без дублей.
- Для вопросов по маппингам и трансформациям используй `search_s2t_transformations`, `list_s2t_transformations` или read-only SQL по `s2t_transformations`; Neo4j используй только для lineage колонок.
- Для точных фильтров по `source_columns`/`target_columns` используй `list_column_catalog`, для подстроки — `search_column_catalog`, а для смысловой близости описаний — `semantic_search_descriptions` с колонковым scope и структурными фильтрами подвыборки.
- Если на специализированном листе колонок отсутствует `table_name`, не наследуй имя предыдущей строки: разрешай таблицу только по однозначной паре с тем же `column_name` на сыром S2T-листе; неоднозначные и ненайденные строки сохраняй без придуманной таблицы и отражай в отчёте.
- Не дедуплицируй строки `source_tables`, `target_tables`, `additional_objects`, `pxf_to_a` и `s2t_transformations`: одинаковые строки исходного Excel являются отдельными фактами.
- `source_layer` и `target_layer` в `s2t_transformations` вычисляй по семантической группе исходного листа детерминированными правилами из `config/table_layers.json`; имена таблиц для этого не анализируй и не добавляй слои в обязательные LLM-роли.
- После сохранения `additional_objects` разбирай каждый непустой SQL через SQLGlot с Greenplum-диалектом: создавай в `s2t_transformations` колонковые связи для выходов SELECT, включая вложенные SELECT/CTE. Для промежуточных scope additional objects сохраняй `NULL -> NULL`; только связи, входящие в конечную таблицу объекта, получают `NULL -> B`. Слой источника не определяй даже при наличии `source_table` в SQL.
- Не обрезай SQL и другие значения Excel при сохранении в `data`; ошибки одного additional object сохраняй в отчёте и продолжай обработку остальных объектов.
- Если в строке S2T отсутствует `target_table`, возвращай явную ошибку с листом и номером строки до начала транзакции; существующие `s2t_transformations` не изменяй.
- Чат не передаёт и не хранит неявный активный `file_id`. Для файловых tools используй только идентификатор, явно названный пользователем, однозначно полученный из истории или найденный через `list_files`/`resolve_file`. Глобальная `s2t_transformations` никогда не ограничивается таким `file_id`.

## Проверка изменений

- Для изменений Flask routes и UI-чата обновляй `tests/test_app.py`.
- Для tools и агентной логики обновляй `tests/test_agent_tools.py`.
- Для хранения S2T-строк обновляй `tests/test_database.py` и `tests/test_s2t_transformations.py`.
- После изменений запускай минимум релевантные тесты, а перед крупным merge - `pytest tests/ -q`.

## Актуальные уточнения агентной части

- Когда пользователь спрашивает про "таблицы", по умолчанию имеются в виду таблицы ETL/S2T-слоев и сохраненные результаты анализа этих слоев, прежде всего `s2t_transformations`, а не физические служебные таблицы SQLite.
- `data` входит в публичную схему и доступна для read-only анализа сохранённых значений строк Excel.
- Для обычных вопросов про DDL/схему показывай публичную ETL-схему: `files`, `file_sheet_headers`, `source_tables`, `target_tables`, `source_columns`, `target_columns`, `additional_objects`, `pxf_to_a`, `s2t_transformations` и `data`.

## Актуальная SQLite-схема

- Для текущей задачи показывай пользовательски значимые таблицы: `files`, `file_sheet_headers`, `source_tables`, `target_tables`, `source_columns`, `target_columns`, `additional_objects`, `pxf_to_a`, `s2t_transformations`, `data`.
- `source_tables` и `target_tables` хранят `id`, `file_id`, `sheet_name`, `row_num`, `table_name`, `description`; одинаковые исходные строки сохраняются отдельно.
- `source_columns` и `target_columns` хранят связанную с `table_name`/`column_name` метаинформацию: `data_type`, `primary_key`, `not_null`, `description` и `description_embedding`; embedding строится из размеченных `column_name` и `description`, а при пустом описании — только из имени. Происхождение строки определяется по `file_id`, `sheet_name` и `row_num`, отдельного `metadata_source` нет. Aliases относятся к физическим заголовкам исходных листов и хранятся в `column_mapping.json`, а не в ETL-каталогах. Отсутствующие специализированные листы и значения дополняются только из сырых строк S2T в `data`.
- `s2t_transformations` является общей таблицей колонковых ETL-связей: содержит строки исходного S2T и lineage, извлечённый из `additional_objects.sql`; nullable `source_layer` и `target_layer` вычисляются по группе исходного листа.
- `additional_objects` хранит `id`, extraction metadata, `name`, `sql`; `pxf_to_a` хранит extraction metadata, `external_a_table`, `materialized_storage`, `replica_table`, `sod`.
- `data` — публичная таблица для анализа и добора значений: `id`, `file_id`, `table_name`, `row_num`, `column_id`, `value`.
- `table_name` в `data` хранит имя листа/таблицы Excel, чтобы анализировать значения без лишнего join.
- Для обычных вопросов пользователя про таблицы, DDL или схему показывай `files`, `file_sheet_headers`, `source_tables`, `target_tables`, `source_columns`, `target_columns`, `additional_objects`, `pxf_to_a`, `s2t_transformations` и `data`.
