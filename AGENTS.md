# AGENTS.md

## Назначение

ETL S2T Parser разбирает Excel-файлы с ETL/S2T-описаниями, сохраняет исходные данные и каталоги в SQLite, строит Neo4j-lineage и отвечает на вопросы через read-only инструментального агента.

## Текущее состояние — 2026-08-27

- Ветка: `codex/multiagent-worker-experiment`; рабочее дерево содержит незакоммиченные изменения агентной части и пользовательский каталог `artifacts/` — не удалять и не перезаписывать их.
- `/chat` по умолчанию использует multiagent; `CHAT_AGENT_MODE=single_agent` оставлен как baseline.
- Актуальный поток: `supervisor → downstream plan → workers → upstream decision → upstream answer`.
- Downstream создаёт полный план из 1–8 задач чтения; coordinator допускает максимум два цикла. Workers идут последовательно и могут лениво читать принятые результаты предыдущих workers.
- Router одновременно выбирает tools, retrieval-skills и schemas; planner вызывает выбранные tools; observer проверяет каждый data-tool result и возвращает только `complete`, `continue` или `reroute`.
- Upstream получает исходную задачу и принятые evidence, решает `pass/reroute`, затем анализирует данные, формирует ответ и выбирает display-results.
- Полные tool-results живут только в run-scoped хранилище; последующим workers передаются короткие `result_id`/schema references. SQLite проекта не изменяется.
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
- `agents/run_metrics.py`, `scripts/run_live_agent_benchmark.py` — пассивная трассировка и последовательные real-HTTP live-сценарии.

## Поток агентного запроса

1. `supervisor` один раз решает, нужен ли доступ к данным. Он передаёт coordinator текущую самодостаточную `task` и отдельный устойчивый `context` до 4000 символов; история, tools и план в context не копируются.
2. `downstream` возвращает обязательный native call `submit_worker_plan` с 1–8 `PlanStep(task)`. В plan нет tools/skills/schemas и отдельных задач анализа, сравнения или оформления.
3. Workers выполняются последовательно. Coordinator добавляет каждой последующей task короткие references всех принятых результатов предыдущих workers; содержимое читается явно через `read_previous_result` либо анализируется через `query_saved_result`. Когда строки задают несколько однотипных входов, planner использует batch-аргумент соответствующего tool, а не расходует цикл на последовательный перебор. Prompt требует не использовать handoff, если следующая task исполнима прямо из исходного запроса.
4. Внутри worker router одним structured вызовом независимо выбирает списки tools, retrieval-skills и schemas. Они могут быть пустыми; выбранные skills/schemas подгружаются лениво. После двух невалидных ответов используется ограниченная общая read-only палитра; при reroute новая палитра дополняет прежнюю.
5. Planner вызывает data-tool. После каждого результата observer проверяет соответствие именно worker-task и возвращает `complete`, `continue` или `reroute`; статуса `blocked` нет. Невалидный structured observer-output повторяется до пяти раз на том же payload без повторного data-tool.
6. `finish_worker` завершает worker и отдаёт summary, факты и только принятые evidence. Подтверждённые факты ссылаются на принятый `evidence_id`.
7. После всех workers upstream получает только `original_task` и evidence: tool name, точные args, preview, `truncated` и безопасный display ID. Worker observations, summaries и runtime refs туда не передаются.
8. `submit_upstream_data_decision` выбирает `pass` или `reroute`. Reroute сбрасывает результаты текущего цикла и передаёт следующему downstream только краткий `problem`; максимум два полных цикла. После `pass` отдельный `submit_upstream_answer` анализирует evidence, формирует ответ и выбирает display evidence.

## Результаты tools и наблюдаемость

- Каждый принятый полный tool-result сохраняется на время coordinator-запуска под непрозрачным `result_id`; `display_ref` хранится отдельно от текстового preview.
- Табличный результат дополнительно получает `result_ref`, список колонок и `truncated`; `query_saved_result` исполняет read-only SQL только над выбранной relation `result`.
- Хранилище удаляется после coordinator и не пишет во внешнюю `excel_data.db`.
- `agents/run_metrics.py` пишет этапы `supervisor`, `downstream_plan`, `router`, `worker_planner`, `observer`, `finish_worker`, `upstream`, планы, маршруты, observations, display-tools, время и provider tokens. Полные tool-results в метрики не копируются.
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
- Live quality проверяется отдельно через `scripts/run_live_agent_benchmark.py`: real-HTTP сценарии идут последовательно, сохраняют transcript/JUnit/comparison report; mock-тесты не оценивают качество модели.
- `--llm-judge` оценивает только исходный запрос, публичный answer и ограниченные display-results. Новые физические идентификаторы в основанном на сохранённых данных SQL требуют подтверждения display либо явного placeholder; маршрут `A → B` должен достигать точного `B`. Не использовать judge как замену ручному разбору плана и evidence.
- Запуск UI: `uv run python app.py`.
