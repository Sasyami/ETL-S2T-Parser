# ETL S2T Agent

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Flask 3](https://img.shields.io/badge/Flask-3.x-green.svg)](https://flask.palletsprojects.com/)
[![LangGraph](https://img.shields.io/badge/agents-LangGraph-orange.svg)](https://www.langchain.com/langgraph)

ETL S2T Agent — chat-first приложение для загрузки и анализа Excel-файлов с Source-to-Target-маппингами. Оно сохраняет исходные факты в SQLite, извлекает S2T- и SQL-lineage, при наличии Neo4j строит графовую проекцию и отвечает на вопросы через многоагентный LangGraph.

> **Актуальный полный live-отчёт:** [39 сценариев — GigaChat-3-Ultra, 3 сентября 2026](LIVE_AGENT_39_RUN_REPORT_2026-09-03.md). Отчёт находится в корне репозитория и включает автоматические результаты, ручную семантическую проверку, повтор Neo4j-сценариев и причины оставшихся провалов.

## Возможности

- загрузка `.xlsx`, `.xls` и `.xlsm` из единого интерфейса чата;
- автоматический выбор строки заголовка CatBoost-моделью;
- сохранение заголовков и значений Excel без обрезки в SQLite;
- классификация листов и настраиваемое сопоставление колонок;
- извлечение S2T, каталогов таблиц, PXF-маппингов и дополнительных объектов;
- разбор SQL дополнительных объектов через SQLGlot;
- read-only вопросы к SQLite и Neo4j на естественном языке;
- полные табличные результаты в отдельном scrollable-блоке, а не в тексте чата;
- сравнение многоагентного режима с базовым одноагентным режимом;
- метрики времени, LLM-вызовов, инструментов и токенов для live-сценариев.

## Архитектура

SQLite является источником исходных фактов. Neo4j хранит только производную проекцию lineage и может быть отключён.

### Загрузка Excel

```mermaid
flowchart LR
    UI["Chat-first UI"] --> API["POST /upload"]
    API --> PARSE["Механический разбор Excel"]
    PARSE --> HEADER["CatBoost: строка заголовка"]
    HEADER --> SQLITE[("SQLite")]
    SQLITE --> GROUPS["Классификация групп листов"]
    GROUPS --> SKILLS["Sheet skills"]
    SKILLS --> SQLITE
    SQLITE --> SUMMARY["Summary и description"]
    SQLITE --> GRAPH["Neo4j projection (опционально)"]
```

`processing/excel.py` читает каждый лист один раз, сохраняет исходные номера строк, разворачивает объединённые ячейки данных и по умолчанию исключает скрытые строки. Включить их можно при загрузке в интерфейсе. Ответ загрузки содержит `data_row_count` для каждого листа и `total_data_row_count` для всей книги; это число разобранных строк данных после фильтрации скрытых строк, независимо от числа заполненных ячеек, а одинаковые строки считаются отдельно.

Строка заголовка выбирается среди первых десяти строк моделью из `models/catboost_header_model.cbm`. Кандидаты с тремя и более пустыми/`Untitled` значениями исключаются, если остаются менее разреженные строки. При ошибке CatBoost используется настроенный LLM-provider.

После механического разбора:

1. лист сопоставляется с группой из `config/sheet_groups.json`;
2. колонки сначала сопоставляются детерминированно по `config/column_mapping.json`;
3. LLM вызывается только для листа с неполным сопоставлением;
4. результат валидируется и транзакционно записывается в целевую таблицу.

Одинаковые строки Excel не дедуплицируются. Если в непустой S2T-строке отсутствует `target_table`, запись завершается явной ошибкой до начала транзакции. Строки без единого S2T-значения не считаются бизнес-строками.

### Многоагентный чат

По умолчанию `CHAT_AGENT_MODE=multiagent`.

```mermaid
flowchart TD
    Q["Запрос + история"] --> S["Supervisor"]
    S -->|данные не нужны| A["Прямой ответ"]
    S -->|нужны данные| OR["Operation router"]
    OR -->|обычный запрос| C["Downstream plan: 1–8 read tasks"]
    C --> W["Последовательные workers"]
    W --> R["Router: tools + retrieval skills + schemas"]
    R --> P["Planner → read-only tool"]
    P --> O["Observer каждого tool result"]
    O -->|continue| P
    O -->|reroute| R
    O -->|complete| F["WorkerOutcome + accepted evidence"]
    F -->|следующая task| W
    F --> U["Upstream data decision"]
    U -->|reroute, максимум один раз| C
    U -->|pass| UA["Upstream answer + display selection"]
    OR -->|S2T-анализ| SC["Typed contract → readers → analyzer"]
    OR -->|тест-протокол| VC["Typed contract → readers → compiler"]
    SC --> UI["Ответ + scrollable results"]
    VC --> UI
    UA --> UI
```

Основные контракты:

- supervisor отдельно формирует исполнимую `task` и устойчивый `context`;
- operation router один раз выбирает общий agentic-поток либо специализированный
  пайплайн и подключает только относящиеся к операции stage-skills;
- неявного «активного файла» нет: полный `filename` разрешается через
  `resolve_file`, а внутренний `file_id` берётся только из запроса или принятого
  результата разрешения; одна операция может ссылаться на несколько файлов;
- в общем потоке downstream сразу создаёт полный план из 1–8 задач чтения;
  workers выполняются последовательно и могут лениво прочитать принятые
  результаты предыдущих workers по коротким `result_id`;
- worker получает текущую задачу и короткие ссылки на доступные зависимости;
  router независимо выбирает tools, retrieval-skills и schemas, а planner
  вызывает только выбранные read-only tools;
- planner видит исходную задачу, последний обмен с инструментом и накопительную observer-выжимку;
- observer вызывается после каждого data-tool result и возвращает только
  `complete`, `continue` или `reroute`; невалидная структура повторно
  запрашивается на том же payload без повторного data-tool;
- полные результаты инструментов не копируются в историю worker: там остаётся ограниченный preview;
- upstream получает только `original_task` и принятые evidence: `evidence_id`, tool name, args, preview, `truncated` и булевый признак `displayable`; внутренний `display_ref`, worker summary, facts, limitations и runtime refs туда не передаются;
- принятые полные tool results сохраняются под run-scoped `result_id` и читаются через `read_previous_result` только когда краткого description недостаточно; табличные SQLite-результаты дополнительно материализуются во временной relation `result`, доступной через `query_saved_result`;
- если tool вернул только preview, схема сохранённого результата содержит `truncated=true`, поэтому его нельзя использовать как полный исходный набор;
- upstream сначала линейно вызывает `submit_upstream_data_decision`: обязательное поле `decision` равно `pass` или `reroute`, а `problem` служит необязательным пояснением. `reroute` запускает чистый повтор чтения со сбросом результатов прошлого цикла; `pass` переводит управление к отдельному `submit_upstream_answer`. Только этот второй вызов выполняет производный SQL/S2T-анализ по исходной task и evidence, формирует обязательный `answer` и опционально выбирает evidence IDs для UI. Отдельного semantic reviewer/repair нет;
- полные данные разрешаются по ссылкам только на границе HTTP-ответа.

Worker завершается самим planner только через native `finish_worker(summary)`. Обычный финальный текст отклоняется; полноту исходных данных определяет тот же structured observer.

Для повторяемых операций предусмотрены два специализированных маршрута:

- `s2t_analysis`: реализация typed S2T-анализа сохранена как экспериментальный
  резерв, но operation-router её не предлагает; обычные запросы анализа идут
  через общий agentic-поток и operation-skills;
- `validation_protocol`: typed contract задаёт загрузки и проверки,
  детерминированные readers получают S2T и метаданные колонок, а compiler строит
  Greenplum SQL-шаблоны и критерии прохождения без исполнения SQL во внешней БД.

Для impact по колонке `trace_neo4j_lineage` возвращает точные
`transformation_id`, а `get_s2t_rules_by_ids` одним параметризованным чтением
получает соответствующие S2T-правила. Planner не генерирует SQL для этого
перехода между Neo4j и SQLite.

Режим `single_agent` сохранён как базовая линия для live-сравнений:

```ini
CHAT_AGENT_MODE=single_agent
```

## Хранилище

По умолчанию SQLite создаётся в `excel_data.db`.

| Таблица | Назначение |
|---|---|
| `files` | загрузки, summary и description |
| `file_sheet_headers` | решения по заголовкам и плоские имена колонок |
| `data` | исходные значения Excel с `file_id`, листом, строкой и колонкой |
| `source_tables` | построчный каталог таблиц-источников |
| `target_tables` | построчный каталог целевых таблиц |
| `source_columns` | колонки источников: заголовки определяются по aliases из `column_mapping.json` и, если ролей не хватает, одной LLM-проверкой; хранятся тип, PK, not-null, описание и embedding из технического имени плюс описания; специализированный лист дополняется сырым S2T |
| `target_columns` | целевые колонки: заголовки определяются по aliases из `column_mapping.json` и, если ролей не хватает, одной LLM-проверкой; хранятся тип, PK, not-null, описание и embedding из технического имени плюс описания; специализированный лист дополняется сырым S2T |
| `additional_objects` | имя и полный SQL дополнительного объекта |
| `pxf_to_a` | внешняя, материализованная и репличная таблицы, СОД |
| `s2t_transformations` | общая таблица колонковых ETL-связей и правил |

Если на специализированном листе колонок отсутствует имя таблицы, оно
подставляется только при однозначном совпадении `column_name` на сыром S2T-листе.
Имя предыдущей строки не наследуется, а неоднозначность остаётся явной в отчёте.

`s2t_transformations` содержит как строки исходных S2T-листов, так и связи, извлечённые из `additional_objects.sql`. Для дополнительных объектов SQLGlot обрабатывает CTE, вложенные SELECT и set-операции; ошибки одного объекта попадают в отчёт и не останавливают остальные.

`source_layer` и `target_layer` определяются по группе листа правилами из `config/table_layers.json`, а не по имени таблицы и не через LLM.

### Neo4j

При настроенном подключении `services/graph_sync.py` пересобирает проекцию одного файла:

- `ETLColumn` и `TRANSFORMS_TO` — lineage колонок;
- `ETLTable` и `TABLE_TRANSFORMS_TO` — lineage таблиц.

Если Neo4j выключен или недоступен, SQLite-анализ сохраняется, а ошибка синхронизации возвращается отдельно. Для вопросов по S2T и трансформациям используется SQLite; Neo4j предназначен для путей и lineage.

## Быстрый запуск

Требования:

- Python 3.12+;
- [uv](https://docs.astral.sh/uv/);
- GigaChat, OpenRouter или локальный Ollama;
- Neo4j 5+ — только для графовых сценариев.

```bash
git clone https://github.com/Xpehutta/ETL-S2T-Parser.git
cd ETL-S2T-Parser
uv sync
```

Скопируйте `.env.example` в `.env`, заполните выбранный provider и запустите:

```bash
uv run python app.py
```

Интерфейс будет доступен на `http://127.0.0.1:5000`. Пути `/` и `/chat_app` открывают один и тот же chat-first экран с загрузкой файла, прогрессом анализа, чатом и просмотром полной таблицы трансформаций.

## Настройка LLM

По умолчанию используется GigaChat.

### GigaChat

```ini
LLM_PROVIDER=gigachat
GIGACHAT_API_KEY=your_key
GIGACHAT_MODEL=GigaChat
GIGACHAT_API_URL=https://api.giga.chat/v1
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_VERIFY_SSL=false
GIGACHAT_TIMEOUT=120
```

Вместо `GIGACHAT_API_KEY` поддерживаются `GIGACHAT_CREDENTIALS` и `GIGACHAT_EMBEDDINGS_CREDENTIALS`.

### Ollama

Модель должна поддерживать native tool calling и structured output.

```bash
ollama pull qwen3.5:9b
```

```ini
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3.5:9b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_NUM_CTX=16384
OLLAMA_TIMEOUT=120
OLLAMA_TEMPERATURE=0
OLLAMA_REASONING=false
```

### OpenRouter

```ini
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=openrouter/free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_TIMEOUT=120
OPENROUTER_TEMPERATURE=0
```

### Neo4j

```ini
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=change_me
NEO4J_DATABASE=neo4j
```

## Конфигурация извлечения

| Файл | Назначение |
|---|---|
| `config/sheet_groups.json` | группы листов и их алиасы |
| `config/column_mapping.json` | роли и варианты названий Excel-колонок |
| `config/usefull_col_extraction.json` | группа листа, целевая SQLite-таблица и поля |
| `config/table_layers.json` | переходы ETL-слоёв по группам листов |

Новые подтверждённые алиасы листов и заголовков добавляются в текущие JSON-конфигурации без дублей.

## HTTP API

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/`, `/chat_app` | chat-first UI |
| `POST` | `/upload` | загрузка и полный анализ Excel |
| `GET` | `/analysis_progress/<upload_id>` | прогресс загрузки |
| `POST` | `/chat` | запрос к выбранному агентному режиму |
| `GET` | `/summary/<file_id>` | summary файла |
| `GET` | `/description/<file_id>` | краткое описание файла |
| `GET` | `/transformations` | глобальная таблица S2T |
| `GET` | `/transformations/<file_id>` | S2T указанного файла |
| `DELETE` | `/transformations/<file_id>` | явная очистка S2T файла |
| `DELETE` | `/storage` | явная полная очистка хранилищ |
| `GET` | `/sheet_groups/<file_id>/classify` | классификация листов |
| `GET` | `/exports/...` | скачивание полных результатов |

История чата хранится в `sessionStorage` браузера и передаётся в `/chat`. В SQLite история не записывается.

## Тесты

Обычный набор не обращается к реальной модели:

```bash
pytest tests/ -q
pytest tests/ --cov=. --cov-config=.coveragerc
```

### Live-сценарии

Live-тесты используют реальный Flask `/chat`, текущую `excel_data.db`, выбранный provider и запущенный Neo4j для графовых сценариев. Supervisor, coordinator, workers, router, tools, observer и aggregator не подменяются. Запросы выполняются строго последовательно, без batching и параллельного pytest.

Опциональный `--llm-judge` после каждого ответа отдельным LLM-вызовом оценивает только исходный запрос, публичный answer и display-results, записывает semantic verdict в transcript/comparison report и валидирует сценарий: `failed` или ошибка judge переводят pytest-тест в failed после выполнения его обычных проверок.

```powershell
$env:RUN_LIVE_AGENT_SCENARIOS = "1"
$env:LIVE_AGENT_MODE = "multiagent"
$env:LLM_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "qwen3.5:9b"
$env:LIVE_AGENT_TRANSCRIPT_PATH = ".test_runs/live-agent.md"
pytest tests/test_live_agent_scenarios.py -q
```

Live-сценарии проверяют обычный диалог, SQLite-count, ссылку на историю,
scrollable-результаты, последовательную передачу между workers, точные S2T-пары, Neo4j-пути и переход
SQLite → Neo4j. Неверные или неполные факты, отсутствие требуемого источника и
инфраструктурные ошибки делают сценарий failed. Отклонения display/UI
записываются как presentation warnings, а превышения времени, LLM-вызовов,
tools и токенов — как efficiency warnings; сами по себе они сценарий не роняют.

Для последовательного сравнения режимов:

```bash
uv run python scripts/run_live_agent_benchmark.py \
  --provider ollama \
  --model qwen3.5:9b \
  --modes multiagent single_agent
```

Отчёты записываются в `.test_runs/` и не попадают в git.

## Структура проекта

```text
app.py                         Flask API и выбор режима чата
processing/excel.py            механический разбор Excel
storage/database.py            схема и хранение исходных данных
storage/s2t.py                 операции с S2T transformations
sheet_skills/                  обработчики групп Excel-листов
services/analysis.py           post-upload pipeline
services/graph_sync.py         проекция SQLite → Neo4j
graph_storage/                 lifecycle и настройки Neo4j
agents/supervisor.py           верхний LangGraph
agents/coordinator.py          выбор pipeline, downstream/workers/upstream
agents/worker.py               worker runtime и работа с зависимостями
agents/chat_graph.py           planner/tool/observer loop
agents/validation_protocol.py  typed S2T readers и анализ
agents/test_protocol.py        compiler SQL тест-протоколов
agents/tools/routing.py        LLM router tools и skills
agents/tools/saved_results.py  run-scoped результаты и read-only relation
agents/tools/                  read-only/write registries и tools
agents/prompts/                runtime prompts и skills
agents/run_metrics.py          метрики live-запусков
config/                        JSON-конфигурации извлечения
templates/chat_app.html        единый интерфейс
scripts/                       benchmark-скрипты
tests/                         unit, integration и live tests
samples/                       примеры S2T Excel
```

## Логи и трассировка

Логи пишутся в консоль и в ротационный UTF-8 файл `logs/agent.log`. Уровень и размер задаются через `LOG_LEVEL`, `LOG_FILE`, `LOG_MAX_BYTES` и `LOG_BACKUP_COUNT`.

Langfuse необязателен. Для включения задайте `LANGFUSE_ENABLED=true`, `LANGFUSE_PUBLIC_KEY` и `LANGFUSE_SECRET_KEY`.

## Безопасность данных

- чат read-only по умолчанию;
- mutation-tools находятся в отдельном registry;
- очистка и повторная запись требуют явного действия пользователя;
- свободный SQL и Cypher ограничены read-only операциями;
- полные tool results не размножаются в LLM-контексте;
- SQLite остаётся источником истины даже при включённом Neo4j.
