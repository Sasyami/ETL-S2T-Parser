import json
from contextlib import nullcontext
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agents.contracts import (
    EvidenceArtifact,
    EvidenceFact,
    PreviousResultReference,
    WORKER_PREVIOUS_RESULTS_MARKER,
    WORKER_STABLE_CONTEXT_MARKER,
    WorkerOutcome,
    WorkerPlan,
    parse_worker_request,
)
from agents.coordinator import CoordinatorAnswer
from agents.tools.saved_results import SavedResultColumn, SavedResultDescriptor


def _tool_message(name, args, call_id):
    clean_args = dict(args)
    if name == "submit_upstream_output":
        action = str(clean_args.pop("action", "answer") or "answer")
        if action == "request_more_data":
            name = "submit_upstream_data_decision"
            clean_args = {
                "decision": "reroute",
                "problem": clean_args.get("problem", ""),
            }
        else:
            name = "submit_upstream_answer"
            clean_args.pop("problem", None)
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": clean_args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _artifact(
    display_ref,
    tool_name,
    preview,
    *,
    evidence_id=None,
    compact_args=None,
    truncated=False,
    dataset_ref=None,
):
    return EvidenceArtifact(
        evidence_id=evidence_id or f"evidence-{display_ref}",
        tool_name=tool_name,
        compact_args=dict(compact_args or {}),
        preview=preview,
        truncated=truncated,
        display_ref=display_ref,
        dataset_ref=dataset_ref,
    )


def _outcome(
    summary,
    *,
    evidence=(),
    datasets=(),
    previous_results=(),
    facts=None,
):
    evidence_items = list(evidence)
    fact_items = facts
    if fact_items is None and evidence_items:
        fact_items = [
            EvidenceFact(
                text=summary,
                evidence_ids=[item.evidence_id for item in evidence_items],
            )
        ]
    return WorkerOutcome(
        summary=summary,
        facts=list(fact_items or []),
        evidence=evidence_items,
        datasets=list(datasets),
        previous_results=list(previous_results),
    )


class _BoundModel:
    def __init__(self, parent, tool_name):
        self.parent = parent
        self.tool_name = tool_name

    def invoke(self, messages, **kwargs):
        del kwargs
        self.parent.messages.append((self.tool_name, list(messages)))
        return self.parent.responses[self.tool_name].pop(0)


class _CoordinatorModel:
    def __init__(self, responses):
        self.responses = {name: list(items) for name, items in responses.items()}
        self.responses.setdefault(
            "select_operation_skills",
            [
                _tool_message(
                    "select_operation_skills",
                    {"skills": []},
                    "operation-skills-1",
                )
            ],
        )
        legacy_upstream = self.responses.pop("submit_upstream_output", [])
        if legacy_upstream:
            decisions = self.responses.setdefault(
                "submit_upstream_data_decision", []
            )
            answers = self.responses.setdefault("submit_upstream_answer", [])
            for item in legacy_upstream:
                calls = list(item.tool_calls) if isinstance(item, AIMessage) else []
                if len(calls) == 1 and calls[0].get("name") == (
                    "submit_upstream_data_decision"
                ):
                    decisions.append(item)
                    continue
                if len(calls) == 1 and calls[0].get("name") == (
                    "submit_upstream_answer"
                ):
                    decisions.append(
                        _tool_message(
                            "submit_upstream_data_decision",
                            {"decision": "pass"},
                            str(calls[0].get("id") or "upstream")
                            + "-decision",
                        )
                    )
                    answers.append(item)
                    continue
                decisions.append(item)
        self.messages = []
        self.tool_choices = []

    def bind_tools(self, tools, tool_choice=None):
        tool_names = [tool["function"]["name"] for tool in tools]
        recorded_choice = (
            tuple(tool_names) if len(tool_names) > 1 else tool_names[0]
        )
        tool_name = tool_names[0]
        self.tool_choices.append((recorded_choice, tool_choice))
        return _BoundModel(self, tool_name)


def _payload(model, tool_name, occurrence=0):
    matching = [messages for name, messages in model.messages if name == tool_name]
    return json.loads(matching[occurrence][1].content)


def _patches(model):
    return (
        patch("agents.coordinator.chat_model", model),
        patch("agents.coordinator.get_callback_handler", return_value=None),
        patch(
            "agents.coordinator.langfuse_trace_context",
            return_value=nullcontext(),
        ),
    )


def _responses(
    *,
    answer,
    used_evidence_ids=(),
    display_evidence_ids=(),
    plan_task="Получи факт.",
):
    return {
        "submit_worker_plan": [
            _tool_message(
                "submit_worker_plan",
                {
                    "steps": [
                        {
                            "task": plan_task,
                        }
                    ]
                },
                "plan-1",
            )
        ],
        "submit_upstream_output": [
            _tool_message(
                "submit_upstream_output",
                {
                    "answer": answer,
                    "used_evidence_ids": list(used_evidence_ids),
                    "display_evidence_ids": list(display_evidence_ids),
                },
                "upstream-1",
            )
        ],
    }


def test_coordinator_graph_routes_tasks_downstream_and_one_result_upstream():
    from agents.coordinator import build_coordinator_graph

    model = _CoordinatorModel({})
    graph = build_coordinator_graph(model)
    graph_view = graph.get_graph()

    assert model.tool_choices == [
        ("select_operation_skills", "select_operation_skills"),
        ("submit_worker_plan", "submit_worker_plan"),
        (
            "submit_upstream_data_decision",
            "submit_upstream_data_decision",
        ),
        ("submit_upstream_answer", "submit_upstream_answer"),
    ]
    assert {
        "downstream_plan",
        "worker",
        "upstream",
    }.issubset(graph_view.nodes)
    assert "upstream_answer" not in graph_view.nodes
    edges = {(edge.source, edge.target) for edge in graph_view.edges}
    assert ("__start__", "downstream_plan") in edges
    assert ("downstream_plan", "worker") in edges
    assert ("upstream", "downstream_plan") in edges
    assert ("upstream", "__end__") in edges


def test_operation_skill_is_selected_once_and_applied_by_stage():
    from agents.coordinator import coordinator_chat

    responses = _responses(answer="Риск оценён.", plan_task="Прочитай правило.")
    responses["select_operation_skills"] = [
        _tool_message(
            "select_operation_skills",
            {"skills": ["Анализ SQL-рисков"]},
            "operation-skills-risk",
        )
    ]
    model = _CoordinatorModel(responses)
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch(
            "agents.coordinator.worker_chat",
            return_value=_outcome("Правило прочитано."),
        ) as worker,
    ):
        result = coordinator_chat("Оцени риск потери строк в A → B.")

    assert result.answer == "Риск оценён."
    assert [name for name, _ in model.messages].count(
        "select_operation_skills"
    ) == 1
    worker_parts = parse_worker_request(worker.call_args.args[0])
    normalized_execution = " ".join(
        worker_parts.operation_execution_context.split()
    )
    normalized_completeness = " ".join(
        worker_parts.operation_completeness_context.split()
    )
    assert (
        "используй единственный directed mapping-reader только с обязательными "
        "`source_table` и `target_table`"
    ) in normalized_execution
    assert "Вызов по ID нерелевантен" in (
        worker_parts.operation_completeness_context
    )
    assert "Вызов по ID нерелевантен" not in (
        worker_parts.operation_execution_context
    )
    assert "используй единственный directed mapping-reader" not in (
        normalized_completeness
    )

    plan_system = next(
        messages[0].content
        for name, messages in model.messages
        if name == "submit_worker_plan"
    )
    decision_system = next(
        messages[0].content
        for name, messages in model.messages
        if name == "submit_upstream_data_decision"
    )
    answer_system = next(
        messages[0].content
        for name, messages in model.messages
        if name == "submit_upstream_answer"
    )
    normalized_decision = " ".join(decision_system.split())
    assert "Не создавай transformation ID" in plan_system
    assert "Нулевой mapping подтверждает отсутствие" in decision_system
    assert "не требуй больше evidence" in normalized_decision
    assert "условный вывод" in normalized_decision
    assert "не являются обязательным условием `pass`" in normalized_decision
    assert "Вызов по ID нерелевантен" not in decision_system
    assert "Различай итоговые" in answer_system
    assert "Нулевой mapping подтверждает отсутствие" not in answer_system


def test_operation_skill_loader_returns_only_requested_stage():
    from agents.tools.context import load_operation_skills

    contexts = {
        stage: load_operation_skills(
            ["Совместимость колонок"],
            stage=stage,
        )
        for stage in (
            "plan",
            "planner",
            "observer",
            "upstream_decision",
            "upstream",
        )
    }

    assert "Создай одну задачу" in contexts["plan"]
    assert "exact pair-reader" in contexts["planner"]
    assert "Pair-result" in contexts["observer"]
    assert "`pass` допустим" in contexts["upstream_decision"]
    assert "Верни фактические" in contexts["upstream"]
    assert len(set(contexts.values())) == 5
    assert all("Покрытие маппинга" not in text for text in contexts.values())
    assert load_operation_skills([], stage="planner") == ""


def test_sql_risk_operation_skill_allows_pass_for_conditional_answer():
    from agents.tools.context import load_operation_skills

    decision = load_operation_skills(
        ["Анализ SQL-рисков"],
        stage="upstream_decision",
    )
    normalized = " ".join(decision.split())

    assert "достаточность только относительно `original_task`" in normalized
    assert "условный вывод" in normalized
    assert "не являются обязательным условием `pass`" in normalized
    assert "Не делай `reroute` ради оценки вероятности" in normalized
    assert "`pass` требует exact mapping" not in normalized


def test_sql_risk_operation_skill_separates_risk_layers_by_stage():
    from agents.tools.context import load_operation_skills

    contexts = {
        stage: " ".join(
            load_operation_skills(
                ["Анализ SQL-рисков"],
                stage=stage,
            ).split()
        )
        for stage in (
            "plan",
            "planner",
            "observer",
            "upstream_decision",
            "upstream",
        )
    }

    assert "полный mapping достаточен" in contexts["plan"]
    assert "Не добавляй metadata только ради проверки ключей" in contexts["plan"]
    assert "Не сокращай metadata-задачу до одной стороны" in contexts["plan"]
    assert "Planner получает факты" in contexts["planner"]
    assert "не оценивай безопасность" in contexts["observer"]
    assert "target-only evidence подтверждает ограничение" in (
        contexts["upstream_decision"]
    )
    answer = contexts["upstream"]
    assert "отсечение входных строк самим SQL" in answer
    assert "размножение или схлопывание строк" in answer
    assert "rejection результата ограничениями загрузки" in answer
    assert "write semantics" in answer
    assert "`Не оценено` никогда не является фактором снижения риска" in answer
    assert "Прямой field mapping не доказывает кардинальность 1:1" in answer
    assert "его нельзя назвать минимальным" in answer
    assert "отсечение входных строк самим SQL" not in contexts["observer"]


def test_plan_requires_only_self_contained_task():
    from agents.coordinator import WorkerPlan

    plan = WorkerPlan.model_validate(
        {
            "steps": [
                {
                    "task": "Первый шаг.",
                },
                {
                    "task": "Второй шаг.",
                },
            ]
        }
    )
    assert [step.task for step in plan.steps] == [
        "Первый шаг.",
        "Второй шаг.",
    ]

    with pytest.raises(ValueError, match="needs_from_previous"):
        WorkerPlan.model_validate(
            {
                "steps": [
                    {
                        "task": "Первый шаг.",
                        "needs_from_previous": "Несуществующий прошлый факт",
                    }
                ]
            }
        )

    with pytest.raises(ValueError, match="depends_on"):
        WorkerPlan.model_validate(
            {
                "steps": [
                    {
                        "task": "Первый шаг.",
                        "depends_on": [],
                    }
                ]
            }
        )

    with pytest.raises(ValueError, match="required_evidence"):
        WorkerPlan.model_validate(
            {
                "steps": [
                    {
                        "task": "Первый шаг.",
                        "required_evidence": ["Лишний критерий"],
                    }
                ]
            }
        )

    with pytest.raises(ValueError, match="task"):
        WorkerPlan.model_validate({"steps": [{}]})


def test_contracts_keep_runtime_refs_out_of_llm_payloads():
    dataset = SavedResultDescriptor(
        result_ref="saved-first",
        source_tool="lookup",
        row_count=1,
        columns=[SavedResultColumn(name="name", sqlite_type="TEXT")],
    )
    artifact = _artifact(
        "display-first",
        "lookup",
        '{"name":"t_example"}',
        evidence_id="evidence-first",
        compact_args={"query": "t_example"},
        dataset_ref=dataset.result_ref,
    )
    outcome = _outcome(
        "Точное имя найдено.",
        evidence=[artifact],
        datasets=[dataset],
        previous_results=[
            PreviousResultReference(
                result_id="result-first",
                description="lookup: точное имя t_example.",
            )
        ],
    )

    assert "display_ref" not in artifact.model_dump()
    assert "dataset_ref" not in artifact.model_dump()
    assert "datasets" not in outcome.model_dump()
    handoff = outcome.handoff_payload()
    assert handoff == {
        "previous_results": [
            {
                "result_id": "result-first",
                "description": "lookup: точное имя t_example.",
            }
        ]
    }
    assert "display_ref" not in json.dumps(handoff)
    assert "dataset_ref" not in json.dumps(handoff)
    assert "preview" not in json.dumps(handoff)
    upstream = outcome.upstream_payload()
    assert "gap" not in handoff
    assert "gap" not in upstream
    assert "datasets" not in upstream
    assert "dataset_ref" not in json.dumps(upstream)
    assert "display_ref" not in json.dumps(upstream)
    assert upstream == {
        "evidence": [
            {
                "evidence_id": "evidence-first",
                "tool_name": "lookup",
                "args": {"query": "t_example"},
                "preview": '{"name":"t_example"}',
                "truncated": False,
                "display_id": "evidence-first",
            }
        ]
    }

    with pytest.raises(ValueError, match="unknown evidence_id"):
        _outcome(
            "Некорректный provenance.",
            facts=[EvidenceFact(text="Факт", evidence_ids=["unknown"])],
        )


def test_upstream_data_decision_is_separate_from_answer_payload():
    from agents.contracts import UpstreamDecision

    decision = UpstreamDecision.model_validate(
        {
            "decision": "reroute",
            "problem": "Не получена вторая запрошенная метрика.",
        }
    )

    assert decision.decision == "reroute"
    assert decision.problem == "Не получена вторая запрошенная метрика."

    with pytest.raises(ValueError, match="answer"):
        UpstreamDecision.model_validate(
            {
                "decision": "pass",
                "answer": "Ответ относится к следующему этапу.",
            }
        )


def test_coordinator_prompts_and_schemas_match_contracts():
    from agents.contracts import WorkerPlan
    from agents.coordinator import (
        _DOWNSTREAM_PLAN_PROMPT,
        _DOWNSTREAM_PLAN_REPAIR_PROMPT,
        _DOWNSTREAM_CAPABILITY_CONTEXT,
        _DOWNSTREAM_TABLE_CONTEXT,
        _UPSTREAM_ANALYSIS_CONTEXT,
        _UPSTREAM_ANSWER_PROMPT,
        _UPSTREAM_DATA_DECISION_PROMPT,
        _plan_tool_schema,
        _upstream_answer_tool_schema,
        _upstream_data_decision_tool_schema,
    )

    combined = "\n".join(
        (_UPSTREAM_DATA_DECISION_PROMPT, _UPSTREAM_ANSWER_PROMPT)
    )
    for domain_detail in (
        "target_table",
        "source_table",
        "GROUP BY",
        "COUNT(DISTINCT",
        "Neo4j",
        "SQLite",
    ):
        assert domain_detail not in combined

    assert len(_DOWNSTREAM_PLAN_PROMPT) < 5000
    assert (
        len(_DOWNSTREAM_PLAN_PROMPT)
        - len(_DOWNSTREAM_TABLE_CONTEXT)
        - len(_DOWNSTREAM_CAPABILITY_CONTEXT)
        < 3200
    )
    assert _DOWNSTREAM_CAPABILITY_CONTEXT in _DOWNSTREAM_PLAN_PROMPT
    assert _DOWNSTREAM_TABLE_CONTEXT in _DOWNSTREAM_PLAN_PROMPT
    assert len(_UPSTREAM_DATA_DECISION_PROMPT) < 1300
    assert len(_UPSTREAM_ANSWER_PROMPT) < 2300
    assert len(_UPSTREAM_ANALYSIS_CONTEXT) < 3500
    assert (
        "row_format=named_records_with_dictionary_refs"
        in _UPSTREAM_ANALYSIS_CONTEXT
    )
    assert "0-based индексом" in _UPSTREAM_ANALYSIS_CONTEXT
    assert "не схлопывай одинаковые occurrences" in (
        _UPSTREAM_ANALYSIS_CONTEXT
    )
    assert "`depends_on`" not in _DOWNSTREAM_PLAN_PROMPT
    assert "needs_from_previous" not in _DOWNSTREAM_PLAN_PROMPT
    assert "required_evidence" not in _DOWNSTREAM_PLAN_PROMPT
    assert "не выбирай tools/skills" in (
        _DOWNSTREAM_PLAN_PROMPT.lower().replace("\n", " ")
    )
    assert "прямо необходимых фактов" in _DOWNSTREAM_PLAN_PROMPT
    assert "Каждый step обязан быть незаменимым" in _DOWNSTREAM_PLAN_PROMPT
    assert "Наличие\nтаблицы в справочнике не требует её чтения" in (
        _DOWNSTREAM_PLAN_PROMPT
    )
    assert "использует результат предыдущего" in (
        _DOWNSTREAM_PLAN_PROMPT
    )
    assert "Минимизируй обмен" in _DOWNSTREAM_PLAN_PROMPT
    assert "только краткие lazy-ссылки" in _DOWNSTREAM_PLAN_PROMPT
    assert "отдельный worker может сначала получить" in (
        _DOWNSTREAM_PLAN_PROMPT
    )
    assert "следующий — найти эти кандидаты в S2T" in (
        _DOWNSTREAM_PLAN_PROMPT
    )
    assert "S2T-поиск по подстроке лексический, не семантический" in (
        _DOWNSTREAM_PLAN_PROMPT
    )
    assert "делает upstream" in _DOWNSTREAM_PLAN_PROMPT
    assert "Сравнение, оценку, объяснение, вывод" in _DOWNSTREAM_PLAN_PROMPT
    assert "роль source/target известна,\nтолько если" in (
        _DOWNSTREAM_PLAN_PROMPT.lower()
    )
    assert "роль результата не задаёт роль кандидата" in (
        _DOWNSTREAM_PLAN_PROMPT.lower()
    )
    assert "Не превращай бизнес-термин в техническое имя" in (
        _DOWNSTREAM_PLAN_PROMPT
    )
    assert "не пиши task как вызов функции" in _DOWNSTREAM_PLAN_PROMPT
    assert "при неизвестной роли — сразу в обоих" in (
        _DOWNSTREAM_PLAN_PROMPT.lower()
    )
    assert "семантический кандидат не\nимеет s2t-роли" in (
        _DOWNSTREAM_PLAN_PROMPT.lower()
    )
    assert "Сохраняй тип поиска из original_task" in _DOWNSTREAM_PLAN_PROMPT
    assert "смысловой поиск цельной естественной фразой" in (
        _DOWNSTREAM_PLAN_PROMPT
    )
    assert "буквальный поиск, только если фрагмент явно дан" in (
        _DOWNSTREAM_PLAN_PROMPT
    )
    assert "Не превращай смысловой поиск в «найти содержащие»" in (
        _DOWNSTREAM_PLAN_PROMPT
    )
    assert "Планируй чтение только тех фактов" in _DOWNSTREAM_PLAN_PROMPT
    assert "просит описать способ будущего действия" in _DOWNSTREAM_PLAN_PROMPT
    assert "Для вывода по сохранённому выражению" in _DOWNSTREAM_PLAN_PROMPT
    assert "глобальную `s2t_transformations` не ограничивай `file_id`" in (
        _DOWNSTREAM_PLAN_PROMPT
    )
    assert "справка, не список шагов" in _DOWNSTREAM_TABLE_CONTEXT
    assert "наличие таблицы не требует её чтения" in _DOWNSTREAM_TABLE_CONTEXT
    assert "описывай нужные данные, не инструмент" in (
        _DOWNSTREAM_CAPABILITY_CONTEXT
    )
    assert "буквальный поиск" in _DOWNSTREAM_CAPABILITY_CONTEXT
    assert "явно данному фрагменту" in _DOWNSTREAM_CAPABILITY_CONTEXT
    assert "смысловой поиск по описаниям" in _DOWNSTREAM_CAPABILITY_CONTEXT
    assert "S2T-строки" in _DOWNSTREAM_CAPABILITY_CONTEXT
    assert "сохранённые результаты прошлых workers" in (
        _DOWNSTREAM_CAPABILITY_CONTEXT
    )
    for tool_name in (
        "list_column_catalog",
        "semantic_search_descriptions",
        "list_s2t_transformations",
        "run_sql",
        "read_previous_result",
    ):
        assert tool_name not in _DOWNSTREAM_CAPABILITY_CONTEXT
    for table_name in (
        "files",
        "file_sheet_headers",
        "source_tables",
        "target_tables",
        "source_columns",
        "target_columns",
        "additional_objects",
        "pxf_to_a",
        "s2t_transformations",
        "data",
    ):
        assert f"`{table_name}`" in _DOWNSTREAM_TABLE_CONTEXT
    assert "бизнес-описания таблиц" in _DOWNSTREAM_TABLE_CONTEXT
    assert "текст правила" in _DOWNSTREAM_TABLE_CONTEXT
    assert "неизвестные бизнес-объекты оставляй текстом поиска" in (
        _DOWNSTREAM_PLAN_REPAIR_PROMPT
    )
    assert "скопируй `original_task`" not in _DOWNSTREAM_PLAN_PROMPT
    plan_schema_text = str(_plan_tool_schema())
    assert "По умолчанию один шаг" not in plan_schema_text
    assert "без отдельных шагов производного анализа" in plan_schema_text
    assert "input_steps" not in plan_schema_text
    assert _plan_tool_schema()["function"]["parameters"]["properties"][
        "steps"
    ]["items"]["required"] == ["task"]
    worker_plan_schema_text = str(WorkerPlan.model_json_schema())
    assert "По умолчанию один шаг" not in worker_plan_schema_text
    assert "лениво использовать принятые результаты" in worker_plan_schema_text
    assert "результаты между шагами не передаются" not in worker_plan_schema_text
    assert "decision=\"pass\"" in _UPSTREAM_DATA_DECISION_PROMPT
    assert "decision=\"reroute\"" in _UPSTREAM_DATA_DECISION_PROMPT
    assert "не предлагай имена таблиц, колонок" in (
        _UPSTREAM_DATA_DECISION_PROMPT
    )
    assert "Не формируй пользовательский ответ" in (
        _UPSTREAM_DATA_DECISION_PROMPT
    )
    assert "used_evidence_ids" in _UPSTREAM_ANSWER_PROMPT
    assert "display_evidence_ids" in _UPSTREAM_ANSWER_PROMPT
    assert "`display_id`" in _UPSTREAM_ANSWER_PROMPT
    assert "summary" not in combined.lower()
    assert "facts" not in combined.lower()
    assert "limitations" not in combined.lower()
    assert "каждый запрошенный" in combined
    assert "значение одной метрики" in combined
    assert "Промежуточный список кандидатов" in (
        _UPSTREAM_DATA_DECISION_PROMPT
    )
    assert "подпиши смысл каждого значения" in _UPSTREAM_ANSWER_PROMPT
    assert "безымянную CSV-последовательность" in _UPSTREAM_ANSWER_PROMPT
    assert "observations" not in combined
    assert "Правила upstream-анализа" in _UPSTREAM_ANALYSIS_CONTEXT
    assert "`LEFT JOIN ... ON p`" in _UPSTREAM_ANALYSIS_CONTEXT
    assert "Полевой маппинг задаёт конкретная S2T-строка" in (
        _UPSTREAM_ANALYSIS_CONTEXT
    )
    assert "`ON TRUE AND p` эквивалентно `ON p`" in " ".join(
        _UPSTREAM_ANALYSIS_CONTEXT.split()
    )
    assert "Технические поля хранилища" in _UPSTREAM_ANALYSIS_CONTEXT
    assert "не переименовывай" in _UPSTREAM_ANALYSIS_CONTEXT
    assert "Не подставляй фиктивное значение" in _UPSTREAM_ANALYSIS_CONTEXT
    assert "`{PLACEHOLDER}`" not in _UPSTREAM_ANALYSIS_CONTEXT
    assert "`analyze`" not in _UPSTREAM_ANALYSIS_CONTEXT

    answer_schema = _upstream_answer_tool_schema()["function"]["parameters"]
    assert answer_schema["required"] == ["answer"]
    assert set(answer_schema["properties"]) == {
        "answer",
        "used_evidence_ids",
        "display_evidence_ids",
    }
    request_schema = _upstream_data_decision_tool_schema()["function"][
        "parameters"
    ]
    assert request_schema["required"] == ["decision"]
    assert request_schema["properties"]["decision"]["enum"] == [
        "pass",
        "reroute",
    ]
    assert set(request_schema["properties"]) == {"decision", "problem"}
    plan_schema = _plan_tool_schema()["function"]["parameters"]
    step_schema = plan_schema["properties"]["steps"]["items"]
    assert step_schema["required"] == ["task"]
    assert "input_steps" not in step_schema["properties"]
    assert "depends_on" not in step_schema["properties"]
    assert "needs_from_previous" not in step_schema["properties"]
    assert "required_evidence" not in step_schema["properties"]


def test_upstream_native_tools_enforce_linear_payloads():
    from agents.coordinator import (
        CoordinatorResponseError,
        _native_upstream_answer,
        _native_upstream_decision,
    )

    pass_decision = _native_upstream_decision(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_upstream_data_decision",
                    "args": {"decision": "pass"},
                    "id": "decision-pass",
                    "type": "tool_call",
                }
            ],
        )
    )
    assert pass_decision.decision == "pass"

    valid_request = _native_upstream_decision(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_upstream_data_decision",
                    "args": {
                        "decision": "reroute",
                        "problem": "Не найден исходный путь.",
                    },
                    "id": "decision-reroute",
                    "type": "tool_call",
                }
            ],
        )
    )
    assert valid_request.decision == "reroute"
    assert valid_request.problem == "Не найден исходный путь."

    request_without_problem = _native_upstream_decision(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_upstream_data_decision",
                    "args": {"decision": "reroute"},
                    "id": "request-without-problem",
                    "type": "tool_call",
                }
            ],
        )
    )
    assert request_without_problem.decision == "reroute"
    assert request_without_problem.problem == ""

    with pytest.raises(CoordinatorResponseError, match="decision: Field required"):
        _native_upstream_decision(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_upstream_data_decision",
                        "args": {},
                        "id": "request-without-decision",
                        "type": "tool_call",
                    }
                ],
            )
        )

    valid_answer = _native_upstream_answer(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_upstream_answer",
                    "args": {
                        "answer": "Путь найден.",
                        "used_evidence_ids": ["evidence-path"],
                        "display_evidence_ids": ["evidence-path"],
                    },
                    "id": "answer-1",
                    "type": "tool_call",
                }
            ],
        )
    )
    assert valid_answer.answer == "Путь найден."

    minimal_answer = _native_upstream_answer(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_upstream_answer",
                    "args": {"answer": "Данных достаточно."},
                    "id": "answer-minimal",
                    "type": "tool_call",
                }
            ],
        )
    )
    assert minimal_answer.used_evidence_ids == []
    assert minimal_answer.display_evidence_ids == []

    with pytest.raises(CoordinatorResponseError, match="answer: Field required"):
        _native_upstream_answer(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_upstream_answer",
                        "args": {
                            "used_evidence_ids": ["evidence-path"],
                            "display_evidence_ids": ["evidence-path"],
                        },
                        "id": "answer-without-text",
                        "type": "tool_call",
                    }
                ],
            )
        )


def test_coordinator_keeps_workers_isolated_and_combines_upstream_output(caplog):
    from agents.coordinator import _UPSTREAM_ANALYSIS_CONTEXT, coordinator_chat

    dataset = SavedResultDescriptor(
        result_ref="saved-first",
        source_tool="lookup",
        row_count=1,
        columns=[SavedResultColumn(name="name", sqlite_type="TEXT")],
    )
    first_artifact = _artifact(
        "display-first",
        "lookup",
        '{"name":"t_example"}',
        evidence_id="evidence-first",
        compact_args={"query": "t_example"},
        dataset_ref="saved-first",
    )
    second_artifact = _artifact(
        "display-second",
        "inspect",
        '{"name":"t_example","valid":true}',
        evidence_id="evidence-second",
        compact_args={"name": "t_example"},
    )
    model = _CoordinatorModel(
        {
            "submit_worker_plan": [
                _tool_message(
                    "submit_worker_plan",
                    {
                        "steps": [
                            {
                                "task": "Найди точное имя.",
                            },
                            {
                                "task": "Проверь найденное имя.",
                            },
                        ]
                    },
                    "plan-1",
                )
            ],
            "submit_upstream_output": [
                _tool_message(
                    "submit_upstream_output",
                    {
                        "answer": "Имя t_example проверено.",
                        "used_evidence_ids": [
                            "evidence-first",
                            "evidence-second",
                        ],
                        "display_evidence_ids": [
                            "evidence-first",
                            "evidence-second",
                        ],
                    },
                    "upstream-1",
                )
            ],
        }
    )
    worker_results = [
        _outcome(
            "Точное имя: t_example.",
            evidence=[first_artifact],
            datasets=[dataset],
            previous_results=[
                PreviousResultReference(
                    result_id="result-first",
                    description="lookup: точное имя t_example.",
                )
            ],
        ),
        _outcome("Имя t_example проверено.", evidence=[second_artifact]),
    ]
    model_patch, callback_patch, trace_patch = _patches(model)
    with caplog.at_level("INFO", logger="agents.coordinator"):
        with (
            model_patch,
            callback_patch,
            trace_patch,
            patch(
                "agents.coordinator.worker_chat",
                side_effect=worker_results,
            ) as worker,
            patch("agents.coordinator.discard_worker_display_refs") as discard,
        ):
            result = coordinator_chat(
                "Найди имя и проверь его.",
                context="Общий фон",
            )

    assert result == CoordinatorAnswer(
        answer="Имя t_example проверено.",
        display_refs=["display-first", "display-second"],
    )
    assert worker.call_count == 2
    assert worker.call_args_list[0].kwargs == {}
    assert worker.call_args_list[1].kwargs == {}
    context_suffix = WORKER_STABLE_CONTEXT_MARKER + "Общий фон"
    assert worker.call_args_list[0].args[0] == "Найди точное имя." + context_suffix
    second_task = worker.call_args_list[1].args[0]
    second_parts = parse_worker_request(second_task)
    assert second_parts.current_task == "Проверь найденное имя."
    assert second_parts.stable_context == "Общий фон"
    assert [
        item.model_dump(mode="json", exclude_none=True)
        for item in (second_parts.previous_results or [])
    ] == [
        {
            "result_id": "result-first",
            "description": "lookup: точное имя t_example.",
        }
    ]
    assert WORKER_PREVIOUS_RESULTS_MARKER in second_task
    discard.assert_not_called()

    upstream = _payload(model, "submit_upstream_answer")
    serialized_upstream = json.dumps(upstream, ensure_ascii=False)
    assert set(upstream) == {"original_task", "evidence"}
    assert upstream["original_task"] == "Найди имя и проверь его."
    assert upstream["evidence"] == [
        {
            "evidence_id": "evidence-first",
            "tool_name": "lookup",
            "args": {"query": "t_example"},
            "preview": '{"name":"t_example"}',
            "truncated": False,
            "display_id": "evidence-first",
        },
        {
            "evidence_id": "evidence-second",
            "tool_name": "inspect",
            "args": {"name": "t_example"},
            "preview": '{"name":"t_example","valid":true}',
            "truncated": False,
            "display_id": "evidence-second",
        },
    ]
    for forbidden in (
        "summary",
        "facts",
        "limitations",
        "status",
        "observation",
        "cycle_history",
        "display_ref",
        "dataset_ref",
        "datasets",
    ):
        assert forbidden not in serialized_upstream
    upstream_messages = [
        messages
        for name, messages in model.messages
        if name == "submit_upstream_answer"
    ][0]
    assert _UPSTREAM_ANALYSIS_CONTEXT in str(upstream_messages[0].content)
    assert "Upstream coordinator result:" in caplog.text


def test_upstream_selects_only_requested_display_and_discards_other_refs():
    from agents.coordinator import coordinator_chat

    stages = []

    def stage_scope(stage):
        stages.append(stage)
        return nullcontext()

    model = _CoordinatorModel(
        _responses(
            answer="Факт получен.",
            used_evidence_ids=("evidence-second",),
            display_evidence_ids=("evidence-second",),
        )
    )
    worker_result = _outcome(
        "Факт получен.",
        evidence=[
            _artifact(
                "display-first",
                "lookup",
                "first",
                evidence_id="evidence-first",
            ),
            _artifact(
                "display-second",
                "inspect",
                "second",
                evidence_id="evidence-second",
            ),
        ],
    )
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch("agents.coordinator.llm_stage", side_effect=stage_scope),
        patch("agents.coordinator.worker_chat", return_value=worker_result),
        patch("agents.coordinator.discard_worker_display_refs") as discard,
    ):
        result = coordinator_chat("Получи факт.")

    assert result == CoordinatorAnswer(
        answer="Факт получен.",
        display_refs=["display-second"],
    )
    discard.assert_called_once_with(["display-first"])
    assert stages == [
        "operation_router",
        "downstream_plan",
        "upstream",
        "upstream",
    ]


def test_coordinator_passes_lazy_result_references_between_workers():
    from agents.coordinator import coordinator_chat

    model = _CoordinatorModel(
        {
            "submit_worker_plan": [
                _tool_message(
                    "submit_worker_plan",
                    {
                        "steps": [
                            {
                                "task": "Получи первый факт.",
                            },
                            {
                                "task": "Получи второй факт.",
                            },
                            {
                                "task": "Проверь второй факт.",
                            },
                        ]
                    },
                    "plan-1",
                )
            ],
            "submit_upstream_output": [
                _tool_message(
                    "submit_upstream_output",
                    {
                        "answer": "Проверка завершена.",
                        "used_evidence_ids": [],
                        "display_evidence_ids": [],
                    },
                    "upstream-1",
                )
            ],
        }
    )
    worker_results = [
        _outcome(
            "Первый факт: A.",
            evidence=[
                _artifact(
                    "display-first",
                    "first_lookup",
                    '{"value":"A"}',
                    evidence_id="evidence-first",
                )
            ],
            previous_results=[
                PreviousResultReference(
                    result_id="result-first",
                    description="first_lookup: первый факт A.",
                )
            ],
        ),
        _outcome(
            "Второй факт: B.",
            evidence=[
                _artifact(
                    "display-second",
                    "second_lookup",
                    '{"value":"B"}',
                    evidence_id="evidence-second",
                )
            ],
            previous_results=[
                PreviousResultReference(
                    result_id="result-second",
                    description="second_lookup: второй факт B.",
                )
            ],
        ),
        _outcome("Второй факт B проверен."),
    ]
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch("agents.coordinator.worker_chat", side_effect=worker_results) as worker,
        patch("agents.coordinator.discard_worker_display_refs"),
    ):
        result = coordinator_chat("Получи два факта и проверь второй.")

    assert result.answer == "Проверка завершена."
    first_parts, second_parts, third_parts = [
        parse_worker_request(call.args[0]) for call in worker.call_args_list
    ]
    assert first_parts.current_task == "Получи первый факт."
    assert first_parts.previous_results is None
    assert second_parts.current_task == "Получи второй факт."
    assert [
        item.result_id for item in (second_parts.previous_results or [])
    ] == ["result-first"]
    assert third_parts.current_task == "Проверь второй факт."
    assert [
        item.result_id for item in (third_parts.previous_results or [])
    ] == ["result-first", "result-second"]


def test_upstream_receives_partial_worker_evidence():
    from agents.coordinator import coordinator_chat

    answer = "Не удалось подтвердить требуемый факт."
    model = _CoordinatorModel(_responses(answer=answer))
    worker_result = _outcome(
        "Tool вернул данные не по той сущности; факт не подтверждён.",
        evidence=[
            _artifact(
                "display-partial",
                "lookup",
                '{"value":"partial"}',
                evidence_id="evidence-partial",
            )
        ],
    )
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch("agents.coordinator.worker_chat", return_value=worker_result),
        patch("agents.coordinator.discard_worker_display_refs") as discard,
    ):
        result = coordinator_chat("Получи факт.")

    assert result == CoordinatorAnswer(answer=answer, display_refs=[])
    upstream = _payload(model, "submit_upstream_answer")
    assert set(upstream) == {"original_task", "evidence"}
    assert upstream["evidence"] == [
        {
            "evidence_id": "evidence-partial",
            "tool_name": "lookup",
            "args": {},
            "preview": '{"value":"partial"}',
            "truncated": False,
            "display_id": "evidence-partial",
        }
    ]
    discard.assert_called_once_with(["display-partial"])


def test_upstream_normalizes_structured_answer_after_data_pass():
    from agents.coordinator import coordinator_chat

    model = _CoordinatorModel(
        _responses(
            answer=[{"value": 42}],
            plan_task="Верни JSON-массив со значением 42.",
        )
    )
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch(
            "agents.coordinator.worker_chat",
            return_value=_outcome("Значение: 42."),
        ),
    ):
        result = coordinator_chat("Верни JSON-массив со значением 42.")

    assert result == CoordinatorAnswer(answer='[{"value":42}]', display_refs=[])
    assert [name for name, _ in model.messages] == [
        "select_operation_skills",
        "submit_worker_plan",
        "submit_upstream_data_decision",
        "submit_upstream_answer",
    ]


def test_upstream_repairs_unknown_evidence_id():
    from agents.coordinator import coordinator_chat

    model = _CoordinatorModel(
        {
            "submit_worker_plan": _responses(answer="unused")[
                "submit_worker_plan"
            ],
            "submit_upstream_output": [
                _tool_message(
                    "submit_upstream_output",
                    {
                        "answer": "Факт получен.",
                        "used_evidence_ids": ["unknown"],
                        "display_evidence_ids": ["unknown"],
                    },
                    "upstream-invalid",
                ),
                _tool_message(
                    "submit_upstream_output",
                    {
                        "answer": "Факт получен.",
                        "used_evidence_ids": ["evidence-result"],
                        "display_evidence_ids": ["evidence-result"],
                    },
                    "upstream-repaired",
                ),
            ],
        }
    )
    worker_result = _outcome(
        "Факт получен.",
        evidence=[
            _artifact(
                "display-result",
                "lookup",
                '{"value":1}',
                evidence_id="evidence-result",
            )
        ],
    )
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch("agents.coordinator.worker_chat", return_value=worker_result),
    ):
        result = coordinator_chat("Получи факт.")

    assert result.display_refs == ["display-result"]
    upstream_calls = [
        messages
        for name, messages in model.messages
        if name == "submit_upstream_answer"
    ]
    assert len(upstream_calls) == 2
    assert isinstance(upstream_calls[1][-2], ToolMessage)
    assert upstream_calls[1][-2].tool_call_id == "upstream-invalid"
    assert "rejected" in upstream_calls[1][-2].content
    assert "доступные evidence_id" in upstream_calls[1][-1].content
    assert '"evidence-result"' in upstream_calls[1][-1].content


def test_upstream_has_no_separate_semantic_review():
    from agents.coordinator import coordinator_chat

    model = _CoordinatorModel(
        {
            "submit_worker_plan": _responses(
                answer="unused",
                plan_task=(
                    "Верни имя и число строк. Полный результат числа "
                    "покажи отдельно."
                ),
            )["submit_worker_plan"],
            "submit_upstream_output": [
                _tool_message(
                    "submit_upstream_output",
                    {
                        "answer": "Имя: t_example.",
                        "used_evidence_ids": ["evidence-name"],
                        "display_evidence_ids": [],
                    },
                    "upstream-incomplete",
                ),
            ],
        }
    )
    worker_result = _outcome(
        "Найдены имя и число строк.",
        evidence=[
            _artifact(
                "display-name",
                "lookup",
                '{"name":"t_example"}',
                evidence_id="evidence-name",
            ),
            _artifact(
                "display-count",
                "count",
                '{"row_count":42}',
                evidence_id="evidence-count",
            ),
        ],
    )
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch("agents.coordinator.worker_chat", return_value=worker_result),
    ):
        result = coordinator_chat(
            "Верни имя и число строк. Полный результат числа покажи отдельно."
        )

    assert result.answer == "Имя: t_example."
    assert result.display_refs == []
    upstream_calls = [
        messages
        for name, messages in model.messages
        if name == "submit_upstream_answer"
    ]
    assert len(upstream_calls) == 1
    assert "submit_upstream_review" not in [name for name, _ in model.messages]


def test_upstream_restarts_cleanly_with_only_problem():
    from agents.coordinator import coordinator_chat

    original_task = "Верни top target_table и все три метрики source_table."
    model = _CoordinatorModel(
        {
            "submit_worker_plan": [
                _tool_message(
                    "submit_worker_plan",
                    {
                        "steps": [
                            {
                                "task": original_task
                            }
                        ]
                    },
                    "plan-cycle-1",
                ),
                _tool_message(
                    "submit_worker_plan",
                    {
                        "steps": [
                            {
                                "task": original_task
                            }
                        ]
                    },
                    "plan-cycle-2",
                ),
            ],
            "submit_upstream_output": [
                _tool_message(
                    "submit_upstream_output",
                    {
                        "action": "request_more_data",
                        "problem": "Отсутствуют три метрики source_table.",
                    },
                    "upstream-request-more",
                ),
                _tool_message(
                    "submit_upstream_output",
                    {
                        "answer": (
                            "target_table=t_example; строк=42; разных "
                            "source_table=3; top_source=s_example; строк=20"
                        ),
                        "used_evidence_ids": [
                            "evidence-complete",
                        ],
                        "display_evidence_ids": ["evidence-complete"],
                    },
                    "upstream-final",
                ),
            ],
        }
    )
    first_outcome = _outcome(
        "target_table=t_example; строк=42",
        evidence=[
            _artifact(
                "display-target",
                "run_sql",
                '{"target_table":"t_example","row_count":42}',
                evidence_id="evidence-target",
            )
        ],
    )
    second_outcome = _outcome(
        (
            "target_table=t_example; строк=42; разных source_table=3; "
            "top_source=s_example; строк=20"
        ),
        evidence=[
            _artifact(
                "display-complete",
                "run_sql",
                (
                    '{"target_table":"t_example","target_rows":42,'
                    '"distinct_sources":3,"top_source":"s_example",'
                    '"source_rows":20}'
                ),
                evidence_id="evidence-complete",
            )
        ],
    )
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch(
            "agents.coordinator.worker_chat",
            side_effect=[first_outcome, second_outcome],
        ) as worker,
    ):
        result = coordinator_chat(original_task)

    assert result == CoordinatorAnswer(
        answer=(
            "target_table=t_example; строк=42; разных source_table=3; "
            "top_source=s_example; строк=20"
        ),
        display_refs=["display-complete"],
    )
    assert worker.call_count == 2
    assert len(
        [
            name
            for name, _ in model.messages
            if name == "submit_upstream_data_decision"
        ]
    ) == 2
    second_plan_payload = _payload(model, "submit_worker_plan", 1)
    assert second_plan_payload == {
        "original_task": original_task,
        "context": "",
        "problem": "Отсутствуют три метрики source_table.",
    }
    second_worker_task = worker.call_args_list[1].args[0]
    assert "target_table=t_example; строк=42" not in second_worker_task
    final_upstream_payload = _payload(model, "submit_upstream_answer")
    assert set(final_upstream_payload) == {"original_task", "evidence"}
    assert [
        item["evidence_id"] for item in final_upstream_payload["evidence"]
    ] == ["evidence-complete"]


def test_coordinator_returns_limited_answer_after_two_data_cycles():
    from agents.coordinator import CoordinatorAnswer, coordinator_chat

    model = _CoordinatorModel(
        {
            "submit_worker_plan": [
                _tool_message(
                    "submit_worker_plan",
                    {"steps": [{"task": "Получи полный результат."}]},
                    "plan-cycle-1",
                ),
                _tool_message(
                    "submit_worker_plan",
                    {"steps": [{"task": "Получи полный результат."}]},
                    "plan-cycle-2",
                ),
            ],
            "submit_upstream_output": [
                _tool_message(
                    "submit_upstream_output",
                    {
                        "action": "request_more_data",
                        "answer": "",
                        "used_evidence_ids": [],
                        "display_evidence_ids": [],
                        "problem": "Первой части недостаточно.",
                    },
                    "request-cycle-1",
                ),
                _tool_message(
                    "submit_upstream_output",
                    {
                        "answer": (
                            "Не удалось получить достаточно подтверждённых "
                            "данных для полного ответа. После второго цикла "
                            "данных всё ещё мало."
                        ),
                        "used_evidence_ids": [],
                        "display_evidence_ids": [],
                    },
                    "pass-cycle-2",
                ),
            ],
        }
    )
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch(
            "agents.coordinator.worker_chat",
            side_effect=[_outcome("Первая часть."), _outcome("Вторая часть.")],
        ),
    ):
        result = coordinator_chat("Получи полный результат.")

    assert result == CoordinatorAnswer(
        answer=(
            "Не удалось получить достаточно подтверждённых данных для "
            "полного ответа. После второго цикла данных всё ещё мало."
        ),
        display_refs=[],
    )


def test_coordinator_empty_task_does_not_call_llm():
    from agents.coordinator import coordinator_chat

    result = coordinator_chat("   ")

    assert result.display_refs == []
    assert "пуст" in result.answer.lower()


def test_coordinator_repairs_plan_that_exceeds_worker_limit():
    from agents.coordinator import coordinator_chat

    invalid_steps = [
        {
            "task": f"Проверка {index}",
        }
        for index in range(1, 10)
    ]
    model = _CoordinatorModel(
        {
            "submit_worker_plan": [
                _tool_message(
                    "submit_worker_plan",
                    {"steps": invalid_steps},
                    "plan-invalid",
                ),
                _tool_message(
                    "submit_worker_plan",
                    {
                        "steps": [
                            {
                                "task": "Выполни девять связанных проверок.",
                            }
                        ]
                    },
                    "plan-repaired",
                ),
            ],
            "submit_upstream_output": [
                _tool_message(
                    "submit_upstream_output",
                    {
                        "answer": "Все проверки выполнены.",
                        "used_evidence_ids": [],
                        "display_evidence_ids": [],
                    },
                    "upstream-1",
                )
            ],
        }
    )
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch(
            "agents.coordinator.worker_chat",
            return_value=_outcome("Все проверки выполнены."),
        ),
    ):
        result = coordinator_chat("Выполни девять связанных проверок.")

    assert result == CoordinatorAnswer(
        answer="Все проверки выполнены.",
        display_refs=[],
    )
    plan_messages = [
        messages
        for name, messages in model.messages
        if name == "submit_worker_plan"
    ]
    assert len(plan_messages) == 2
    assert isinstance(plan_messages[1][-2], ToolMessage)
    assert plan_messages[1][-2].tool_call_id == "plan-invalid"
    assert "rejected" in plan_messages[1][-2].content
    assert "от 1 до 8 элементов" in plan_messages[1][-1].content
    assert "непустую `task`" in plan_messages[1][-1].content


def test_coordinator_uses_generated_task_without_semantic_checks():
    from agents.coordinator import coordinator_chat

    model = _CoordinatorModel(
        {
            "submit_worker_plan": [
                _tool_message(
                    "submit_worker_plan",
                    {"steps": [{"task": "Получи факт напрямую."}]},
                    "plan-shortened",
                )
            ],
            "submit_upstream_output": _responses(
                answer="Факт получен."
            )["submit_upstream_output"],
        }
    )
    model_patch, callback_patch, trace_patch = _patches(model)
    with (
        model_patch,
        callback_patch,
        trace_patch,
        patch(
            "agents.coordinator.worker_chat",
            return_value=_outcome("Факт получен."),
        ) as worker,
    ):
        result = coordinator_chat("Получи факт.")

    assert result == CoordinatorAnswer(answer="Факт получен.", display_refs=[])
    worker.assert_called_once_with("Получи факт напрямую.")
    plan_calls = [
        messages
        for name, messages in model.messages
        if name == "submit_worker_plan"
    ]
    assert len(plan_calls) == 1
    assert all(name != "dispatch_worker" for name, _ in model.messages)
