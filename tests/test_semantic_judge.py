import json

from agents.semantic_judge import (
    IdentifierEvidenceAudit,
    SemanticJudgeVerdict,
    _IDENTIFIER_AUDIT_PROMPT,
    judge_agent_response,
)


class _StructuredJudge:
    def __init__(self, result):
        self.result = result
        self.messages = None
        self.retry_attempts = None

    def with_retry(self, *, stop_after_attempt):
        self.retry_attempts = stop_after_attempt
        return self

    def invoke(self, messages):
        self.messages = list(messages)
        return self.result


class _JudgeModel:
    def __init__(self, result):
        self.structured = _StructuredJudge(result)
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, method=None):
        self.schema = schema
        self.method = method
        return self.structured


class _SchemaAwareJudgeModel:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def with_structured_output(self, schema, method=None):
        self.calls.append(schema)
        return _StructuredJudge(self.results[schema])


def test_semantic_judge_receives_only_user_visible_result():
    model = _JudgeModel(
        {"status": "passed", "reason": "Путь и глубина указаны."}
    )

    verdict = judge_agent_response(
        query="Покажи путь и глубину.",
        answer="A → B, глубина 1.",
        display_items=[{"name": "path", "content": '{"path":["A","B"]}'}],
        model=model,
    )

    assert verdict == SemanticJudgeVerdict(
        status="passed",
        reason="Путь и глубина указаны.",
    )
    assert model.schema is SemanticJudgeVerdict
    assert model.method == "function_calling"
    assert model.structured.retry_attempts == 3
    payload = json.loads(model.structured.messages[1].content)
    assert payload == {
        "query": "Покажи путь и глубину.",
        "answer": "A → B, глубина 1.",
        "display_results": [
            {"name": "path", "content": '{"path":["A","B"]}'}
        ],
    }
    assert "worker" not in model.structured.messages[1].content.lower()
    judge_prompt = model.structured.messages[0].content
    assert "Выдели только явно запрошенные части query" in judge_prompt
    assert "не требуется фактически исполнять и измерять" in judge_prompt
    assert "квалифицированная ссылка" in judge_prompt
    assert "Downstream impact от source и есть\nreverse lineage" in judge_prompt
    assert "операция не\nзапрошена" in judge_prompt
    assert "каталоговых/семантических кандидатов недостаточно" in judge_prompt
    assert "source→target-пара, правило" in judge_prompt
    assert "слово «перечисли» само по себе" in judge_prompt
    assert "второго направления не требуй" in judge_prompt
    assert "upstream" not in judge_prompt
    assert "пустой/архивный display сам по себе не является ошибкой" in judge_prompt
    assert "ровно одну самую существенную" in judge_prompt
    assert "не придумывай отсутствующие в query требования" in judge_prompt
    assert "транзитивный обход до terminal targets" in judge_prompt
    assert "Следуй оставшимся проверкам строго по порядку" in judge_prompt
    assert "Evidence-аудит новых физических идентификаторов уже выполнен" in judge_prompt
    assert "до полного B" in judge_prompt
    assert "B::subquery" in judge_prompt

    audit_prompt = _IDENTIFIER_AUDIT_PROMPT
    assert "таблиц, колонок, ключей" in audit_prompt
    assert "`display_results[*].content`" in audit_prompt
    assert "Сам answer не подтверждает" in audit_prompt
    assert "`display_results=[]` означает ноль evidence" in audit_prompt
    assert "Не считай\nлогическое продолжение или правдоподобие" in audit_prompt
    assert "placeholders в угловых скобках" in audit_prompt


def test_semantic_judge_rejects_llm_reported_unconfirmed_identifier():
    model = _SchemaAwareJudgeModel(
        {
            IdentifierEvidenceAudit: {
                "unconfirmed_identifiers": [
                    "known_table",
                    "placeholder_column",
                    "made_up_column",
                ]
            },
            SemanticJudgeVerdict: {
                "status": "passed",
                "reason": "Не должен вызываться.",
            },
        }
    )

    verdict = judge_agent_response(
        query="Для file_id=3 составь SQL для known_table.",
        answer="SELECT <placeholder_column>, made_up_column FROM known_table",
        display_items=[],
        model=model,
    )

    assert verdict.status == "failed"
    assert "made_up_column" in verdict.reason
    assert model.calls == [IdentifierEvidenceAudit]
