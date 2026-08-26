import json

from agents.semantic_judge import SemanticJudgeVerdict, judge_agent_response


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
    assert "только явные требования query" in judge_prompt
    assert "фактического выполнения и измеренных результатов" in judge_prompt
    assert "квалифицированную ссылку" in judge_prompt
    assert "Возвращённый downstream и есть reverse lineage" in judge_prompt
    assert "операция не запрошена" in judge_prompt
    assert "перечисли» само по себе" in judge_prompt
    assert "второго «обратного» направления нет" in judge_prompt
    assert "upstream" not in judge_prompt
    assert "не доказывает ошибку answer" in judge_prompt
    assert "ровно\nодну, самую существенную критическую ошибку" in judge_prompt
    assert "Не добавляй вторичные претензии" in judge_prompt
    assert "которого нет в query" in judge_prompt
    assert "транзитивные уровни до terminal targets" in judge_prompt
    assert "означает status=failed" in judge_prompt
