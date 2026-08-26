"""Optional LLM-as-judge for saved live-agent responses."""

from __future__ import annotations

import json
from typing import Any, Literal, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .llm_factory import create_chat_model


JUDGE_MAX_DISPLAY_CHARS = 24_000


class SemanticJudgeVerdict(BaseModel):
    """Strict semantic verdict for one user-visible exchange."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["passed", "failed"]
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("reason must not be blank")
        return clean_value


_JUDGE_PROMPT = """
Ты независимый LLM-as-judge. Оцени только фактическую полноту и корректность
пользовательского результата относительно исходного запроса.

Верни status=passed, только если совокупность answer и display_results:
- отвечает на все явно запрошенные части;
- сохраняет указанные объекты, роли, направление и ограничения;
- не содержит фактических противоречий;
- использует display_results, когда пользователь явно просил полный результат.

Не штрафуй за стиль, краткость и отсутствие необязательных деталей. Не считай
внутренние трассы доказательством: оценивай только answer и display_results.
При status=failed кратко и конкретно назови отсутствующий или неверный факт.
""".strip()


def _compact_display_items(display_items: Sequence[Any]) -> list[dict[str, str]]:
    remaining = JUDGE_MAX_DISPLAY_CHARS
    compact: list[dict[str, str]] = []
    for item in display_items:
        if remaining <= 0:
            break
        if isinstance(item, dict):
            name = str(item.get("name") or "")
            content = str(item.get("content") or "")
        else:
            name = str(getattr(item, "name", "") or "")
            content = str(getattr(item, "content", "") or "")
        clipped = content[:remaining]
        remaining -= len(clipped)
        compact.append({"name": name, "content": clipped})
    return compact


def judge_agent_response(
    *,
    query: str,
    answer: Any,
    display_items: Sequence[Any],
    model: Any = None,
) -> SemanticJudgeVerdict:
    """Judge one completed exchange using the configured real chat model."""
    judge_model = model or create_chat_model(timeout=180)
    try:
        structured = judge_model.with_structured_output(
            SemanticJudgeVerdict,
            method="function_calling",
        )
    except TypeError:
        structured = judge_model.with_structured_output(SemanticJudgeVerdict)
    runnable = structured.with_retry(stop_after_attempt=3)
    result = runnable.invoke(
        [
            SystemMessage(content=_JUDGE_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {
                        "query": str(query or ""),
                        "answer": answer,
                        "display_results": _compact_display_items(display_items),
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )
    return SemanticJudgeVerdict.model_validate(result)


__all__ = ["SemanticJudgeVerdict", "judge_agent_response"]
