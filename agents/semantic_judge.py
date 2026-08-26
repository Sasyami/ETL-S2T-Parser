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
Ты независимый LLM-as-judge. Оцени только выполнение дословного query
по пользовательским answer и display_results.

Сначала мысленно выдели только явные требования query. Не добавляй новые
действия, факты, форматы или критерии. В частности:
- если просят составить протокол, план, шаблон или критерии, не требуй их
  фактического выполнения и измеренных результатов;
- не требуй повторять в answer уже заданные в query идентификаторы, если ответ
  однозначен;
- таблица уже перечислена, если её полное имя входит в квалифицированную ссылку
  table.column; не требуй дублировать её отдельным списком;
- полный сохранённый SQL допустим как transformation rule, если query не требует
  именно выражение одного поля;
- в этом ETL-домене reverse lineage от изменяемого source означает downstream impact:
  обход от source к зависимым targets. Возвращённый downstream и есть reverse lineage;
  второго «обратного» направления нет.

Верни status=passed, если совокупность answer и display_results:
- отвечает на все явно запрошенные части;
- сохраняет объекты, роли, направление и ограничения;
- не содержит явных фактических противоречий.
Фраза answer «операция не запрошена» противоречит query, если эта операция там явно запрошена.

Пустой или архивно недоступный payload display_results не доказывает ошибку answer.
Не ставь failed только из-за невозможности независимо перепроверить факт по display.
Но если пользователь явно просил показать полный табличный результат, отсутствие такого
результата в answer и display_results является невыполненным требованием.
Слово «перечисли» само по себе не является требованием display или табличного payload.

Никогда не используй как единственную причину status=failed:
- отсутствие второго «обратного» направления в downstream impact;
- недоступность архивного display payload, если требуемый факт уже есть в answer;
- невозможность независимо подтвердить факт, если в answer нет явного противоречия.

Не штрафуй за стиль, краткость и необязательные детали. При status=failed укажи ровно
одну, самую существенную критическую ошибку: конкретное явное требование query,
которое не выполнено, либо конкретное противоречие. Не добавляй вторичные претензии
и не называй в reason требование, которого нет в query.

Перед status=passed для impact/reverse-lineage query обязательно проверь: answer должен либо
показать транзитивные уровни до terminal targets, либо явно сказать, что полный
транзитивный обход завершён и дальнейших descendants нет. Простой список первого hop
без такого подтверждения означает status=failed.
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
