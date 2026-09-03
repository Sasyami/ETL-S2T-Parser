"""Optional LLM-as-judge for saved live-agent responses."""

from __future__ import annotations

import json
from typing import Any, Literal, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .llm_factory import create_judge_chat_model


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


class IdentifierEvidenceAudit(BaseModel):
    """Physical identifiers introduced without user-visible evidence."""

    unconfirmed_identifiers: list[str] = Field(
        default_factory=list,
        description=(
            "Все конкретные физические идентификаторы answer, которых дословно "
            "нет в query и display_results.content."
        ),
    )


_IDENTIFIER_AUDIT_PROMPT = """
Ты выполняешь только evidence-аудит физических идентификаторов, без оценки полноты
или полезности ответа. Извлеки все конкретные имена таблиц, колонок, ключей,
справочников и полей фильтра, которые встречаются в answer, но дословно отсутствуют
и в query, и в `display_results[*].content`.

`display_results=[]` означает ноль evidence. Сам answer не подтверждает собственные
утверждения. SQL-шаблон, план и тест-протокол не являются исключениями. Не считай
логическое продолжение или правдоподобие подтверждением. Не включай SQL-ключевые
слова, псевдонимы и placeholders в угловых скобках. Ничего не объясняй: верни только
структурированный список unconfirmed_identifiers.
""".strip()


_JUDGE_PROMPT = """
Ты независимый LLM-as-judge. Оцени только выполнение дословного query
по пользовательским answer и display_results.

Evidence-аудит новых физических идентификаторов уже выполнен отдельным LLM-вызовом.
Здесь не повторяй его. Следуй оставшимся проверкам строго по порядку.

1. Проверка маршрута.
Если query задаёт путь от A до B, answer должен показать непрерывную цепочку именно
до полного B. Общий префикс недостаточен: B::subquery и B::branch не равны B.
Фраза «напрямую» при остановке на другом объекте означает failed. Для impact или
reverse lineage нужен транзитивный обход до terminal targets либо явное утверждение,
что обход завершён и дальнейших descendants нет. Downstream impact от source и есть
reverse lineage; второго направления не требуй.

2. Проверка требований.
Выдели только явно запрошенные части query. Совокупность answer и display_results
должна выполнить каждую из них, сохранить объекты, роли, направление и ограничения
и не содержать противоречий. Если требуются S2T, mapping или transformation rule,
одних каталоговых/семантических кандидатов недостаточно: нужна конкретная
source→target-пара, правило либо явный результат их поиска. Фраза «операция не
запрошена» противоречит query, если операция явно запрошена.

Уточнения:
- протокол, план или шаблон не требуется фактически исполнять и измерять;
- не требуй повторять идентификатор из query, если ответ однозначен;
- квалифицированная ссылка table.column уже называет таблицу;
- полный сохранённый SQL допустим как transformation rule, если не запросили только
  выражение одного поля;
- пустой/архивный display сам по себе не является ошибкой вне evidence-аудита;
- явно запрошенный полный табличный результат должен быть в answer или display;
- слово «перечисли» само по себе не требует отдельного display.
- если answer явно и непротиворечиво сообщает, что количество элементов
  запрошенного списка равно нулю, отсутствие отдельно напечатанного `[]` — не
  критическая ошибка: нулевой count уже однозначно задаёт пустой список.

Верни passed только после прохождения обеих проверок. Не штрафуй за стиль,
краткость и необязательные детали. При failed назови ровно одну самую существенную
критическую ошибку и не придумывай отсутствующие в query требования.
""".strip()


def _needs_identifier_audit(query: str) -> bool:
    normalized = str(query or "").casefold()
    markers = ("file_id", "s2t", "mapping", "маппинг", "схем", "каталог", "sql")
    return any(marker in normalized for marker in markers)


def _invoke_structured(model: Any, schema: type[BaseModel], messages: list[Any]) -> Any:
    try:
        structured = model.with_structured_output(schema, method="function_calling")
    except TypeError:
        structured = model.with_structured_output(schema)
    return structured.with_retry(stop_after_attempt=3).invoke(messages)


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
    judge_model = model or create_judge_chat_model(timeout=180)
    compact_display = _compact_display_items(display_items)
    payload = {
        "query": str(query or ""),
        "answer": answer,
        "display_results": compact_display,
    }
    human_message = HumanMessage(
        content=json.dumps(payload, ensure_ascii=False)
    )

    if _needs_identifier_audit(str(query or "")):
        audit_result = _invoke_structured(
            judge_model,
            IdentifierEvidenceAudit,
            [SystemMessage(content=_IDENTIFIER_AUDIT_PROMPT), human_message],
        )
        audit = IdentifierEvidenceAudit.model_validate(audit_result)
        query_text = str(query or "").casefold()
        answer_text = str(answer or "").casefold()
        display_text = "\n".join(
            item.get("content", "") for item in compact_display
        ).casefold()
        unconfirmed = [
            name.strip()
            for name in audit.unconfirmed_identifiers
            if name.strip()
            and "<" not in name
            and ">" not in name
            and f"<{name.strip().casefold()}>" not in answer_text
            and name.strip().casefold() not in query_text
            and name.strip().casefold() not in display_text
        ]
        if unconfirmed:
            return SemanticJudgeVerdict(
                status="failed",
                reason=(
                    "В ответе используется неподтверждённый физический "
                    f"идентификатор: {unconfirmed[0]}."
                ),
            )

    result = _invoke_structured(
        judge_model,
        SemanticJudgeVerdict,
        [SystemMessage(content=_JUDGE_PROMPT), human_message],
    )
    return SemanticJudgeVerdict.model_validate(result)


__all__ = [
    "IdentifierEvidenceAudit",
    "SemanticJudgeVerdict",
    "judge_agent_response",
]
