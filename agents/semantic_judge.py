"""Optional LLM-as-judge for saved live-agent responses."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Mapping, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .llm_factory import create_judge_chat_model


JUDGE_MAX_DISPLAY_CHARS = 24_000
JUDGE_MAX_HISTORY_MESSAGES = 12
JUDGE_MAX_HISTORY_MESSAGE_CHARS = 8_000
JUDGE_MAX_HISTORY_CHARS = 16_000

_IDENTIFIER_QUOTES = str.maketrans("", "", "`\"")
_DOT_WHITESPACE_RE = re.compile(r"\s*\.\s*")


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


class IdentifierEvidenceFinding(BaseModel):
    """One model-classified answer surface absent from visible evidence."""

    model_config = ConfigDict(extra="forbid")

    surface: str = Field(min_length=1)
    kind: Literal["physical_identifier", "generic_technical_term"]

    @field_validator("surface")
    @classmethod
    def _strip_surface(cls, value: str) -> str:
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("surface must not be blank")
        return clean_value


class IdentifierEvidenceAudit(BaseModel):
    """Model-owned classification of answer surfaces absent from evidence."""

    model_config = ConfigDict(extra="forbid")

    findings: list[IdentifierEvidenceFinding] = Field(
        default_factory=list,
        description=(
            "Термины из answer, отсутствующие в пользовательском запросе и "
            "видимых display-results, с классификацией каждого термина."
        ),
    )


_IDENTIFIER_AUDIT_PROMPT = """
Ты выполняешь только evidence-аудит терминов, без оценки полноты или полезности
ответа. Найди встречающиеся в answer термины, которые дословно отсутствуют в query,
пользовательских сообщениях history и `display_results[*].content`, и классифицируй
каждый термин:

- `physical_identifier` — конкретное собственное имя физической таблицы, колонки,
  схемы, ключа, справочника или поля фильтра;
- `generic_technical_term` — общий тип ограничения, SQL-конструкция, имя атрибута
  метаданных, роль сущности, служебная категория или иное нарицательное понятие.

Evidence требуется только для `physical_identifier`. Общий технический термин не
становится физическим идентификатором из-за подчёркивания, верхнего регистра,
косой черты или использования в техническом описании.

History передана в хронологическом порядке с явными ролями. Сообщение assistant
само по себе не подтверждает физический идентификатор; подтверждением считается
только query, сообщение user либо display-result.

`display_results=[]` означает ноль evidence. Сам answer не подтверждает собственные
утверждения. SQL-шаблон, план и тест-протокол не являются исключениями. Не считай
логическое продолжение или правдоподобие подтверждением. Не включай псевдонимы и
placeholders в угловых скобках. Не включай имя, если answer явно предлагает его
только как неподтверждённый вариант в уточняющем вопросе и не использует как факт.
Ничего не объясняй: верни только структурированный список `findings`.
""".strip()


_JUDGE_PROMPT = """
Ты независимый LLM-as-judge. Оцени выполнение текущего query с учётом
role-aware history по пользовательским answer и display_results.

Каждый элемент `display_results` уже является отдельным пользовательским
scrollable UI-блоком; его `content` — показанное пользователю полное содержимое
этого блока. Не требуй от answer повторять эти строки или дополнительно описывать
механизм отображения.

History передана в хронологическом порядке. Явные факты, определения и правила
user считаются условиями задачи; текст assistant сам по себе не делает факт
подтверждённым. Более позднее явное сообщение user отменяет противоречащее раннее.
Если query зависит от неоднозначной либо только предположенной assistant ссылки,
краткий уточняющий вопрос является корректным ответом, а догадка — failed.

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


def _normalize_identifier_surface(value: Any) -> str:
    """Normalize only quoting and dot spacing in an identifier surface."""

    without_identifier_quotes = str(value or "").translate(_IDENTIFIER_QUOTES)
    return _DOT_WHITESPACE_RE.sub(".", without_identifier_quotes).casefold()


def _identifier_surface_occurs(name: str, text: str) -> bool:
    """Match an identifier surface without accepting longer-name substrings."""

    if not name:
        return False
    return re.search(
        rf"(?<![\w$]){re.escape(name)}(?![\w$])",
        text,
    ) is not None


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


def _compact_history(history: Sequence[Any]) -> list[dict[str, str]]:
    compact: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        compact.append(
            {
                "role": role,
                "content": content[:JUDGE_MAX_HISTORY_MESSAGE_CHARS],
            }
        )

    compact = compact[-JUDGE_MAX_HISTORY_MESSAGES:]
    while (
        len(compact) > 1
        and sum(len(item["content"]) for item in compact)
        > JUDGE_MAX_HISTORY_CHARS
    ):
        compact.pop(0)
    if compact:
        compact[-1]["content"] = compact[-1]["content"][:JUDGE_MAX_HISTORY_CHARS]
    return compact


def judge_agent_response(
    *,
    query: str,
    answer: Any,
    display_items: Sequence[Any],
    history: Sequence[Any] = (),
    model: Any = None,
) -> SemanticJudgeVerdict:
    """Judge one completed exchange using the configured real chat model."""
    judge_model = model or create_judge_chat_model(timeout=180)
    compact_display = _compact_display_items(display_items)
    compact_history = _compact_history(history)
    payload = {
        "query": str(query or ""),
        "answer": answer,
        "display_results": compact_display,
        "display_contract": {
            "each_item_is_separate_ui": True,
            "each_item_is_scrollable": True,
            "content_is_user_visible": True,
        },
    }
    if compact_history:
        payload["history"] = compact_history
    human_message = HumanMessage(
        content=json.dumps(payload, ensure_ascii=False)
    )

    # Run the evidence boundary for every judged response. Deciding whether a
    # query is "about data" from natural-language keywords is itself an
    # unreliable intent heuristic.
    audit_result = _invoke_structured(
        judge_model,
        IdentifierEvidenceAudit,
        [SystemMessage(content=_IDENTIFIER_AUDIT_PROMPT), human_message],
    )
    audit = IdentifierEvidenceAudit.model_validate(audit_result)
    confirmed_request_text = _normalize_identifier_surface(
        "\n".join(
            [
                str(query or ""),
                *(
                    item["content"]
                    for item in compact_history
                    if item["role"] == "user"
                ),
            ]
        )
    )
    answer_text = _normalize_identifier_surface(answer)
    display_text = _normalize_identifier_surface(
        "\n".join(item.get("content", "") for item in compact_display)
    )
    unconfirmed: list[str] = []
    for finding in audit.findings:
        if finding.kind != "physical_identifier":
            continue
        name = finding.surface.strip()
        normalized_name = _normalize_identifier_surface(name)
        if not name or not normalized_name or "<" in name or ">" in name:
            continue
        # The audit model extracts identifiers *from the answer*. Treat that
        # provenance as a code-side invariant. SQL identifier quotes and dot
        # spacing are surface syntax; no other spelling conversion (for
        # example, underscore <-> dot) is accepted.
        if not _identifier_surface_occurs(normalized_name, answer_text):
            continue
        if re.search(
            rf"<\s*{re.escape(normalized_name)}\s*>",
            answer_text,
        ):
            continue
        if _identifier_surface_occurs(
            normalized_name,
            confirmed_request_text,
        ):
            continue
        if _identifier_surface_occurs(normalized_name, display_text):
            continue
        unconfirmed.append(name)
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
    "IdentifierEvidenceFinding",
    "SemanticJudgeVerdict",
    "judge_agent_response",
]
