"""Agent-facing wrapper around shared entity resolution."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List

from langchain_core.tools import tool
from pydantic import Field

from ..run_metrics import record_entity_resolution
from ..entity_resolution import (
    MAX_BATCH_ENTITIES,
    EntityMention,
    resolve_entity_batch,
)


@tool(parse_docstring=True)
def resolve_entities(
    entities: Annotated[
        List[EntityMention],
        Field(min_length=1, max_length=MAX_BATCH_ENTITIES),
    ],
) -> Dict[str, Any]:
    """Разрешить неподтверждённые имена файлов и ролевых S2T-таблиц.

    Используй batch-вызов перед exact reader только для опечатки, частичного
    имени, смыслового описания или неоднозначного mention. Уже подтверждённое
    полное техническое имя передавай сразу ролевому exact reader, не вызывая
    resolver. Для смыслового mention укажи strategy=semantic. Роли source и
    target разрешаются независимо по глобальной S2T и никогда не смешиваются.
    Не выбирай кандидата самостоятельно при status=ambiguous: результат
    сохраняет всех кандидатов и их provenance. coverage=truncated означает,
    что источник semantic-кандидатов вернул не весь набор.

    Args:
        entities: Непустой список из 1–50 mention с entity_type, role,
            strategy и опциональным file_id для catalog semantic scope.
    """
    resolutions = resolve_entity_batch(entities)
    record_entity_resolution(
        [
            {
                **item.model_dump(mode="json"),
                "resolver_invoked": True,
            }
            for item in resolutions
        ]
    )
    return {
        "resolutions": [item.model_dump(mode="json") for item in resolutions],
        "resolved": sum(item.status == "resolved" for item in resolutions),
        "ambiguous": sum(item.status == "ambiguous" for item in resolutions),
        "unresolved": sum(item.status == "unresolved" for item in resolutions),
    }


__all__ = ["resolve_entities"]
