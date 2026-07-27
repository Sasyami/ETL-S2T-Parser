"""Tool for exposing the planner's current progress."""

from typing import Dict

from langchain_core.tools import tool


@tool(parse_docstring=True)
def show_plan(done: str, to_do: str) -> Dict[str, str]:
    """Показать уже выполненную и оставшуюся части текущего плана.

    Используй инструмент, когда перед следующим действием полезно явно
    зафиксировать ход многошаговой задачи. Он не читает и не изменяет данные,
    а только возвращает переданный planner план в ToolMessage, чтобы дальнейшее
    рассуждение опиралось на явные завершённые и предстоящие шаги.

    Args:
        done: Что уже выполнено и какие факты получены на текущем этапе.
        to_do: Что ещё требуется сделать для ответа на запрос пользователя.
    """
    return {
        "done": done,
        "to_do": to_do,
    }
