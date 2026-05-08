"""LLM-backed agents: Excel headers, summaries, schema match, generic chat, S2T insights."""


def __getattr__(name: str):
    if name == "insights_chat":
        from agents.insights_agent import insights_chat as _insights_chat

        return _insights_chat
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
