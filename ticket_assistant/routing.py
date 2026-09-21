from .state import State


def route_after_retrieval(state: State) -> str:
    if state["documents"]:
        return "answer"

    if state["retry_count"] < 1:
        return "rewrite"

    return "refuse"