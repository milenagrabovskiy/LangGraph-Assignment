from typing import TypedDict, Annotated

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class State(TypedDict):
    question: str
    query: str
    documents: list[Document]
    retry_count: int
    answer: str
    messages: Annotated[list[AnyMessage], add_messages]