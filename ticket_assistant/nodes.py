from langchain_core.messages import HumanMessage, AIMessage

from .llm import get_chat_model
from .state import State
from .retriever import get_retriever



def retrieve_node(state: State) -> dict:
    retriever = get_retriever()
    documents = retriever.invoke(state["query"])

    return {
        "documents": documents
    }



# def route_after_retrieval(state: State) -> str:
#
#     # decide what to do after retrieving documents
#     if state["documents"]: # if Document objs saved into state
#         return "answer" # then answer with augmented prompt
#
#     if state["retry_count"] < 1:
#         return "rewrite" # rewrite and retrieve again
#
#     return "refuse" # if already tried rewriting but cant get relevant docs, refuse



def rewrite_node(state: State) -> dict:
    model = get_chat_model()

    response = model.invoke(f"Rewrite this query to improve retrieval. Query: {state['query']}"
                            f"Only return the rewritten query and nothing else.")

    return {
        "query": response.text,
        "retry_count": state["retry_count"] + 1
    }


def answer_node(state: State) -> dict:
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n"
        f"{doc.page_content}"
        for doc in state["documents"]
    )

    answer = get_chat_model().invoke(
        f"""
Answer the question using ONLY the context provided.

If the context does not contain enough information to answer the question,
say: "I don't know based on the provided documents."

Do not guess or use outside knowledge.
If you answer, cite the source filename(s).

Question: {state["question"]} Context: {context}
"""
    )

    return {
        "answer": answer.text,
        "messages": [AIMessage(content=answer.text)]
    }


def refuse_node(state: State) -> dict:
    refusal = "Couldn't find information to answer this question"

    return {"answer": refusal, "messages": [AIMessage(content=refusal)]}



def prepare_query_node(state: State) -> dict:
    history = state["messages"] or []

    if not history:
        return {
            "query": state["question"],
            "messages": [HumanMessage(content=state["question"])],
            "retry_count": 0,
            "documents": []
        }

    response = get_chat_model().invoke(
        history + [HumanMessage(content=(
                    "Rewrite the current question as a standalone search query "
                    "using the previous conversation for context.\n\n"
                    f"Current question: {state['question']}\n\n"
                    "Return only the rewritten search query."
                ))]
    )

    return {
        "query": response.text,
        "messages": [HumanMessage(content=state["question"])]
    }
