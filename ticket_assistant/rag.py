""" Using retrievers as a part of a chain for RAG """

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .llm import get_chat_model


GROUNDED_SYSTEM = """You answer question's about our company's internal policies.

The policy excerpts below are the ONLY source you may use. They are this
company's internal policy and they override anything you believe about how
the company normally operates.

- If the excerpts answer the question, answer from them and cite the filenames.
- If the excerpts do NOT cover the question, say exactly "The provided documents do not cover that."
- Do not fill the gap from general knowledge.
- Quote specific figures (days, percentages, prices) exactly as written.
- End your answer with a "Sources:" line naming the documents you used.

Policy excerpts:
{context}"""


def format_docs(docs: list[Document]) -> str:
    """ turn each found doc into one long string that can be added to the model system prompt """

    blocks = []
    for doc in docs:
        meta = doc.metadata
        header = f"source: {meta.get("source", "unknown")}"
        if meta.get("section"):
            header += f" | section: {meta.get("section")}"

        blocks.append(f"--- {header} ---\n{doc.page_content.strip()}")

    result = "\n\n".join(blocks)
    return result


def build_rag_chain() -> Runnable:

    prompt = ChatPromptTemplate.from_messages([
        ("system", GROUNDED_SYSTEM),
        ("human", "{question}")
    ])

    return prompt | get_chat_model()


if __name__ == "__main__":

    print("=== TRYING RAG CHAIN ===")

    docs = [
        Document(
            page_content="Customers may return an item within 30 days.",
            metadata={"source": "refund-policy.md"}
        )
    ]

    chain = build_rag_chain()

    print(chain.invoke({
        "question": "How long does a customer have to return an item?",
        "context": format_docs(docs)
    }))