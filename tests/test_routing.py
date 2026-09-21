from langchain_core.documents import Document
from ticket_assistant.routing import route_after_retrieval


def test_route_to_answer_when_documents_found():
    state = {"documents": [Document(page_content="Policy information")],
            "retry_count": 0}

    assert route_after_retrieval(state) == "answer"



def test_retry_bound_refuses_after_retry_used():
    state = {"documents": [],
        "retry_count": 1}

    assert route_after_retrieval(state) == "refuse"