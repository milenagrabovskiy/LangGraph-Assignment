from .graph import graph



#empty state
state = {
    "question": "",
    "query": "",
    "documents": [],
    "retry_count": 0,
    "answer": "",
    "messages": []
}


while True:
    question = input("User >>>\n ")

    if question.lower() in {"quit", "exit"}:
        break

    state["question"] = question
    state = graph.invoke(state)

    print(f"Assistant >>>\n {state['answer']}")