from langgraph.constants import START, END

from .state import State
from langgraph.graph import StateGraph

from .nodes import retrieve_node, answer_node, refuse_node, rewrite_node, prepare_query_node
from .routing import route_after_retrieval

graph_builder = StateGraph(State)

# add nodes
graph_builder.add_node("prepare_query", prepare_query_node)
graph_builder.add_node("retriever", retrieve_node)
graph_builder.add_node("answer", answer_node)
graph_builder.add_node("rewrite", rewrite_node)
graph_builder.add_node("refuse", refuse_node)

graph_builder.add_edge(START, "prepare_query")

graph_builder.add_edge("prepare_query", "retriever")

graph_builder.add_conditional_edges("retriever", route_after_retrieval,
                                   {
                                       "answer": "answer",
                                       "rewrite": "rewrite",
                                       "refuse": "refuse"
                                   }
                                   )

graph_builder.add_edge("rewrite", "retriever")
graph_builder.add_edge("answer", END)
graph_builder.add_edge("refuse", END)


graph = graph_builder.compile()

if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())