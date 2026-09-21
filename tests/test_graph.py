from ticket_assistant.graph import graph


def test_graph_nodes():
    graph_data = graph.get_graph()
    node_names = set(graph_data.nodes.keys())

    assert "prepare_query" in node_names
    assert "retriever" in node_names
    assert "answer" in node_names
    assert "rewrite" in node_names
    assert "refuse" in node_names


def test_graph_edges():
    graph_data = graph.get_graph()

    edges = {(edge.source, edge.target)
        for edge in graph_data.edges}

    assert ("prepare_query", "retriever") in edges
    assert ("rewrite", "retriever") in edges
    assert ("answer", "__end__") in edges
    assert ("refuse", "__end__") in edges