from ticket_assistant.retriever import load_document_chunks


def test_documents_load_and_split():
    chunks = load_document_chunks()
    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert "source" in chunk.metadata