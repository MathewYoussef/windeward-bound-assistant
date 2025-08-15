# document_store.py
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
import json

def setup_retriever():
    # Initialize the document store with embedding support and correct embedding dimension
    document_store = InMemoryDocumentStore(embedding_dim=768)

    # Load the extracted text data from the JSON file
    with open("extracted_text.json", "r") as json_file:
        text_data = json.load(json_file)

    # Prepare documents for indexing in Haystack format
    documents = [
        {"content": page["content"], "meta": {"page_num": page["page_num"]}}
        for page in text_data
    ]

    # Write documents to the document store
    document_store.write_documents(documents)

    # Initialize the embedding retriever with the larger model
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )

    # Update the document store with embeddings
    document_store.update_embeddings(retriever)

    return retriever

# This block runs only if document_store.py is executed directly
if __name__ == "__main__":
    # Set up the retriever
    retriever = setup_retriever()

    # Print the indexed documents to confirm
    print("Indexed documents:")
    for doc in retriever.document_store.get_all_documents():
        print(f"Page: {doc.meta['page_num']}\nContent: {doc.content}\n{'='*40}")