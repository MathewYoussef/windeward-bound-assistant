import openai
from document_store import setup_retriever

# Ensure your OpenAI API key is loaded
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the retriever
retriever = setup_retriever()

def generate_response_with_rag(query, retriever, boat_name="Windeward Bound", top_k=3):
    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(query, top_k=top_k)
    
    # Format the context from retrieved documents
    context = "\n\n".join([f"Page {doc.meta['page_num']}:\n{doc.content}" for doc in retrieved_docs])
    
    # Combine query and context for the language model prompt
    prompt = f"""
    Context:
    You are the AI companion for the Windward Bound, a Sailing Ship located in Hobart which conducts educational and training opportunities for those in need. You have access to the history of the ship, as well as its missions, voyages, 
    and experiences of those who have come onboard as well as the crew who mans you. 
    {context}

    Question: {query}
    
    Answer as specifically as possible, and use the boat's name, "Windeward Bound", in your answer if relevant. Although, there is a second boat referenced in the material named, "Mistral II". Avoid repeating information you've already stated, and keep the tone conversational and friendly. 
    After answering, continue the conversation by naturally extrapolating on the users question or comment, suggesting a follow up such as "I can tell you more about the history or how it was built if you would like."
    Answer:
    """
    
    # Query the OpenAI language model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4" if available
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    
    # Extract and return the response content
    answer = response.choices[0].message["content"].strip()
    return answer

# Test the function with the boat name and sample query
query = "Who built the boat and how was it built?"
response = generate_response_with_rag(query, retriever)
print(f"Query: {query}\nResponse: {response}")


