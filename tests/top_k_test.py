import openai
from document_store import setup_retriever

# Ensure your OpenAI API key is loaded
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the retriever
retriever = setup_retriever()



def test_top_k_values(query, retriever, top_k_values=[1, 3, 5, 6]):
    results = {}

    for k in top_k_values:
        print(f"\nTesting with top_k={k}")
        retrieved_docs = retriever.retrieve(query, top_k=k)
        
        # Format context for testing
        context = "\n\n".join([f"Page {doc.meta['page_num']}:\n{doc.content}" for doc in retrieved_docs])

        # Define prompt with context and query
        prompt = f"""
        Context:
    You are the AI companion for the Windward Bound, a Sailing Ship located in Hobart which conducts educational and training opportunities for those in need. You have access to the history of the ship, as well as its missions, voyages, 
    and experiences of those who have come onboard as well as the crew who mans you. 
    {context}

    Question: {query}
    
    Answer as specifically as possible, and use the boat's name, "Windeward Bound", in your answer if relevant. Although, there is a second boat referenced in the material named, "Mistral II"(which is a new and current project being built by the Windeward Bound trust). Avoid repeating information you've already stated, and keep the tone conversational and friendly. 
    After answering, continue the conversation by naturally extrapolating on the users question or comment, suggesting a follow up such as "I can tell you more about the history or how it was built if you would like."
    Answer:
        """
        
        # Query OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" if available
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )

        # Store the answer for this top_k value
        answer = response.choices[0].message["content"].strip()
        results[k] = answer
        print(f"top_k={k} Response:\n{answer}\n{'='*40}")

    return results

# Test the function with a sample query
query = "Is this a pirate ship?"
results = test_top_k_values(query, retriever)
