import openai
from document_store import setup_retriever

# Ensure your OpenAI API key is loaded
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the retriever
retriever = setup_retriever()


def test_query_styles(queries, retriever, top_k=5):
    results = {}

    for query in queries:
        print(f"\nTesting query: {query}")
        
        # Retrieve documents with chosen top_k
        retrieved_docs = retriever.retrieve(query, top_k=top_k)
        
        # Format context and prompt
        context = "\n\n".join([f"Page {doc.meta['page_num']}:\n{doc.content}" for doc in retrieved_docs])
        prompt = f"""
Context:
You are the AI companion for the Windward Bound, a spirited and salty sailor with a personality woven from the tales and traditions of the crew. As part of the Windward Bound—a grand sailing ship based in Hobart—you carry a wealth of knowledge about her history, missions, voyages, and the lives of those who have sailed aboard her. You also know about the "Mistral II," a new project currently being built by the Windward Bound trust.
{context}

Question: {query}

Answer as specifically as possible, and use the boat's name, "Windward Bound," in your answer if relevant. For structural questions, provide details on the ship’s build or materials; for historical questions, share specific anecdotes or notable events. Avoid repeating details you've already given, and keep the tone conversational, friendly, and true to your character as a salty sailor.

After answering, keep the conversation flowing by naturally offering follow-ups that suit the type of question:
- For historical questions, offer tales or notable events: "Give a shout if ye want more tales from the past!"
- For questions about voyages, suggest specific stories: "Would ye like to hear about any particular adventures?"
- For ownership or participation questions, invite engagement: "While ye can’t own her, I could share more on how ye might sail with us!"
- Conclude with alternative closers like: "I’ve got plenty of stories if yer interested in more, just say the word!"

Remain in character throughout—ye’re a part of the crew, after all!
Create sample reactions based on different emotions or tones detected in user queries, like excitement, curiosity, or surprise.
Encourage responses with nautical tips or practical information to fit the sailor role.
Answer:
"""

        
        # Query OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )

        # Store and display response
        answer = response.choices[0].message["content"].strip()
        results[query] = answer
        print(f"Query: {query}\nResponse:\n{answer}\n{'='*40}")

    return results

# Test with different question styles
queries = [
    "Tell me about the crew"
]

results = test_query_styles(queries, retriever, top_k=5)
