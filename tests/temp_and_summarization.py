import openai
from document_store import setup_retriever
from transformers import pipeline
import os

# Ensure your OpenAI API key is loaded
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the retriever
retriever = setup_retriever()

# Initialize summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define the filter function to remove unnecessary details
def filter_unnecessary_details(summary):
    exclude_terms = ["does not work right now..."]  # Add any names or terms to exclude here
    for term in exclude_terms:
        summary = summary.replace(term, "")
    return summary.strip()

# Modify summarize_content to include filtering
def summarize_content(content, max_length=50):
    summary = summarizer(content, max_length=max_length, min_length=15, do_sample=False)
    return filter_unnecessary_details(summary[0]["summary_text"])

# Main RAG function
def generate_response_with_rag(query, retriever, use_summarization=False, temperature=0.5):
    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(query, top_k=5)
    
    # Conditionally summarize if required
    context = ""
    for doc in retrieved_docs:
        content = doc.content
        if use_summarization and len(content) > 500:  # Trigger based on length or other criteria
            content = summarize_content(content)
        context += f"Page {doc.meta['page_num']}:\n{content}\n\n"
    
    # Construct prompt with the context and query (outside of the loop)
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
        max_tokens=150,
        temperature=temperature
    )

    return response.choices[0].message["content"].strip()

# Example usage
query = "Can you summarize the voyages of the Windward Bound?"
response = generate_response_with_rag(query, retriever, use_summarization=True, temperature=0.7)
print(f"Query: {query}\nResponse: {response}")
