import openai

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # or "gpt-4" if available
    messages=[{"role": "user", "content": "Say hello!"}],
    max_tokens=5
)

print(response.choices[0].message["content"].strip())