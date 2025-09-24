from openai import OpenAI


client = OpenAI(
    api_key='fake-api-key',
    base_url='http://localhost:8000'
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": input("Enter a message: ")
        }
    ],
    model="mock-gpt"
)

print()
print("Response:")
print(chat_completion.choices[0].message.content)