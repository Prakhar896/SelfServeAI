from openai import OpenAI

client = OpenAI(
    api_key='real-api-key',
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

# stream = client.chat.completions.create(
#     model="mock-gpt",
#     messages=[{"role": "user", "content": input("Enter a message: ")}],
#     stream=True,
# )

# for chunk in stream:
#     print(chunk.choices[0].delta.content or "")