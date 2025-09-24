from typing import List, Optional
from pydantic import BaseModel
import time
from fastapi import FastAPI

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mock-gpt"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

app = FastAPI(title='OpenAI Compatible API')

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.messages and request.messages[0].role == 'user':
        response_content = "As a mock AI assistant, I received your message: " + request.messages[-1].content
    else:
        response_content = 'As a mock AI assistant, I can only echo messages. No previous messsages found.'
    
    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_content,
                "refusal": None,
                "annotations": []
            },
            "logprobs": None,
            "finish_reason": "stop"
        }]
    }
