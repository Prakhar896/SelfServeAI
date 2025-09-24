from typing import List, Optional
from pydantic import BaseModel
import time, asyncio, json
from fastapi import FastAPI
from starlette.responses import StreamingResponse

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mock-gpt"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

async def _resp_async_generator(model: str, text_resp: str):
    tokens = text_resp.split(" ")
    
    for i, token in enumerate(tokens):
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": model,
            "choices": [{"delta": {"content": token + " "}}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(1)
    
    yield "data: [DONE]\n\n"

app = FastAPI(title='OpenAI Compatible API')

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.messages:
        response_content = "As a mock AI assistant, I received your message: " + request.messages[-1].content
    else:
        response_content = 'As a mock AI assistant, I can only echo messages. No previous messsages found.'
    
    if request.stream:
        return StreamingResponse(_resp_async_generator(request.model, response_content), media_type="application/x-ndjson")
    
    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": ChatMessage(role="assistant", content=response_content),
            "logprobs": None,
            "finish_reason": "stop"
        }]
    }
