import time
from fastapi import APIRouter, Depends
from .model import ChatCompletionRequest, ChatMessage
from . import service
from ..database.core import db_required

router = APIRouter(
    prefix='/chat',
    tags=['Chat Completions']
)

@router.post("/completions")
async def chat_completions(request: ChatCompletionRequest, db: db_required):
    response_content, db_interaction = service.obtainResponse(request, db)
    
    return {
        "id": db_interaction.id,
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
