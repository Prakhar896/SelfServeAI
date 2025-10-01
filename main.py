from dotenv import load_dotenv
load_dotenv()
import time, asyncio, json, uuid
from typing import Annotated, List
from fastapi import FastAPI, HTTPException, Depends
from starlette.responses import JSONResponse
from schemas import ChatMessage, ChatCompletionRequest
from sqlalchemy.orm import Session
from database import engine, get_db
import models

app = FastAPI(title='SelfServeAI')
models.Base.metadata.create_all(bind=engine)

db_required = Annotated[Session, Depends(get_db)]

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, db: db_required):
    if request.messages:
        response_content = "As a mock AI assistant, I received your message: " + request.messages[-1].content
    else:
        response_content = 'As a mock AI assistant, I can only echo messages. No previous messages found.'
    
    db_interaction = models.Interaction(id=uuid.uuid4().hex, prompt=request.messages[-1].content if request.messages else "", response=response_content)
    db.add(db_interaction)
    db.commit()
    db.refresh(db_interaction)
    
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
