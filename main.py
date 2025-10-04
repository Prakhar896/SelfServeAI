import os
from dotenv import load_dotenv
load_dotenv()
import time, asyncio, json, uuid
from typing import Annotated, List
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from starlette.responses import JSONResponse
from schemas import ChatMessage, ChatCompletionRequest
from sqlalchemy.orm import Session
from database import engine, get_db
from middleware import AuthMiddleware
import models
from slm import SmolLM

app = FastAPI(title='SelfServeAI')
models.Base.metadata.create_all(bind=engine)
if os.getenv('RESPONSE_MODE', 'mock') == 'slm':
    SmolLM.load()

app.add_middleware(AuthMiddleware)

db_required = Annotated[Session, Depends(get_db)]

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, db: db_required):
    response_content = None
    if os.getenv('RESPONSE_MODE', 'mock') != 'slm':
        if request.messages:
            response_content = "As a mock AI assistant, I received your message: " + request.messages[-1].content
        else:
            response_content = 'As a mock AI assistant, I can only echo messages. No previous messages found.'
    else:
        response_content = SmolLM.infer(request.messages[-1].content if request.messages else "No input from user.", max_new_tokens=int(os.getenv('MAX_TOKENS', 100)))
    
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
