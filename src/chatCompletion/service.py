import os, uuid
from ..entities import Interaction
from .model import ChatCompletionRequest, ChatMessage
from sqlalchemy.orm import Session
from ..slm import SmolLM

def obtainResponse(request: ChatCompletionRequest, db: Session):
    response_content = None
    if os.getenv('RESPONSE_MODE', 'mock') != 'slm':
        if request.messages:
            response_content = "As a mock AI assistant, I received your message: " + request.messages[-1].content
        else:
            response_content = 'As a mock AI assistant, I can only echo messages. No previous messages found.'
    else:
        response_content = SmolLM.infer(request.messages[-1].content if request.messages else "No input from user.", max_new_tokens=int(os.getenv('MAX_TOKENS', 100)))
    
    db_interaction = Interaction(id=uuid.uuid4().hex, prompt=request.messages[-1].content if request.messages else "", response=response_content)
    db.add(db_interaction)
    db.commit()
    db.refresh(db_interaction)
    
    return response_content, db_interaction