import os, uuid
from ..entities import Interaction
from .model import ChatCompletionRequest, ChatMessage
from sqlalchemy.orm import Session
from ..slm import SmolLM
from ..ai import LLMInterface, LMProvider, LMVariant, InteractionContext
from ..ai import Interaction as LMInteraction

def obtainResponse(request: ChatCompletionRequest, db: Session):
    response_content = None
    
    mode = os.getenv('RESPONSE_MODE', 'mock')

    if mode == 'mock':
        if request.messages:
            response_content = "As a mock AI assistant, I received your message: " + request.messages[-1].content
        else:
            response_content = 'As a mock AI assistant, I can only echo messages. No previous messages found.'
    elif mode == 'slm':
        response_content = SmolLM.infer(request.messages[-1].content if request.messages else "No input from user.", max_new_tokens=int(os.getenv('MAX_TOKENS', 100)))
    elif mode == 'openai':
        cont = InteractionContext(
            provider=LMProvider.OPENAI,
            variant=LMVariant.GPT_5_MINI
        )
        cont.addInteraction(
            LMInteraction(
                role=LMInteraction.Role.USER,
                content=request.messages[-1].content if request.messages else "No input from user."
            )
        )
        response = LLMInterface.engage(cont)
        
        if isinstance(response, str):
            print('OpenAI inference error:', response)
            response_content = 'Error occurred.'
        else:
            response_content = response.content
    
    db_interaction = Interaction(id=uuid.uuid4().hex, prompt=request.messages[-1].content if request.messages else "Empty", response=response_content)
    db.add(db_interaction)
    db.commit()
    db.refresh(db_interaction)
    
    return response_content, db_interaction