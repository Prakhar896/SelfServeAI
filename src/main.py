import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from .slm import SmolLM
from .database.core import Base, engine
from .api import add_middlewares, register_routes
from .logging import configure_logging, LogLevels
from contextlib import asynccontextmanager

configure_logging(LogLevels.info)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv('RESPONSE_MODE', 'mock') == 'slm':
        SmolLM.load()
    
    yield
    
    SmolLM.model = None
    SmolLM.tokenizer = None

app = FastAPI(title='SelfServeAI', lifespan=lifespan)
Base.metadata.create_all(bind=engine)

add_middlewares(app)
register_routes(app)