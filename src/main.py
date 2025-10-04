import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from .slm import SmolLM
from .database.core import Base, engine
from .api import add_middlewares, register_routes
from .logging import configure_logging, LogLevels

configure_logging(LogLevels.info)

app = FastAPI(title='SelfServeAI')
Base.metadata.create_all(bind=engine)
if os.getenv('RESPONSE_MODE', 'mock') == 'slm':
    SmolLM.load()

add_middlewares(app)
register_routes(app)

