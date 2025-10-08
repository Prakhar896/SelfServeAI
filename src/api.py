from fastapi import FastAPI
from src.chatCompletion.controller import router as chat_router
from src.middleware import AuthMiddleware
from fastapi.middleware.cors import CORSMiddleware

origins = ['*']

def add_middlewares(app: FastAPI):
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(AuthMiddleware)

def register_routes(app: FastAPI):
    app.include_router(chat_router)