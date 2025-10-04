from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response, HTTPException

class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.auth_token = 'fake-api-key'
    
    async def dispatch(self, request, call_next):
        if not request.url.path.startswith('/chat'):
            return await call_next(request)
        
        api_key = request.headers.get('authorization')
        
        if not api_key:
            return Response(content='Invalid API key.', status_code=401)    
                
        if api_key.split(' ')[1] != self.auth_token:
            return Response(content='Invalid API key.', status_code=403)

        return await call_next(request)