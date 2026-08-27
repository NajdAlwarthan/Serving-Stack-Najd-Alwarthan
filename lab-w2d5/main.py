import os
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

app = FastAPI()
security = HTTPBearer(auto_error=False)

API_KEY = os.getenv("API_KEY", "")
MAX_TOKENS_LIMIT = int(os.getenv("MAX_TOKENS", "256"))
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if API_KEY:
        if not credentials or credentials.credentials != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/v1/models", dependencies=[Depends(verify_token)])
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

class ChatRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 100

@app.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
def chat_completions(req: ChatRequest):
    tokens_to_generate = min(req.max_tokens, MAX_TOKENS_LIMIT)
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {"completion_tokens": tokens_to_generate}
    }
