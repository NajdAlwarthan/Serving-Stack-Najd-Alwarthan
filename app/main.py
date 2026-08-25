"""serving-stack: the FastAPI service (week 2, CPU, tiny model)."""
from __future__ import annotations

import os
import time
import uuid

import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    HealthResponse,
    ModelCard,
    ModelList,
    ResponseMessage,
    Usage,
)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

app = FastAPI(title="serving-stack", version="wk2")

# Load once at import time. CPU only this week.
print(f"loading {MODEL_ID} on cpu ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.to("cpu")
model.eval()
print("model ready")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness and readiness."""
    return HealthResponse(status="ok", model=MODEL_ID)


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------
@app.get("/v1/models", response_model=ModelList)
def list_models() -> ModelList:
    """List the served model id(s)."""
    card = ModelCard(
        id=MODEL_ID,
        object="model",
        created=int(time.time()),
        owned_by="owner",
    )
    return ModelList(object="list", data=[card])


# ---------------------------------------------------------------------------
# POST /v1/chat/completions (non-streaming)
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """Run the model over the messages and return an OpenAI-compatible completion."""
    if req.model != MODEL_ID:
        raise HTTPException(status_code=400, detail=f"Model '{req.model}' not found. Serving '{MODEL_ID}'.")

    # 1. Build prompt with explicit dictionary return structure
    messages_dict = [m.model_dump() for m in req.messages]
    encoded = tokenizer.apply_chat_template(
        messages_dict,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"].to("cpu")
    prompt_tokens = input_ids.shape[1]

    # 2. Generate model output
    do_sample = req.temperature is not None and req.temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": req.max_tokens or 128,
        "do_sample": do_sample,
    }
    if do_sample and req.temperature:
        gen_kwargs["temperature"] = req.temperature

    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)

    # 3. Decode completion tokens
    new_tokens = out[0][prompt_tokens:]
    completion_tokens = len(new_tokens)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # 4. Finish reason
    finish_reason = "length" if completion_tokens >= (req.max_tokens or 128) else "stop"

    # 5. Assemble response object
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )