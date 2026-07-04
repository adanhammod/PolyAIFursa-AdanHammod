import base64
import io
import json
import logging
import os
import uuid
from contextvars import ContextVar
from typing import Optional

import boto3

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langchain_core").setLevel(logging.DEBUG)

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from pydantic import BaseModel

YOLO_SERVICE_URL = os.environ.get("YOLO_SERVICE_URL", "http://localhost:8080")
MODEL = os.environ.get("MODEL")

# Text-only models
ALLOWED_MODELS = {
<<<<<<< HEAD
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/amazon.nova-micro-v1:0",
    "bedrock/amazon.nova-lite-v1:0",
    "bedrock/openai.gpt-oss-20b-1:0",
    "bedrock/meta.llama3-1-8b-instruct-v1:0",
    "bedrock/mistral.mistral-7b-instruct-v0:2",
    "bedrock_converse:openai.gpt-oss-20b-1:0",
=======
    "openai:gpt-5.4-mini",
    "anthropic:claude-haiku-4-5",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/amazon.nova-micro-v1:0",
    "bedrock/amazon.nova-lite-v1:0",
>>>>>>> main
}

if MODEL not in ALLOWED_MODELS:
    allowed_list = "\n  ".join(sorted(ALLOWED_MODELS))
    raise SystemExit(
        f"\n[ERROR] MODEL='{MODEL}' is not allowed.\n"
        f"Set MODEL in your .env to one of the supported text-only models:\n  {allowed_list}\n"
    )

SYSTEM_PROMPT = (
    "You are an AI vision assistant. You help users understand and analyze images. "
    "Use the available tools to extract information from images. "
)

<<<<<<< HEAD
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
s3_client = boto3.client("s3", region_name=AWS_REGION)

_current_image_b64: ContextVar[Optional[str]] = ContextVar(
    "current_image_b64", default=None
)
_annotated_image_s3_key: ContextVar[Optional[str]] = ContextVar(
    "annotated_image_s3_key", default=None
=======
_current_image_b64: ContextVar[Optional[str]] = ContextVar(
    "current_image_b64", default=None
)
_annotated_image_url: ContextVar[Optional[str]] = ContextVar(
    "annotated_image_url", default=None
>>>>>>> main
)


@tool
def detect_objects() -> str:
    """Detect and identify objects in the image provided by the user using YOLO object detection."""
    image_b64 = _current_image_b64.get()
    if not image_b64:
        return json.dumps({"error": "No image was provided by the user."})

    image_bytes = base64.b64decode(image_b64)
    original_key = f"originals/{uuid.uuid4()}.jpg"
    s3_client.upload_fileobj(io.BytesIO(image_bytes), AWS_S3_BUCKET, original_key)

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{YOLO_SERVICE_URL}/predict",
            json={"image_s3_key": original_key},
        )
        response.raise_for_status()

    result = response.json()
    annotated_key = result.get("annotated_image_s3_key")
    if annotated_key:
        _annotated_image_s3_key.set(annotated_key)

    return json.dumps(result)


# Registry: map tool name -> tool function
TOOLS = {detect_objects.name: detect_objects}

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

llm = init_chat_model(
    MODEL,
    temperature=0,
    rate_limiter=rate_limiter,
    region_name=AWS_REGION,
)

if not llm.profile.get("tool_calling"):
    raise SystemExit(
        f"[ERROR] Model '{MODEL}' does not support tool calling, "
        "which is required by this agent."
    )

llm_with_tools = llm.bind_tools(list(TOOLS.values()))


def run_agent(history: list, max_iterations: int = 10) -> tuple[str, dict]:
    """
    Simple ReAct loop:
      1. Send messages to the LLM.
      2. If the LLM requests tool calls, execute them and append results.
      3. Repeat until the LLM returns a plain text response.
    Returns (response_text, token_counts) where token_counts has "input", "output", "total" keys.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history
    tokens = {"input": 0, "output": 0, "total": 0}
    iterations = 0

    while True:
        iterations += 1

        if iterations > max_iterations:
            return (
                "Error: Agent exceeded maximum iterations without producing a final answer.",
                tokens,
            )

        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        meta = response.usage_metadata or {}
        tokens["input"] += meta.get("input_tokens", 0)
        tokens["output"] += meta.get("output_tokens", 0)
        tokens["total"] += meta.get("total_tokens", 0)

        # No tool calls, the model produced its final answer
        if not response.tool_calls:
            return response.content, tokens

        # Execute every tool the model requested
        for tool_call in response.tool_calls:
            tool_fn = TOOLS[tool_call["name"]]
            tool_result = tool_fn.invoke(tool_call)  # returns a ToolMessage
            messages.append(tool_result)

            # LangChain invokes tools in a copied context, so ContextVar.set() inside
            # the tool is invisible to chat(). Extract the key from the ToolMessage here,
            # where we share the same context as the caller.
            try:
                payload = json.loads(tool_result.content)
                annotated_key = payload.get("annotated_image_s3_key")
                if annotated_key:
                    _annotated_image_s3_key.set(annotated_key)
            except Exception:
                logging.exception(
                    "Failed to parse tool result for annotated_image_s3_key"
                )


app = FastAPI(title="Vision Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://prod.adan.fursa.click:3000",
        "http://adan-dev.fursa.click:3000",
        "http://localhost:3000",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    image_base64: Optional[str] = None  # only on user messages that carry an image


class ChatRequest(BaseModel):
    messages: list[ChatMessage]  # full conversation thread, oldest first


class ChatResponse(BaseModel):
    response: str
    annotated_image_base64: Optional[str] = None
    tokens_used: dict  # {"input": int, "output": int, "total": int}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    lc_messages = []
    latest_image = None

    for msg in request.messages:
        if msg.role == "user":
            if msg.image_base64:
                latest_image = msg.image_base64  # saved for detect_objects tool
                content = (
                    msg.content
                    + "\n[An image was uploaded. Use existing tools to analyze it according to user instructions.]"
                )
            else:
                content = msg.content
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=msg.content))

    token_img = _current_image_b64.set(latest_image)
    token_key = _annotated_image_s3_key.set(None)
    try:
        response_text, tokens_used = run_agent(lc_messages)
        annotated_image_b64 = None

        annotated_key = _annotated_image_s3_key.get()
        logging.info("Annotated image S3 key: %s", annotated_key)

        response_text = "\n".join(
            line
            for line in response_text.splitlines()
            if "Annotated image:" not in line
            and "http://localhost:8080/prediction/" not in line
            and (not annotated_key or annotated_key not in line)
        ).strip()

        if annotated_key:
            try:
                buf = io.BytesIO()
                s3_client.download_fileobj(AWS_S3_BUCKET, annotated_key, buf)
                annotated_image_b64 = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                logging.exception("Failed to fetch annotated image from S3")

        return ChatResponse(
            response=response_text,
            annotated_image_base64=annotated_image_b64,
            tokens_used=tokens_used,
        )
    finally:
        _current_image_b64.reset(token_img)
        _annotated_image_s3_key.reset(token_key)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
