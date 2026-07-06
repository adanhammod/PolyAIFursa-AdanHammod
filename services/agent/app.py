import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Optional

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
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import StructuredTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, create_model

YOLO_SERVICE_URL = os.environ.get("YOLO_SERVICE_URL", "http://localhost:8080")
IMG_PROC_MCP_URL = os.environ.get("IMG_PROC_MCP_URL", "http://127.0.0.1:9000")
MODEL = os.environ.get("MODEL")

# Text-only models
ALLOWED_MODELS = {
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/amazon.nova-micro-v1:0",
    "bedrock/amazon.nova-lite-v1:0",
    "bedrock/openai.gpt-oss-20b-1:0",
    "bedrock/meta.llama3-1-8b-instruct-v1:0",
    "bedrock/mistral.mistral-7b-instruct-v0:2",
    "bedrock_converse:openai.gpt-oss-20b-1:0",
}

if MODEL not in ALLOWED_MODELS:
    allowed_list = "\n  ".join(sorted(ALLOWED_MODELS))
    raise SystemExit(
        f"\n[ERROR] MODEL='{MODEL}' is not allowed.\n"
        f"Set MODEL in your .env to one of the supported text-only models:\n  {allowed_list}\n"
    )

SYSTEM_PROMPT = (
    "You are an AI vision assistant. Follow these tool-selection rules strictly:\n"
    "- Use 'rotate' when the user asks to rotate the image.\n"
    "- Use 'blur' when the user asks to blur the image.\n"
    "- Use 'flip' when the user asks to flip or mirror the image.\n"
    "- Use 'resize' when the user asks to resize the image.\n"
    "- Use 'crop' when the user asks to crop the image.\n"
    "- Use 'add_noise' when the user asks to add noise to the image.\n"
    "- Use 'detect_objects' ONLY when the user asks to analyze, detect, identify, "
    "count, or describe objects in the image.\n"
    "Do NOT call detect_objects for image-processing requests.\n\n"
    "Response format:\n"
    "- For image-processing tools (rotate, blur, flip, resize, crop, add_noise): "
    "respond with ONE short sentence describing the completed action. "
    "Example: 'Done! I rotated the image 90° clockwise.'\n"
    "- For detect_objects: summarize detected objects naturally in 1–2 sentences. "
    "Example: 'I found 5 people and 4 cars in the image.'\n"
    "- Include confidence scores or bounding boxes ONLY if the user explicitly requests them.\n"
    "- Do NOT generate: markdown image links, 'Annotated image' labels, "
    "'Rotated image' labels, placeholder captions, raw base64, or URLs.\n"
    "- The frontend displays the processed image automatically — do not mention it in text.\n"
    "- Keep responses under 2–3 sentences unless the user explicitly asks for detail."
)

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
s3_client = boto3.client("s3", region_name=AWS_REGION)

_current_image_b64: ContextVar[Optional[str]] = ContextVar(
    "current_image_b64", default=None
)
_annotated_image_s3_key: ContextVar[Optional[str]] = ContextVar(
    "annotated_image_s3_key", default=None
)
_processed_image_b64: ContextVar[Optional[str]] = ContextVar(
    "processed_image_b64", default=None
)


def _call_mcp(tool_name: str, arguments: dict) -> str:
    """Call a tool on the img-proc-mcp HTTP server synchronously."""
    async def _inner() -> str:
        async with streamable_http_client(f"{IMG_PROC_MCP_URL}/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return result.content[0].text

    return asyncio.run(_inner())


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


# Map JSON Schema primitive types to Python types used in create_model().
_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}

def _img_tag(b64: Optional[str]) -> str:
    """Return a safe, non-reversible log tag for an image (never logs raw base64)."""
    if not b64:
        return "none"
    tag = hashlib.sha256(b64[:256].encode()).hexdigest()[:8]
    return f"len={len(b64)} tag={tag}"


_HIDDEN_BLOCK_TYPES = {"reasoning_content", "reasoning", "thinking"}


def _extract_visible_text(content) -> str:
    """Return only the user-visible text from an LLM response content value.

    Some models (e.g. extended-thinking variants) include reasoning/thinking
    blocks alongside their user-facing text. We must never expose those blocks
    to the frontend — neither as raw dicts nor as stringified representations.

    Rules:
    - str  → returned as-is
    - list → only items with type == "text" contribute; reasoning/thinking
              blocks and unknown dict types are silently dropped
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in _HIDDEN_BLOCK_TYPES:
                    continue
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                # Unknown dict types: silently ignored
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


# Standalone image-label lines the LLM emits after MCP tool calls,
# e.g. "Rotated image", "Blurred image.". Stripped from response text
# when the actual image is being returned in annotated_image_base64.
_IMAGE_LABEL_RE = re.compile(
    r'^\s*(?:annotated|rotated?|blurred?|flipped?|resized?|cropped?|processed)\s+image[.!:]*\s*$',
    re.IGNORECASE,
)

# Bedrock toolUse.name must match ^[a-zA-Z][a-zA-Z0-9_]*$.
_INVALID_TOOL_CHAR = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_tool_name(name: str) -> str:
    """Replace characters disallowed by Bedrock's toolUse.name constraint."""
    safe = _INVALID_TOOL_CHAR.sub("_", name)
    if safe and not safe[0].isalpha():
        safe = "t_" + safe
    return safe


# Bedrock toolUse.name pattern for runtime validation of LLM-returned names.
_VALID_TC_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_VALID_TC_NAME_PREFIX_RE = re.compile(r"^[a-zA-Z0-9_-]+")


def _clean_tool_call_name(raw: str) -> str | None:
    """Return a Bedrock-valid name from a raw LLM-generated tool call name.

    Some models hallucinate garbage after the tool name, e.g.
    'rotate<|channel|>commentary'. Bedrock rejects assistant messages whose
    toolUse.name falls outside [a-zA-Z0-9_-]+ in subsequent conversation turns.

    Returns the leading valid segment, or None if no valid prefix exists.
    """
    if _VALID_TC_NAME_RE.match(raw):
        return raw
    m = _VALID_TC_NAME_PREFIX_RE.match(raw)
    if m:
        logging.warning("Malformed tool name %r — truncated to %r", raw, m.group())
        return m.group()
    logging.error("Unrecoverable tool name from LLM: %r — call dropped", raw)
    return None


def _build_image_proc_wrapper(mcp_tool) -> StructuredTool:
    """Build a sync LangChain tool from a discovered MCP tool.

    Strips `image_b64` from the LLM-visible schema and injects it from the
    request-scoped ContextVar at call time.

    args_schema can be either a raw JSON Schema dict (what langchain-mcp-adapters
    v0.3+ returns via tool.inputSchema) or a Pydantic model class.  Both are
    handled so the wrapper works regardless of adapter version.
    """
    tool_name = mcp_tool.name          # original MCP name — used for _call_mcp
    safe_name = _sanitize_tool_name(tool_name)
    if safe_name != tool_name:
        logging.info("Tool name sanitized for Bedrock: %r → %r", tool_name, safe_name)
    schema = mcp_tool.args_schema

    schema_fields: dict[str, Any] = {}
    if isinstance(schema, dict):
        # JSON Schema dict — current langchain-mcp-adapters behaviour:
        # convert_mcp_tool_to_langchain_tool() sets args_schema=tool.inputSchema
        # which is the raw MCP protocol dict, e.g.
        # {"type": "object", "properties": {"angle": {"type": "number"}}, "required": ["angle"]}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        for fname, prop in properties.items():
            if fname == "image_b64":
                continue
            py_type = _JSON_TYPE_MAP.get(prop.get("type", ""), Any)
            default = ... if fname in required else prop.get("default", None)
            schema_fields[fname] = (py_type, default)
    else:
        # Pydantic model class — other adapters or future versions
        for fname, fi in schema.model_fields.items():
            if fname == "image_b64":
                continue
            annotation = fi.annotation if fi.annotation is not None else Any
            default = fi.default if not fi.is_required() else ...
            schema_fields[fname] = (annotation, default)

    DynSchema = create_model(f"{tool_name}_Schema", **schema_fields)

    def _run(**kwargs):
        image_b64 = _current_image_b64.get()
        if not image_b64:
            return json.dumps({"error": "No image was provided by the user."})
        logging.info("MCP tool %r: input image %s", tool_name, _img_tag(image_b64))
        result_b64 = _call_mcp(tool_name, {"image_b64": image_b64, **kwargs})
        return json.dumps({"processed_image_b64": result_b64})

    return StructuredTool.from_function(
        name=safe_name,   # Bedrock-safe; _call_mcp still uses original tool_name
        description=mcp_tool.description or f"Apply {tool_name} to the user's image.",
        func=_run,
        args_schema=DynSchema,
    )


# Module-level defaults — overwritten by lifespan once MCP tools are discovered.
TOOLS: dict = {detect_objects.name: detect_objects}

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

llm_with_tools = llm.bind_tools([detect_objects])


async def _init_tools() -> None:
    """Discover MCP tools dynamically and rebind llm_with_tools."""
    global TOOLS, llm_with_tools
    client = MultiServerMCPClient(
        {
            "img-proc": {
                "url": f"{IMG_PROC_MCP_URL}/mcp",
                "transport": "streamable_http",
            }
        }
    )
    mcp_tools = await client.get_tools()
    logging.info("MCP raw tool names from get_tools(): %s", [t.name for t in mcp_tools])
    image_proc_tools = [_build_image_proc_wrapper(t) for t in mcp_tools]
    all_tools = [detect_objects] + image_proc_tools
    TOOLS = {t.name: t for t in all_tools}
    logging.info("Tools bound to LLM: %s", [t.name for t in all_tools])
    llm_with_tools = llm.bind_tools(all_tools)
    logging.info(
        "MCP tools discovered (sanitized names): %s",
        [t.name for t in image_proc_tools],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _init_tools()
    yield


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

        # Log raw tool_calls exactly as the LLM returned them.
        logging.info(
            "LLM tool_calls (raw): %s",
            [{"name": tc["name"], "id": tc.get("id")} for tc in response.tool_calls],
        )

        # Validate and sanitize tool call names BEFORE appending to history.
        # Root cause of Bedrock ValidationException: the LLM can emit garbage
        # after the tool name (e.g. "rotate<|channel|>commentary"). Bedrock
        # re-validates toolUse.name in every assistant message sent in later
        # turns, and rejects anything outside [a-zA-Z0-9_-]+.
        cleaned_calls: list[dict] = []
        needs_rebuild = False
        for tc in response.tool_calls:
            clean = _clean_tool_call_name(tc["name"])
            if clean is None:
                needs_rebuild = True          # no valid prefix — drop the call
            elif clean != tc["name"]:
                cleaned_calls.append({**tc, "name": clean})
                needs_rebuild = True
            else:
                cleaned_calls.append(tc)

        if needs_rebuild:
            response = AIMessage(
                content=response.content,
                tool_calls=cleaned_calls,
                usage_metadata=response.usage_metadata,
            )
            logging.info(
                "AIMessage rebuilt with sanitized names: %s",
                [tc["name"] for tc in cleaned_calls],
            )

        messages.append(response)

        meta = response.usage_metadata or {}
        tokens["input"] += meta.get("input_tokens", 0)
        tokens["output"] += meta.get("output_tokens", 0)
        tokens["total"] += meta.get("total_tokens", 0)

        # No tool calls → final answer (also covers the all-names-dropped case)
        if not response.tool_calls:
            return _extract_visible_text(response.content), tokens

        # Execute every tool the model requested
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]   # already validated/sanitized above
            safe_args = {k: v for k, v in tool_call["args"].items() if "b64" not in k and "base64" not in k}
            logging.info(
                "Tool call: name=%r args=%s id=%s",
                tool_name, safe_args, tool_call.get("id"),
            )

            if tool_name not in TOOLS:
                # Send an error ToolMessage to keep toolUse/toolResult balanced.
                logging.error("Unknown tool %r — available: %s", tool_name, sorted(TOOLS))
                messages.append(ToolMessage(
                    content=json.dumps({"error": f"Unknown tool '{tool_name}'. Use: {sorted(TOOLS)}"}),
                    tool_call_id=tool_call["id"],
                ))
                continue

            tool_fn = TOOLS[tool_name]
            try:
                tool_result = tool_fn.invoke(tool_call)  # returns a ToolMessage
            except Exception:
                logging.exception("Tool execution failed: %s", tool_name)
                raise
            logging.info(
                "Tool finished: %s | ToolMessage.tool_call_id=%s",
                tool_name, tool_result.tool_call_id,
            )

            # Extract side-effect data and strip image base64 before adding to
            # LLM context — raw base64 exceeds Bedrock's context length limit.
            # LangChain invokes tools in a copied context, so ContextVar.set()
            # inside the tool is invisible here; we read from the ToolMessage.
            sanitized_content = tool_result.content
            try:
                payload = json.loads(tool_result.content)
                logging.info("Tool result keys: %s", list(payload.keys()))
                annotated_key = payload.get("annotated_image_s3_key")
                if annotated_key:
                    _annotated_image_s3_key.set(annotated_key)
                processed_b64 = payload.get("processed_image_b64")
                logging.info("processed_image_b64 detected: %s", bool(processed_b64))
                if processed_b64:
                    _current_image_b64.set(processed_b64)
                    _processed_image_b64.set(processed_b64)
                    sanitized_content = json.dumps({"processed_image": True})
                    logging.info(
                        "Updated _current_image_b64 to processed result: %s", _img_tag(processed_b64)
                    )
            except Exception:
                logging.exception("Failed to parse tool result")

            tool_msg = ToolMessage(
                content=sanitized_content, tool_call_id=tool_result.tool_call_id
            )
            logging.info(
                "Appending ToolMessage: tool_call_id=%s keys=%s",
                tool_msg.tool_call_id,
                list(json.loads(sanitized_content).keys()) if sanitized_content.startswith("{") else "text",
            )
            messages.append(tool_msg)


app = FastAPI(title="Vision Agent", lifespan=lifespan)

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

    # Extract image only from the MOST RECENT user message.
    # Older user messages may still carry stale image_base64 from previous
    # requests; iterating forward and taking the last match would silently
    # fall back to a prior image whenever the newest message has no upload,
    # causing request N to process the wrong image.
    latest_image: Optional[str] = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            latest_image = msg.image_base64  # None if this message has no upload
            break

    user_img_positions = [
        i for i, m in enumerate(request.messages)
        if m.role == "user" and m.image_base64
    ]
    logging.info(
        "Request: %d messages; user messages with image at positions %s; latest image: %s",
        len(request.messages), user_img_positions, _img_tag(latest_image),
    )

    for msg in request.messages:
        if msg.role == "user":
            if msg.image_base64:
                marker = "[User uploaded an image.]"
                content = f"{marker}\n{msg.content}" if msg.content.strip() else marker
            else:
                content = msg.content
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=msg.content))

    logging.info("Setting _current_image_b64: %s", _img_tag(latest_image))
    token_img = _current_image_b64.set(latest_image)
    token_key = _annotated_image_s3_key.set(None)
    logging.info("Reset _processed_image_b64 to None for new request")
    token_proc = _processed_image_b64.set(None)
    try:
        response_text, tokens_used = run_agent(lc_messages)
        annotated_image_b64 = None

        annotated_key = _annotated_image_s3_key.get()
        processed_b64 = _processed_image_b64.get()
        logging.info("Annotated image S3 key: %s | processed_b64 present: %s",
                     annotated_key, bool(processed_b64))

        response_text = "\n".join(
            line
            for line in response_text.splitlines()
            if "Annotated image:" not in line
            and "http://localhost:8080/prediction/" not in line
            and (not annotated_key or annotated_key not in line)
            and (not processed_b64 or not _IMAGE_LABEL_RE.match(line))
        ).strip()

        if annotated_key:
            try:
                buf = io.BytesIO()
                s3_client.download_fileobj(AWS_S3_BUCKET, annotated_key, buf)
                annotated_image_b64 = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                logging.exception("Failed to fetch annotated image from S3")

        # MCP image-processing tools (rotate, blur, etc.) return base64 directly —
        # they do not write to S3, so annotated_key is None for these calls.
        if not annotated_image_b64 and processed_b64:
            annotated_image_b64 = processed_b64

        logging.info("annotated_image_base64 present: %s", annotated_image_b64 is not None)
        return ChatResponse(
            response=response_text,
            annotated_image_base64=annotated_image_b64,
            tokens_used=tokens_used,
        )
    finally:
        _current_image_b64.reset(token_img)
        _annotated_image_s3_key.reset(token_key)
        _processed_image_b64.reset(token_proc)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")
