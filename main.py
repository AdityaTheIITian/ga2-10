import os
import re

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.environ.get("MODEL", "llama3.2")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ExtractIn(BaseModel):
    text: str


class Invoice(BaseModel):
    vendor: str = Field(min_length=1)
    amount: float
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


SYSTEM = (
    "Extract invoice fields from the text and reply with JSON only.\n"
    "- vendor: the company issuing the invoice (the party to be paid), exactly as written.\n"
    "- amount: the TOTAL amount due as a plain number, no currency symbols or thousands separators.\n"
    "- currency: 3-letter uppercase ISO code (map $ to USD, € to EUR, £ to GBP).\n"
    "- date: the payment DUE date as YYYY-MM-DD. Ignore issue or invoice dates."
)

SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
NUM_RE = re.compile(r"(?<![\d-])\d[\d,]*(?:\.\d+)?(?![\d]|-\d)")


def to_float(s: str):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def fallback(text: str) -> dict:
    out = {}
    m = re.search(r"due[^\d]{0,40}(\d{4}-\d{2}-\d{2})", text, re.I)
    dates = DATE_RE.findall(text)
    if m:
        out["date"] = m.group(1)
    elif dates:
        out["date"] = dates[-1]

    m = re.search(r"\b(USD|EUR|GBP)\b", text)
    if m:
        out["currency"] = m.group(1)
    else:
        for sym, code in SYMBOLS.items():
            if sym in text:
                out["currency"] = code
                break

    m = re.search(r"(?:total|amount due|balance due|due)[^\d\n]{0,30}(\d[\d,]*(?:\.\d+)?)(?!\d|-\d)", text, re.I)
    if m and to_float(m.group(1)) is not None:
        out["amount"] = to_float(m.group(1))
    else:
        nums = [to_float(n) for n in NUM_RE.findall(text)]
        nums = [n for n in nums if n is not None]
        if nums:
            out["amount"] = max(nums)

    m = re.search(
        r"([A-Z][\w&.\-]*(?:\s+[A-Za-z][\w&.\-]*){0,3}?\s+(?:Industries|Ltd\.?|Inc\.?|LLC|Corp\.?|GmbH|Co\.?)(?:\s+Ltd\.?)?)",
        text,
    )
    if m:
        out["vendor"] = m.group(1).strip()
    return out


async def ask_llm(text: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
        "format": Invoice.model_json_schema(),
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0},
    }
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        return Invoice.model_validate_json(r.json()["message"]["content"]).model_dump()


def reconcile(llm: dict, text: str) -> dict:
    """Keep each LLM value only if the text supports it; otherwise use the regex fallback."""
    fb = fallback(text)
    out = dict(llm)

    if out.get("date") not in DATE_RE.findall(text):
        out["date"] = fb.get("date", out.get("date"))

    nums = [to_float(n) for n in NUM_RE.findall(text)]
    if not any(n is not None and abs(n - out.get("amount", -1)) <= 0.01 for n in nums):
        out["amount"] = fb.get("amount", out.get("amount"))

    if out.get("vendor", "").lower() not in text.lower():
        out["vendor"] = fb.get("vendor", out.get("vendor"))

    explicit = re.search(r"\b(USD|EUR|GBP)\b", text)
    if explicit:
        out["currency"] = explicit.group(1)
    elif "currency" not in out and "currency" in fb:
        out["currency"] = fb["currency"]
    return out


@app.post("/extract", response_model=Invoice)
async def extract(body: ExtractIn):
    text = body.text.strip()
    if not text:
        return JSONResponse(status_code=422, content={"detail": "text is empty"})

    try:
        llm = await ask_llm(text)
    except Exception:
        llm = {}  # Ollama down, timeout, or unparseable output: fall back to regex only

    merged = reconcile(llm, text) if llm else fallback(text)
    try:
        return Invoice(**merged)
    except (ValidationError, TypeError):
        return JSONResponse(status_code=422, content={"detail": "could not extract invoice fields"})
