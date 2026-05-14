import io
import json
import logging
import re
from typing import List, Optional

import fitz  # PyMuPDF
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

MAX_PDF_CHARS = 400_000
CONFIDENCE_THRESHOLD = 80
PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-5.4"


# ---------------------------------------------------------------------------
# Pydantic schema — enforces structure on every model response
# ---------------------------------------------------------------------------

class Buyer(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    contact_person: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    tax_id: Optional[str] = None


class Supplier(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    address: Optional[str] = None
    contact_person: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    tax_id: Optional[str] = None


class LineItem(BaseModel):
    item_no: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    delivery_date: Optional[str] = None
    notes: Optional[str] = None


class PaymentTerms(BaseModel):
    billing_cycle: Optional[str] = None
    payment_due_date: Optional[str] = None
    payment_method: Optional[str] = None


class PurchaseOrder(BaseModel):
    po_number: Optional[str] = None
    po_date: Optional[str] = None
    delivery_date: Optional[str] = None
    status: Optional[str] = None
    currency: Optional[str] = None
    buyer: Optional[Buyer] = None
    supplier: Optional[Supplier] = None
    line_items: List[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax_rate: Optional[float] = None
    tax_amount: Optional[float] = None
    shipping_cost: Optional[float] = None
    discount: Optional[float] = None
    grand_total: Optional[float] = None
    payment_terms: Optional[PaymentTerms] = None
    delivery_terms: Optional[str] = None
    shipping_address: Optional[str] = None
    notes: Optional[str] = None
    confidence_score: Optional[int] = None


# ---------------------------------------------------------------------------
# PDF reading — native text layer with OCR fallback for scanned PDFs
# ---------------------------------------------------------------------------

def _extract_text_native(doc: fitz.Document) -> list[str]:
    pages_text = []
    for i, page in enumerate(doc):
        try:
            text = page.get_text()
            if text.strip():
                pages_text.append(f"--- Page {i + 1} ---\n{text.strip()}")
        except Exception as e:
            logger.warning(f"Native text extraction failed for page {i + 1}: {e}")
    return pages_text


def _extract_text_ocr(doc: fitz.Document) -> list[str]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.error(
            "OCR fallback requires pytesseract and Pillow — "
            "install them to support scanned PDFs"
        )
        return []

    pages_text = []
    for i, page in enumerate(doc):
        try:
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR accuracy
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="jpn+eng")
            if text.strip():
                pages_text.append(f"--- Page {i + 1} [OCR] ---\n{text.strip()}")
            else:
                logger.warning(f"OCR returned no text for page {i + 1}")
        except Exception as e:
            logger.warning(f"OCR failed for page {i + 1}: {e}")
    return pages_text


def read_pdf_text(pdf_path: str) -> str:
    logger.info(f"Reading PDF: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        pages_text = _extract_text_native(doc)

        if not pages_text:
            logger.warning("No native text found — attempting OCR fallback")
            pages_text = _extract_text_ocr(doc)
            if not pages_text:
                logger.error(
                    "OCR fallback also returned no text. "
                    "Ensure Tesseract is installed (brew install tesseract) "
                    "and pytesseract/Pillow are in your Python environment."
                )

    if not pages_text:
        raise ValueError(
            f"No text could be extracted from: {pdf_path} (tried native + OCR)"
        )

    full_text = "\n\n".join(pages_text)

    if len(full_text) > MAX_PDF_CHARS:
        logger.warning(
            f"PDF text is {len(full_text)} chars — truncating to {MAX_PDF_CHARS}"
        )
        full_text = full_text[:MAX_PDF_CHARS]

    logger.info(f"Extracted {len(full_text)} chars from {len(pages_text)} pages")
    return full_text


# ---------------------------------------------------------------------------
# Prompts — built via concatenation to avoid f-string injection from PDF text
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Purchase Order data extraction assistant. "
    "Return ONLY valid JSON with no markdown, no code blocks, and no explanation."
)

_SCHEMA = """{
  "po_number": "string or null",
  "po_date": "string or null",
  "delivery_date": "string or null",
  "status": "string or null",
  "currency": "string or null",
  "buyer": {
    "name": "string or null",
    "address": "string or null",
    "contact_person": "string or null",
    "phone": "string or null",
    "email": "string or null",
    "tax_id": "string or null"
  },
  "supplier": {
    "id": "string or null",
    "name": "string or null",
    "address": "string or null",
    "contact_person": "string or null",
    "phone": "string or null",
    "email": "string or null",
    "tax_id": "string or null"
  },
  "line_items": [
    {
      "item_no": "string or null",
      "description": "string or null",
      "quantity": "number or null",
      "unit": "string or null",
      "unit_price": "number or null",
      "total_price": "number or null",
      "delivery_date": "string or null",
      "notes": "string or null"
    }
  ],
  "subtotal": "number or null",
  "tax_rate": "number or null",
  "tax_amount": "number or null",
  "shipping_cost": "number or null",
  "discount": "number or null",
  "grand_total": "number or null",
  "payment_terms": {
    "billing_cycle": "string or null",
    "payment_due_date": "string or null",
    "payment_method": "string or null"
  },
  "delivery_terms": "string or null",
  "shipping_address": "string or null",
  "notes": "string or null",
  "confidence_score": "integer between 0 and 100 — your confidence the extracted data is accurate and complete"
}"""

_INSTRUCTIONS = (
    "Extract information from the Purchase Order document below and return it as a STRICT JSON object.\n\n"
    "IMPORTANT RULES:\n"
    "- Return ONLY valid JSON. No markdown, no code blocks, no explanation.\n"
    "- Always use exactly these field names — do not rename or add extra fields.\n"
    "- If a field is not found in the document, use null for strings/numbers and [] for arrays.\n"
    "- For all amounts and prices, return numbers only (no currency symbols).\n"
    "- For dates, return in YYYY-MM-DD format if possible, otherwise return as-is.\n\n"
    "Required JSON structure:\n"
)

_FALLBACK_PREFIX = (
    "The previous extraction attempt scored below the confidence threshold. "
    "You are a higher-capability model being used to produce a more accurate result. "
    "Apply extra care to line items, totals, and tax fields.\n\n"
)


def _build_prompt(pdf_text: str, is_fallback: bool = False) -> str:
    prefix = _FALLBACK_PREFIX if is_fallback else ""
    return prefix + _INSTRUCTIONS + _SCHEMA + "\n\nDocument text:\n" + pdf_text


# ---------------------------------------------------------------------------
# Model calling — retries on transient errors, validates schema on every call
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_model(model: str, prompt: str) -> dict:
    logger.info(f"Calling model={model}...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    logger.debug(f"Raw response from {model}: {raw[:200]}...")

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"{model} returned invalid JSON: {e}\nRaw snippet: {raw[:500]}")
        raise ValueError(f"{model} returned invalid JSON: {e}")

    try:
        validated = PurchaseOrder(**data)
        logger.info(f"Schema validation passed for {model}")
        return validated.model_dump()
    except ValidationError as e:
        logger.error(f"{model} response failed schema validation: {e}")
        raise ValueError(f"{model} response failed schema validation: {e}")


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------

def extract_purchase_order(pdf_text: str) -> dict:
    logger.info("Starting Purchase Order extraction...")

    extracted = _call_model(PRIMARY_MODEL, _build_prompt(pdf_text, is_fallback=False))
    confidence = extracted.get("confidence_score")

    if confidence is None or confidence < CONFIDENCE_THRESHOLD:
        reason = (
            "missing confidence_score"
            if confidence is None
            else f"confidence_score={confidence} below threshold={CONFIDENCE_THRESHOLD}"
        )
        logger.warning(
            f"{PRIMARY_MODEL} result insufficient ({reason}) — escalating to {FALLBACK_MODEL}"
        )
        extracted = _call_model(FALLBACK_MODEL, _build_prompt(pdf_text, is_fallback=True))
    else:
        logger.info(
            f"{PRIMARY_MODEL} confidence_score={confidence} — above threshold, using this result"
        )

    return extracted


def process_pdf(pdf_path: str) -> dict:
    if not pdf_path:
        raise ValueError("pdf_path is empty")

    pdf_text = read_pdf_text(pdf_path)
    extracted_data = extract_purchase_order(pdf_text)

    return {
        "documentType": "PURCHASE_ORDER",
        "extractedData": extracted_data,
    }
