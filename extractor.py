import json
import logging
import re

import fitz  # PyMuPDF
from openai import OpenAI

from config import OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

MAX_PDF_CHARS = 400_000


def read_pdf_text(pdf_path: str) -> str:
    """
    Extract raw text from a PDF using PyMuPDF.
    Reads all pages and joins them with a separator.
    """
    logger.info(f"Reading PDF text from: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        pages_text = []

        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages_text.append(f"--- Page {i + 1} ---\n{text.strip()}")

    if not pages_text:
        raise ValueError(f"No text could be extracted from: {pdf_path}")

    full_text = "\n\n".join(pages_text)

    if len(full_text) > MAX_PDF_CHARS:
        logger.warning(
            f"PDF text is {len(full_text)} chars — truncating to {MAX_PDF_CHARS} "
            f"to stay within model context limits"
        )
        full_text = full_text[:MAX_PDF_CHARS]

    logger.info(f"Extracted {len(full_text)} characters from {len(pages_text)} pages")
    return full_text


SYSTEM_PROMPT = "You are a Purchase Order data extraction assistant. Return ONLY valid JSON with no markdown, no code blocks, and no explanation."

PURCHASE_ORDER_PROMPT = """Extract information from the Purchase Order document below and return it as a STRICT JSON object.

IMPORTANT RULES:
- Return ONLY valid JSON. No markdown, no code blocks, no explanation.
- Always use exactly these field names — do not rename or add extra fields.
- If a field is not found in the document, use null for strings/numbers and [] for arrays.
- For all amounts and prices, return numbers only (no currency symbols).
- For dates, return in YYYY-MM-DD format if possible, otherwise return as-is from the document.

Required JSON structure:
{{
  "po_number": "string or null",
  "po_date": "string or null",
  "delivery_date": "string or null",
  "status": "string or null",
  "currency": "string or null",

  "buyer": {{
    "name": "string or null",
    "address": "string or null",
    "contact_person": "string or null",
    "phone": "string or null",
    "email": "string or null",
    "tax_id": "string or null"
  }},

  "supplier": {{
    "id": "string or null",
    "name": "string or null",
    "address": "string or null",
    "contact_person": "string or null",
    "phone": "string or null",
    "email": "string or null",
    "tax_id": "string or null"
  }},

  "line_items": [
    {{
      "item_no": "string or null",
      "description": "string or null",
      "quantity": "number or null",
      "unit": "string or null",
      "unit_price": "number or null",
      "total_price": "number or null",
      "delivery_date": "string or null",
      "notes": "string or null"
    }}
  ],

  "subtotal": "number or null",
  "tax_rate": "number or null",
  "tax_amount": "number or null",
  "shipping_cost": "number or null",
  "discount": "number or null",
  "grand_total": "number or null",

  "payment_terms": {{
    "billing_cycle": "string or null",
    "payment_due_date": "string or null",
    "payment_method": "string or null"
  }},

  "delivery_terms": "string or null",
  "shipping_address": "string or null",
  "notes": "string or null",

  "confidence_score": "integer between 0 and 100 — your confidence that the extracted data is accurate and complete"
}}

Document text:
{pdf_text}"""


CONFIDENCE_THRESHOLD = 80
PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-5.4"


def _call_model(model: str, prompt: str) -> dict:
    """Call a single OpenAI model and return parsed JSON."""
    logger.info(f"Calling model={model}...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    logger.debug(f"Raw response from {model}: {raw[:200]}...")

    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()

    try:
        extracted = json.loads(raw)
        logger.info(f"Extraction successful from {model}, keys: {list(extracted.keys())}")
        return extracted
    except json.JSONDecodeError as e:
        logger.error(f"{model} returned invalid JSON: {e}")
        logger.error(f"Raw response was: {raw[:500]}")
        raise ValueError(f"{model} returned invalid JSON: {e}")


def extract_purchase_order(pdf_text: str) -> dict:
    """
    Send extracted PDF text to OpenAI with a strict Purchase Order schema.
    Uses gpt-4o-mini first; if confidence_score < 80, falls back to gpt-5.4.
    Always returns the predefined structure.
    """
    prompt = PURCHASE_ORDER_PROMPT.format(pdf_text=pdf_text)

    logger.info("Sending text to OpenAI for Purchase Order extraction...")

    extracted = _call_model(PRIMARY_MODEL, prompt)
    confidence = extracted.get("confidence_score")

    if confidence is None:
        logger.warning(f"{PRIMARY_MODEL} did not return a confidence_score — assuming low confidence, falling back to {FALLBACK_MODEL}")
        extracted = _call_model(FALLBACK_MODEL, prompt)
    elif confidence < CONFIDENCE_THRESHOLD:
        logger.warning(
            f"{PRIMARY_MODEL} confidence_score={confidence} is below threshold={CONFIDENCE_THRESHOLD}. "
            f"Falling back to {FALLBACK_MODEL}..."
        )
        extracted = _call_model(FALLBACK_MODEL, prompt)
    else:
        logger.info(f"{PRIMARY_MODEL} confidence_score={confidence} — above threshold, using this result.")

    return extracted


def process_pdf(pdf_path: str) -> dict:
    """
    Full pipeline:
    1. Read text from PDF using PyMuPDF
    2. Send text to OpenAI for structured PO extraction
    Returns dict with documentType + extractedData ready for Kafka
    """
    if not pdf_path:
        raise ValueError("pdf_path is empty")

    pdf_text = read_pdf_text(pdf_path)
    extracted_data = extract_purchase_order(pdf_text)

    return {
        "documentType": "PURCHASE_ORDER",
        "extractedData": extracted_data
    }
