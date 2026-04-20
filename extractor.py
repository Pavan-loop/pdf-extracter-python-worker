import json
import logging
import re

import fitz  # PyMuPDF
from openai import OpenAI

from config import OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)


def read_pdf_text(pdf_path: str) -> str:
    """
    Extract raw text from a PDF using PyMuPDF.
    Reads all pages and joins them with a separator.
    """
    logger.info(f"Reading PDF text from: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages_text = []

    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages_text.append(f"--- Page {i + 1} ---\n{text.strip()}")

    doc.close()

    if not pages_text:
        raise ValueError(f"No text could be extracted from: {pdf_path}")

    full_text = "\n\n".join(pages_text)
    logger.info(f"Extracted {len(full_text)} characters from {len(pages_text)} pages")
    return full_text


PURCHASE_ORDER_PROMPT = """You are a Purchase Order data extraction assistant.

Extract information from the Purchase Order document below and return it as a STRICT JSON object.

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
  "notes": "string or null"
}}

Document text:
{pdf_text}"""


def extract_purchase_order(pdf_text: str) -> dict:
    """
    Send extracted PDF text to OpenAI with a strict Purchase Order schema.
    Always returns the predefined structure.
    """
    prompt = PURCHASE_ORDER_PROMPT.format(pdf_text=pdf_text)

    logger.info("Sending text to OpenAI for Purchase Order extraction...")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    logger.debug(f"Raw OpenAI response: {raw[:200]}...")

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    try:
        extracted = json.loads(raw)
        logger.info(f"Extraction successful, keys: {list(extracted.keys())}")
        return extracted
    except json.JSONDecodeError as e:
        logger.error(f"OpenAI returned invalid JSON: {e}")
        logger.error(f"Raw response was: {raw[:500]}")
        raise ValueError(f"OpenAI returned invalid JSON: {e}")


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