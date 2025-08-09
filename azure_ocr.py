import os
import base64
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=MISTRAL_API_KEY)

def extract_text_from_image_bytes(image_bytes):
    """
    Use Mistral OCR to extract text from raw image bytes (e.g., JPEG).
    """
    b64 = base64.b64encode(image_bytes).decode()
    document = {
        "type": "document_url",
        "document_url": f"data:image/jpeg;base64,{b64}"
    }

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document=document,
        include_image_base64=False
    )

    # The OCR response includes pages structured in markdown or text
    # You can adjust based on actual response format
    texts = []
    if hasattr(ocr_response, "pages"):
        # Example: iterates pages and collects text
        for page in ocr_response.pages:
            texts.append(page.get("text", ""))
    elif isinstance(ocr_response, dict) and "text" in ocr_response:
        texts.append(ocr_response["text"])
    else:
        texts.append(str(ocr_response))

    return "\n\n".join(texts)
