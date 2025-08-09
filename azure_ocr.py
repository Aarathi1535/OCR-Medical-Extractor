import os
import base64
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

def extract_text_from_image(image_data: bytes) -> str:
    # Convert image bytes to base64
    b64_image = base64.b64encode(image_data).decode()

    # Correct payload for Mistral OCR
    document = {
        "type": "document_base64",
        "document_base64": b64_image
    }

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document=document,
        include_image_base64=False
    )

    extracted_text = ""
    if hasattr(ocr_response, "pages"):
        for page in ocr_response.pages:
            extracted_text += page.get("text", "") + "\n"
    elif isinstance(ocr_response, dict) and "text" in ocr_response:
        extracted_text = ocr_response["text"]
    else:
        extracted_text = str(ocr_response)

    return extracted_text.strip()
