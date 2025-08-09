import os
import base64
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

def extract_text_from_image(image_data: bytes) -> str:
    # Encode image to base64
    b64_image = base64.b64encode(image_data).decode()

    # Correct Mistral OCR payload
    document = {
        "type": "base64",
        "base64": b64_image
    }

    # Call Mistral OCR
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document=document,
        include_image_base64=False
    )

    extracted_text = ""
    if hasattr(ocr_response, "pages"):
        for page in ocr_response.pages:
            extracted_text += page.get("text", "") + "\n"
    else:
        extracted_text = str(ocr_response)

    return extracted_text.strip()
