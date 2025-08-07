import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import base64
from io import BytesIO

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

def analyze_diagram(image_pil):
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    response = model.generate_content(
        [
            "Describe any diagrams or medical visuals in this image. Do not assume missing parts. Return only what is clearly visible.",
            {"mime_type": "image/jpeg", "data": image_bytes}
        ]
    )
    return response.text
