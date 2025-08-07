import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")

def extract_text_from_image(image_data):
    ocr_url = f"{AZURE_ENDPOINT}/vision/v3.2/read/analyze"
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_KEY,
        'Content-Type': 'application/octet-stream'
    }

    response = requests.post(ocr_url, headers=headers, data=image_data)
    response.raise_for_status()
    operation_url = response.headers["Operation-Location"]

    # Poll for result
    while True:
        result = requests.get(operation_url, headers={'Ocp-Apim-Subscription-Key': AZURE_KEY}).json()
        if "status" in result and result["status"] in ["succeeded", "failed"]:
            break
        time.sleep(1)

    extracted_text = ""
    if result["status"] == "succeeded":
        for read_result in result["analyzeResult"]["readResults"]:
            for line in read_result["lines"]:
                extracted_text += line["text"] + "\n"
    return extracted_text
