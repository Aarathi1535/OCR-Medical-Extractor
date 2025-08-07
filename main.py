import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from azure_ocr import extract_text_from_image
from gemini_diagram import analyze_diagram
from PIL import Image
import io

st.title("📝 Medical Answer Sheet Evaluator")

uploaded_file = st.file_uploader("Upload answer sheet PDF", type=["pdf"])
if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for i, page in enumerate(doc):
        st.subheader(f"Page {i + 1}")
        
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Show page image
        st.image(img, caption="Answer Sheet Image")

        # Convert PIL image to bytes (JPEG)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        byte_img = buffer.getvalue()

        with st.spinner("Extracting handwritten text..."):
            text = extract_text_from_image(byte_img)
            st.text_area("🧠 Extracted Text", value=text, height=200)

        with st.spinner("Analyzing diagram..."):
            diagram_info = analyze_diagram(img)
            st.text_area("📊 Diagram Description", value=diagram_info, height=200)

        st.markdown("---")
