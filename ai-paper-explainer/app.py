import streamlit as st
from pypdf import PdfReader
from transformers import pipeline

st.title("AI Research Paper Explainer")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    st.subheader("Extracted Text")
    st.write(text[:1000])

    if st.button("Generate Summary"):

        generator = pipeline("text-generation", model="gpt2")

        prompt = "Summarize this research paper:\n" + text[:800]

        result = generator(prompt, max_length=200, num_return_sequences=1)

        st.subheader("AI Summary")
        st.write(result[0]['generated_text'])