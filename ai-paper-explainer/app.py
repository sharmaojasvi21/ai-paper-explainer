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

    st.markdown("---")

    # -------- Q&A FEATURE --------
    st.subheader("Ask Questions from the Paper")

    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if question:
            generator = pipeline("text-generation", model="gpt2")

            prompt = f"Answer the question based on the text:\n{text[:800]}\nQuestion: {question}"

            result = generator(prompt, max_length=200, num_return_sequences=1)

            st.subheader("Answer")
            st.write(result[0]['generated_text'])
        else:
            st.warning("Please enter a question")