# pages

# load packages
import streamlit as st

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

# input user query
query = st.text_input("Mortal, pose your request!")

# import packages
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from pathlib import Path

file_10k = open("new_file_test.txt", encoding = "utf8")
text_10k = file_10k.read()
#text_10k = "Trees tend to overfit if the gardener ceases to stop their growth to achieve greatness."

model_name = "deepset/roberta-base-squad2"

nlp = pipeline("question-answering", model = model_name, tokenizer = model_name)
QA_input = {
    "question": query,
    "context": text_10k
}

res = nlp(QA_input)

@st.cache
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

st.write("result:", res)
