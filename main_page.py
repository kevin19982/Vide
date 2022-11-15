# pages

# load packages
import streamlit as st

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

# input user query
query = st.text_input("Mortal, pose your request!")

# model
import os
os.system("pip install transformers  --user")
os.system("pip install tensorflow --user")

from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer, pipeline
from pathlib import Path

file_10k = open("C:/Users/siriv/Downloads/Data_Science_Project/test_files/new_file_test.txt", encoding = "utf8")
#text_10k = file_10k.read()
text_10k = "Trees tend to overfit if the gardener ceases to stop their growth to achieve greatness."

model_name = "deepset/roberta-base-squad2"

nlp = pipeline("question-answering", model = model_name, tokenizer = model_name)
QA_input = {
    "question": query,
    "context": text_10k
}

res = nlp(QA_input)

model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

st.write("result:", res)
