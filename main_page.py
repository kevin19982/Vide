# pages

# load streamlit package
import streamlit as st

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

# upload file
file_10k = st.file_uploader("Please upload a txt of the document you would like to examine.")
# input user query
query = st.text_input("Mortal being, pose your request!")

# retriever
# import packages
import nltk  # natural language toolkit
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
import glob
import re
import os
import numpy as np
import sys
import math
import operator
from sentence_transformers import SentenceTransformer
import sentence_transformers
from io import StringIO
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download('omw-1.4')

# load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# text is english
Stopwords = set(stopwords.words("english")) 

# transform document
file = StringIO(file_10k.getvalue().decode("utf-8"))
text = file.read()

# split text into sentences
text_sentences = text.split(". ")

# calculate embeddings
embeddings_query = model.encode(query, convert_to_tensor = True)
embeddings_sentences = model.encode(text_sentences, convert_to_tensor = True)  # the longest step

# compute cosine-similarities
cosine_scores_query = sentence_transformers.util.pytorch_cos_sim(embeddings_query, embeddings_sentences)

# get indices of highest cosine scores
idx = np.where(cosine_scores_query[0] > 0.4)[0]

# join retrieved sentences
sentences_answer = ". ".join(np.array(text_sentences)[idx])


# q and a
# loading packages
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from pathlib import Path

# get an answer to the question
model_name = "deepset/roberta-base-squad2"

nlp = pipeline("question-answering", model = model_name, tokenizer = model_name)
QA_input = {
    "question": query,
    "context": sentences_answer
}

res = nlp(QA_input)

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

st.write("Answer:", res)

