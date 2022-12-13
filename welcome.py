# pages

# load packages
import streamlit as st
from PIL import Image
import pandas as pd

# page configurations
st.set_page_config(
    menu_items={
        "Get Help": "http://www.vide-qa.de/contacts",
        "Report a bug": "http://www.vide-qa.de/contacts",
        "About": "This is a project by 2 students."
    }
)

# import dataframe with companies and years
#from comp_year import comp_year_df

# dataframe for company-year-combinations
comp_year_df = pd.read_csv("company_year.csv")

# load pictures
logo_3 = Image.open("logo_temp/vide_design_3.png")

# header
header1, header2, header3, header4, header5 = st.columns(5)
with header3:
    st.image(logo_3)

# title
st.title("VIDE Question-Answering")

# sidebar name of page
with st.sidebar:
    st.image(logo_3)
st.sidebar.markdown("# Application")

st.markdown("## Welcome")
# introductory text
st.write("Welcome to Vide - Question and Answering, a platform to extract relevant information from financial documents. We aim to deliver fast and concise answers to any question you have regarding recent developments in the financial world.")

st.write("To get answers to your questions please either upload a .txt-file or select a company and year from the dropdown-menu. Subsequently, please type in your question into the dedicated field. The last step is to wait for your answer.")


# application
st.markdown("## Application")
st.write("In the first step you can either upload a .txt-file of the document you want to ask about or select a company and year from the selection boxes below.")

# create columns for document choice
doc1, doc2 = st.columns(2)

# upload file
with doc1:
    file_10k = st.file_uploader("Please upload a txt of the document you would like to examine.")

# select company and year
with doc2:
    choices_company = st.selectbox("Please select the company of your choice.", comp_year_df["company"].unique())
    choices_quarter = st.selectbox("Please select the year of your choice.", comp_year_df.loc[comp_year_df["company"] == choices_company, "year"])

st.write("In the next step, you can input your question and afterwards wait for an answer regarding your question.")

# input user query
query = st.text_input("Please insert your question into the text-box below.")

# start computations
st.write("Lastly, please start the answer retrieval by clicking the following Submit-button.")
submit = st.button("Submit")

# computation
if submit:  # if submit-button is clicked
    if query:  # if query is inserted, start computation
        with st.spinner("Please wait, our army of insanely intelligent raccoons skims the document for your answer."):
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
            import random
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
            idx = np.where(cosine_scores_query[0] > 0.4)[0]   # the problem lies here

            # join retrieved sentences
            sentences_answer = ". ".join(np.array(text_sentences)[idx])


        # if the list of sentences with a cosine score above the threshold is not empty
            if list(np.array(text_sentences)[idx]):
            # print index, cosine score and sentence
                #for i in idx:
                    #st.write(i, ": ", np.round(cosine_scores_query[0][i].numpy(), 2), "; ", text_sentences[i])


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

                st.write("Answer: ", res["answer"])
            else:
                answers_else = {"There was no answer found in the document. Maybe, I don't know, throw a dice?",
                "Error, begin human extinction.",
                "Have you tried tarrot cards?",
                "I did my best, I swear.",
                "Error, it seems like you accidently uploaded the lyrics of the first Pokemon theme.",
                "The morning is nigh, you did try your best for now, no answer in sight. - Thoughts of a lonely program 1st edition",
                "A wise person once said to trust your gut. Might be better than trusting me in this case.",
                "You seem desparate to ask for help from me. And I seem desparate to give you this help... I didn't mean it, please ask me something, I have a family to feed.",
                "Bark bark bark... I'm sorry, I accidentally loaded the wrong language module. It should be fine now.",
                "Have you tried making music before? It's nice (pls do, I need someone to talk about my interests with).",
                "Unfortunately, I hit a wall. Literally. My head hurts..."}
                st.write("Answer: ", random.sample(answers_else, 1)[0])  # print one randomly selected sentence, if no answer to the question could be found
                st.write("(No answer found, please insert another question or try another document.)")
    else:
        st.write("Please enter a query into the intended text-box to initiate a search.")
