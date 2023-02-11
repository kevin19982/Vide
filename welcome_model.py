# pages

# load packages
import streamlit as st
from PIL import Image
import pandas as pd
import json



# page configurations
st.set_page_config(
    menu_items={
        "Get Help": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "Report a bug": "http://www.vide-qa.de/contacts",
        "About": "This is a project by 2 students based on the finqa dataset."
    }
)



# setup of the page
# load pictures
import datetime
if datetime.datetime.now().month == 12:  # in december
    logo = Image.open("C:/Users/siriv/Downloads/Data_Science_Project/scripts/logo_temp/vide_design_winter.png")  # winter-theme
else:  # every other month
    logo = Image.open("C:/Users/siriv/Downloads/Data_Science_Project/scripts/logo_temp/vide_design_3.png")  # base


# header
header1, header2, header3, header4, header5 = st.columns(5)
with header3:
    st.image(logo)


# title
st.title("VIDE Question-Answering")


# sidebar name of page
with st.sidebar:
    st.image(logo)
st.sidebar.markdown("# Application")


# introductory text
st.markdown("## Welcome")
st.write("Welcome to Vide - Question and Answering, a platform to extract relevant information from financial documents. We aim to deliver fast and concise answers to any question you have regarding recent developments in the financial world.")
st.write("To get answers to your questions please either upload a .txt-file or select a company and year from the dropdown-menu. Subsequently, please type in your question into the dedicated field. The last step is to wait for your answer.")


# application
st.markdown("## Application")
st.write("In the first step you can either upload a .txt-file of the document you want to ask about or select a company and year from the selection boxes below.")



# user-choice of specifications
# document choice
# import dataframe with companies and years
#from company_year import comp_year

# dataframe for company-year-combinations
comp_year_df = pd.read_csv("C:/Users/siriv/Downloads/Data_Science_Project/scripts/data/company_year.csv")


# create columns for document choice
doc1, doc2 = st.columns(2)
# upload file
with doc1:
    file_10k = st.file_uploader("Please upload a txt of the document you would like to examine.")  # document upload
# select company and year
with doc2:
    choices_company = st.selectbox("Please select the company of your choice.", comp_year_df["company"].unique())
    choices_quarter = st.selectbox("Please select the year of your choice.", comp_year_df.loc[comp_year_df["company"] == choices_company, "year"])


# choice of question
st.write("In the next step, you can input your question and afterwards wait for an answer regarding your question.")
# input user query
query = st.text_input("Please insert your question into the text-box below.")


# recording- and translation-functions may be enabled again once testing finished
# recording 
#audio_q = 0

#import os

#os.chdir("C:/Users/siriv/Downloads/Data_Science_Project/scripts/")

#from audiorecorder import audiorecorder

#st.write("Or you can input your question by recording it:")
#audio = audiorecorder("Click to record", "Recording...")

#if len(audio) > 0:
#    st.audio(audio.tobytes())
#    audio_q = 1
#
#    wav_file = open("audio.mp3", "wb")
#    wav_file.write(audio.tobytes())


# whisper api to translate audio to text
#import whisper
#import torch
#import numpy as np
query_2 = 0
#if audio_q == 1:
#    model = whisper.load_model("tiny")
#    result = model.transcribe("audio.mp3")
#    st.write(result["text"])
#    query_2 = result["text"]


# translation
#import deepl

#auth_key = "1dedb1b6-7a6a-db27-52aa-ac5b2d646c67:fx"
#translator = deepl.Translator(auth_key) 

#if query:   # translate text-input
#    text = query
#    target_language = "EN-GB"

#    query = translator.translate_text(text, target_lang=target_language).text
#    st.write("Your quesion: ", query)
#elif query_2 !=0:  # translate audio-input
#    text = query_2
#    target_language = "EN-GB"

#    query = translator.translate_text(text, target_lang=target_language).text
#    st.write("Your quesion: ", query)
#else:
#    pass



# choice of model
model_type = st.radio("Which type of question are you asking?",
("Calculation", "Fact"))
# start computations
st.write("Lastly, please start the answer retrieval by clicking the following Submit-button. A notificatioin sound will play, once your answer is ready.")
submit = st.button("Submit")  # submit-button



# model
# load in basic json-file (template) for text and question
# add text to json-file 
# add table to json-file
# load input-template (basic json-file with empty inputs)
#query = "What is up?"  # test
json_in = "C:/Users/siriv/Downloads/Data_Science_Project/scripts/input_template.json"
with open(json_in) as f_in:
    model_data = json.load(f_in)

if query or query_2 != 0:  # if query input
    # add question
    model_data[0]["qa"]["question"] = query if query else query_2
    # add text
    model_data[0]["pre_text"] = file_10k

#st.write(model_data)
#model_data[0]["qa"]["question"] = "How you doing, fella?"
#st.write(model_data[0]["qa"]["question"])



# computation
if submit:  # if submit-button is clicked
    if query or query_2:  # if query is inserted, start computation
        with st.spinner("Please wait, our army of insanely intelligent raccoons skims the document for your answer."):
            import base64
            graphic_waiting = open("C:/Users/siriv/Downloads/Data_Science_Project/scripts/racoon/graphic_waiting_4.gif", "rb")
            contents_waiting = graphic_waiting.read()
            data_waiting = base64.b64encode(contents_waiting).decode("utf-8")
            graphic_waiting.close()
            waiting_location = st.empty()
            waiting_location.markdown(
                f'<img src="data:image/gif;base64,{data_waiting}" alt="waiting gif">',
                unsafe_allow_html=True)

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
            file = StringIO(file_10k.getvalue().decode("unicode_escape"))
            text = file.read()

            # split text into sentences
            text_sentences = text.split(". ")

            # calculate embeddings
            if query:
                embeddings_query = model.encode(query, convert_to_tensor = True)
            else:
                embeddings_query = model.encode(query_2, convert_to_tensor = True)
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
                if model_type == "Fact":
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
                    """input finqa model"""
            else:
                answers_else = {
                    "There was no answer found in the document. Maybe, I don't know, throw a dice?",
                    "Error, begin human extinction.",
                    "Have you tried tarrot cards?",
                    "I did my best, I swear.",
                    "Error, it seems like you accidently uploaded a recipe for a nice vegetarian stew. Not the time and place, to be frank, but maybe nice for another time.",
                    "The morning is nigh, you did try your best for now, no answer in sight. - Thoughts of a lonely program 1st edition",
                    "A wise person once said to trust your gut. Might be better than trusting me in this case.",
                    "You seem desparate to ask for help from me. And I seem desparate to give you this help... I didn't mean it, please ask me something, I have a family to feed.",
                    "Bark bark bark... I'm sorry, I accidentally loaded the wrong language module. It should be fine now.",
                    "This is music to my ears. Nothing I can make sense of, but music nontheless.",
                    "Unfortunately, I hit a wall. Literally. My head hurts...",
                    "Sometimes we had the answer within ourselves all along."}
                st.write("Answer: ", random.sample(answers_else, 1)[0])  # print one randomly selected sentence, if no answer to the question could be found
                st.write("(No answer found, please insert another question or try another document.)")
                graphic_no_find = Image.open("C:/Users/siriv/Downloads/Data_Science_Project/scripts/racoon/racoon_5.png")
                st.image(graphic_no_find)
    else:
        st.write("Please enter a query into the intended text-box to initiate a search.")
    done = 1
    if done == 1:
        # winter-theme if it is december
        if datetime.datetime.now().month == 12:
            st.snow()  # winter-theme
        # notification-sound (when computation finished)
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load("C:/Users/siriv/Downloads/Data_Science_Project/scripts/w_short.mp3")
        pygame.mixer.music.play()  # play notification-sound

        waiting_location.text("")  # stop loading-animation
    done = 0


# add some empty space for spacing
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")


# show question examples
graphic_1 = Image.open("C:/Users/siriv/Downloads/Data_Science_Project/scripts/vide_graphic_1.png")
st.image(graphic_1)


