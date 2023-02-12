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
    logo = Image.open("logo_temp/vide_design_winter.png")  # winter-theme
else:  # every other month
    logo = Image.open("logo_temp/vide_design_3.png")  # base


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
comp_year_df = pd.read_csv("company_year.csv")


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
st.write("If you run into problems with the results, it might be helpful to rephrase the question, give extra information or leave some information out.")


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
json_in = "input_template.json"
with open(json_in) as f_in:
    model_data = json.load(f_in)

#if query or query_2 != 0:  # if query input
    # add question
#    model_data[0]["qa"]["question"] = query if query else query_2
    # add text
#    model_data[0]["pre_text"] = file_10k

#st.write(model_data)
#model_data[0]["qa"]["question"] = "How you doing, fella?"
#st.write(model_data[0]["qa"]["question"])



# computation
if submit:  # if submit-button is clicked
    if query or query_2:  # if query is inserted, start computation
        with st.spinner("Please wait, our army of insanely intelligent raccoons skims the document for your answer."):
            import base64
            graphic_waiting = open("racoon/graphic_waiting_4.gif", "rb")
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
            #st.write(text)

            # split text into sentences
            text_sentences = text.split(". ")
            #st.write(text_sentences)

            # calculate embeddings
            if query:
                embeddings_query = model.encode(query, convert_to_tensor = True)
            else:
                embeddings_query = model.encode(query_2, convert_to_tensor = True)
            embeddings_sentences = model.encode(text_sentences, convert_to_tensor = True)  # the longest step

            # compute cosine-similarities
            cosine_scores_query = sentence_transformers.util.pytorch_cos_sim(embeddings_query, embeddings_sentences)

            # get indices of highest cosine scores
            idx = np.where(cosine_scores_query[0] > 0.3)[0]   # the problem lies here
            #st.write("idx: ", idx)
            # join retrieved sentences
            #st.write(text_sentences)
            sentences_answer = np.array(text_sentences)[idx]  #sentences_answer = ". ".join(np.array(text_sentences)[idx])
            sentences_answer = sentences_answer.tolist()
            #st.write(sentences_answer)


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
                    """"
                    things that can be altered in the script: "changeable"
                    """

                    # enter query and text
                    if query or query_2 != 0:  # if query input
                        # add question
                        model_data[0]["qa"]["question"] = query if query else query_2
                        # add text
                        model_data[0]["pre_text"] = [
                            "entergy corporation and subsidiaries management 2019s financial discussion and analysis a result of the entergy louisiana and entergy gulf states louisiana business combination , results of operations for 2015 also include two items that occurred in october 2015 : 1 ) a deferred tax asset and resulting net increase in tax basis of approximately $ 334 million and 2 ) a regulatory liability of $ 107 million ( $ 66 million net-of-tax ) as a result of customer credits to be realized by electric customers of entergy louisiana , consistent with the terms of the stipulated settlement in the business combination proceeding .",
                            "see note 2 to the financial statements for further discussion of the business combination and customer credits .",
                            "results of operations for 2015 also include the sale in december 2015 of the 583 mw rhode island state energy center for a realized gain of $ 154 million ( $ 100 million net-of-tax ) on the sale and the $ 77 million ( $ 47 million net-of-tax ) write-off and regulatory charges to recognize that a portion of the assets associated with the waterford 3 replacement steam generator project is no longer probable of recovery .",
                            "see note 14 to the financial statements for further discussion of the rhode island state energy center sale .",
                            "see note 2 to the financial statements for further discussion of the waterford 3 write-off .",
                            "results of operations for 2014 include $ 154 million ( $ 100 million net-of-tax ) of charges related to vermont yankee primarily resulting from the effects of an updated decommissioning cost study completed in the third quarter 2014 along with reassessment of the assumptions regarding the timing of decommissioning cash flows and severance and employee retention costs .",
                            "see note 14 to the financial statements for further discussion of the charges .",
                            "results of operations for 2014 also include the $ 56.2 million ( $ 36.7 million net-of-tax ) write-off in 2014 of entergy mississippi 2019s regulatory asset associated with new nuclear generation development costs as a result of a joint stipulation entered into with the mississippi public utilities staff , subsequently approved by the mpsc , in which entergy mississippi agreed not to pursue recovery of the costs deferred by an mpsc order in the new nuclear generation docket .",
                            "see note 2 to the financial statements for further discussion of the new nuclear generation development costs and the joint stipulation .",
                            "net revenue utility following is an analysis of the change in net revenue comparing 2015 to 2014 .",
                            "amount ( in millions ) .",
                            "the retail electric price variance is primarily due to : 2022 formula rate plan increases at entergy louisiana , as approved by the lpsc , effective december 2014 and january 2015 ; 2022 an increase in energy efficiency rider revenue primarily due to increases in the energy efficiency rider at entergy arkansas , as approved by the apsc , effective july 2015 and july 2014 , and new energy efficiency riders at entergy louisiana and entergy mississippi that began in the fourth quarter 2014 ; and 2022 an annual net rate increase at entergy mississippi of $ 16 million , effective february 2015 , as a result of the mpsc order in the june 2014 rate case .",
                            "see note 2 to the financial statements for a discussion of rate and regulatory proceedings .",
                            "The 2014 net revenue is $ 5735 .",
                            "The retail electric price is 187 .",
                            "The volume/weather is 95 .",
                            "The waterford 3 replacement steam generator provision is -32 .",
                            "The 2015 net revenue is $ 5829 .",
                            "The 2016 number of employees is 10 .",
                            "The 2017 number of employees is 12 .",
                            "Nice weather we have today, truely ."
                        ]
                    
                    #st.write(model_data[0])

                    # import packages
                    import torch
                    import argparse
                    import collections
                    import json
                    import os
                    import re
                    import string
                    import sys
                    import random
                    import numpy as np
                    import enum
                    import six
                    import copy
                    from six.moves import map
                    from six.moves import range
                    from six.moves import zip
                    import time
                    import shutil
                    import io
                    import subprocess
                    import zipfile
                    import math
                    import torch.nn.functional as F
                    from tqdm import tqdm
                    from transformers import BertTokenizer, BertModel, BertConfig
                    from torch import nn
                    import torch.optim as optim
                    from datetime import datetime
                    import logging
                    from transformers import BertTokenizer
                    from transformers import BertConfig
                    from sympy import simplify





                    # retriever
                    # config retriever
                    class conf():

                        prog_name = "retriever"

                        # set up your own path here
                        root_path = ""
                        output_path = "path_to_store_outputs/"
                        cache_dir = "path_for_other_cache/"

                        # the name of your result folder.
                        model_save_name = "retriever-bert-base-test"

                        train_file = root_path + "dataset/train.json"
                        valid_file = root_path + "dataset/dev.json"

                        test_file = root_path + "model_input_no_table.json"  # changeable, changed to model_data before reading examples in the test-part of the script

                        op_list_file = "operation_list.txt"
                        const_list_file = "constant_list.txt"

                        # model choice: bert, roberta
                        pretrained_model = "bert"
                        model_size = "bert-base-uncased"

                        # pretrained_model = "roberta"
                        # model_size = "roberta-base"

                        # train, test, or private
                        # private: for testing private test data
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        mode = "test"  # changed from mode = "train"
                        resume_model_path = ""

                        ### to load the trained model in test time
                        saved_model_path = root_path + "model_retriever.pt" 
                        build_summary = False

                        option = "rand"
                        neg_rate = 3
                        topn = 5

                        sep_attention = True
                        layer_norm = True
                        num_decoder_layers = 1

                        max_seq_length = 512
                        max_program_length = 100
                        n_best_size = 20
                        dropout_rate = 0.1

                        batch_size = 16
                        batch_size_test = 16
                        epoch = 2
                        learning_rate = 2e-5

                        report = 300
                        report_loss = 100



                    # define special tokens
                    _SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)



                    # general_utils
                    def remove_space(text_in):
                        res = []

                        for tmp in text_in.split(" "):
                            if tmp != "":
                                res.append(tmp)

                        return " ".join(res)


                    def table_row_to_text(header, row):
                        '''
                        use templates to convert table row to text
                        '''
                        res = ""
                        
                        if header[0]:
                            res += (header[0] + " ")

                        for head, cell in zip(header[1:], row[1:]):
                            res += ("the " + row[0] + " of " + head + " is " + cell + " ; ")
                        
                        res = remove_space(res)
                        return res.strip()



                    # finqa_utils
                    def str_to_num(text):
                        text = text.replace(",", "")
                        try:
                            num = int(text)
                        except ValueError:
                            try:
                                num = float(text)
                            except ValueError:
                                if text and text[-1] == "%":
                                    num = text
                                else:
                                    num = None
                        return num


                    def prog_token_to_indices(prog, numbers, number_indices, max_seq_length,
                                            op_list, op_list_size, const_list,
                                            const_list_size):
                        prog_indices = []
                        for i, token in enumerate(prog):
                            if token in op_list:
                                prog_indices.append(op_list.index(token))
                            elif token in const_list:
                                prog_indices.append(op_list_size + const_list.index(token))
                            else:
                                if token in numbers:
                                    cur_num_idx = numbers.index(token)
                                else:
                                    cur_num_idx = -1
                                    for num_idx, num in enumerate(numbers):
                                        if str_to_num(num) == str_to_num(token):
                                            cur_num_idx = num_idx
                                            break
                                assert cur_num_idx != -1
                                prog_indices.append(op_list_size + const_list_size +
                                                    number_indices[cur_num_idx])
                        return prog_indices


                    def indices_to_prog(program_indices, numbers, number_indices, max_seq_length,
                                        op_list, op_list_size, const_list, const_list_size):
                        prog = []
                        for i, prog_id in enumerate(program_indices):
                            if prog_id < op_list_size:
                                prog.append(op_list[prog_id])
                            elif prog_id < op_list_size + const_list_size:
                                prog.append(const_list[prog_id - op_list_size])
                            else:
                                prog.append(numbers[number_indices.index(prog_id - op_list_size
                                                                        - const_list_size)])
                        return prog


                    class MathQAExample(
                            collections.namedtuple(
                                "MathQAExample",
                                "filename_id question all_positive \
                                pre_text post_text table"
                            )):

                        def convert_single_example(self, *args, **kwargs):
                            return convert_single_mathqa_example(self, *args, **kwargs)


                    class InputFeatures(object):
                        """A single set of features of data."""

                        def __init__(self,
                                    filename_id,
                                    retrieve_ind,
                                    tokens,
                                    input_ids,
                                    segment_ids,
                                    input_mask,
                                    label):

                            self.filename_id = filename_id
                            self.retrieve_ind = retrieve_ind
                            self.tokens = tokens
                            self.input_ids = input_ids
                            self.input_mask = input_mask
                            self.segment_ids = segment_ids
                            self.label = label


                    def tokenize(tokenizer, text, apply_basic_tokenization=False):
                        """Tokenizes text, optionally looking up special tokens separately.
                        Args:
                        tokenizer: a tokenizer from bert.tokenization.FullTokenizer
                        text: text to tokenize
                        apply_basic_tokenization: If True, apply the basic tokenization. If False,
                            apply the full tokenization (basic + wordpiece).
                        Returns:
                        tokenized text.
                        A special token is any text with no spaces enclosed in square brackets with no
                        space, so we separate those out and look them up in the dictionary before
                        doing actual tokenization.
                        """

                        if conf.pretrained_model in ["bert", "finbert"]:
                            _SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
                        elif conf.pretrained_model in ["roberta", "longformer"]:
                            _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

                        tokenize_fn = tokenizer.tokenize
                        if apply_basic_tokenization:
                            tokenize_fn = tokenizer.basic_tokenizer.tokenize

                        tokens = []
                        for token in text.split(" "):
                            if _SPECIAL_TOKENS_RE.match(token):
                                if token in tokenizer.get_vocab():
                                    tokens.append(token)
                                else:
                                    tokens.append(tokenizer.unk_token)
                            else:
                                tokens.extend(tokenize_fn(token))

                        return tokens


                    def _detokenize(tokens):
                        text = " ".join(tokens)

                        text = text.replace(" ##", "")
                        text = text.replace("##", "")

                        text = text.strip()
                        text = " ".join(text.split())
                        return text


                    def program_tokenization(original_program):
                        original_program = original_program.split(', ')
                        program = []
                        for tok in original_program:
                            cur_tok = ''
                            for c in tok:
                                if c == ')':
                                    if cur_tok != '':
                                        program.append(cur_tok)
                                        cur_tok = ''
                                cur_tok += c
                                if c in ['(', ')']:
                                    program.append(cur_tok)
                                    cur_tok = ''
                            if cur_tok != '':
                                program.append(cur_tok)
                        program.append('EOF')
                        return program



                    def get_tf_idf_query_similarity(allDocs, query):
                        """
                        vectorizer: TfIdfVectorizer model
                        docs_tfidf: tfidf vectors for all docs
                        query: query doc
                        return: cosine similarity between query and all docs
                        """
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        from sklearn.metrics.pairwise import cosine_similarity

                        vectorizer = TfidfVectorizer(stop_words='english')
                        docs_tfidf = vectorizer.fit_transform(allDocs)
                        
                        query_tfidf = vectorizer.transform([query])
                        cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()
                        
                        # print(cosineSimilarities)
                        return cosineSimilarities


                    def wrap_single_pair(tokenizer, question, context, label, max_seq_length,
                                        cls_token, sep_token):
                        '''
                        single pair of question, context, label feature
                        '''
                        
                        question_tokens = tokenize(tokenizer, question)
                        this_gold_tokens = tokenize(tokenizer, context)

                        tokens = [cls_token] + question_tokens + [sep_token]
                        segment_ids = [0] * len(tokens)

                        tokens += this_gold_tokens
                        segment_ids.extend([0] * len(this_gold_tokens))

                        if len(tokens) > max_seq_length:
                            tokens = tokens[:max_seq_length-1]
                            tokens += [sep_token]
                            segment_ids = segment_ids[:max_seq_length]

                        input_ids = tokenizer.convert_tokens_to_ids(tokens)
                        input_mask = [1] * len(input_ids)

                        padding = [0] * (max_seq_length - len(input_ids))
                        input_ids.extend(padding)
                        input_mask.extend(padding)
                        segment_ids.extend(padding)

                        assert len(input_ids) == max_seq_length
                        assert len(input_mask) == max_seq_length
                        assert len(segment_ids) == max_seq_length
                        
                        this_input_feature = {
                            "context": context,
                            "tokens": tokens,
                            "input_ids": input_ids,
                            "input_mask": input_mask,
                            "segment_ids": segment_ids,
                            "label": label
                        }
                        
                        return this_input_feature

                    def convert_single_mathqa_example(example, option, is_training, tokenizer, max_seq_length,
                                                    cls_token, sep_token):
                        """Converts a single MathQAExample into Multiple Retriever Features."""
                        """ option: tf idf or all"""
                        """train: 1:3 pos neg. Test: all"""

                        pos_features = []
                        features_neg = []
                        
                        question = example.question
                        all_text = example.pre_text + example.post_text

                        if is_training:
                            for gold_ind in example.all_positive:

                                this_gold_sent = example.all_positive[gold_ind]
                                this_input_feature = wrap_single_pair(
                                    tokenizer, question, this_gold_sent, 1, max_seq_length,
                                    cls_token, sep_token)

                                this_input_feature["filename_id"] = example.filename_id
                                this_input_feature["ind"] = gold_ind
                                pos_features.append(this_input_feature)
                                
                            num_pos_pair = len(example.all_positive)
                            num_neg_pair = num_pos_pair * conf.neg_rate
                                
                            pos_text_ids = []
                            pos_table_ids = []
                            for gold_ind in example.all_positive:
                                if "text" in gold_ind:
                                    pos_text_ids.append(int(gold_ind.replace("text_", "")))
                                elif "table" in gold_ind:
                                    pos_table_ids.append(int(gold_ind.replace("table_", "")))

                            all_text_ids = range(len(example.pre_text) + len(example.post_text))
                            all_table_ids = range(1, len(example.table))
                            
                            all_negs_size = len(all_text) + len(example.table) - len(example.all_positive)
                            if all_negs_size < 0:
                                all_negs_size = 0
                                        
                            # test: all negs
                            # text
                            for i in range(len(all_text)):
                                if i not in pos_text_ids:
                                    this_text = all_text[i]
                                    this_input_feature = wrap_single_pair(
                                        tokenizer, example.question, this_text, 0, max_seq_length,
                                        cls_token, sep_token)
                                    this_input_feature["filename_id"] = example.filename_id
                                    this_input_feature["ind"] = "text_" + str(i)
                                    features_neg.append(this_input_feature)
                                # table      
                            for this_table_id in range(len(example.table)):
                                if this_table_id not in pos_table_ids:
                                    this_table_row = example.table[this_table_id]
                                    this_table_line = table_row_to_text(example.table[0], example.table[this_table_id])
                                    this_input_feature = wrap_single_pair(
                                        tokenizer, example.question, this_table_line, 0, max_seq_length,
                                        cls_token, sep_token)
                                    this_input_feature["filename_id"] = example.filename_id
                                    this_input_feature["ind"] = "table_" + str(this_table_id)
                                    features_neg.append(this_input_feature)
                                    
                        else:
                            pos_features = []
                            features_neg = []
                            question = example.question

                            ### set label as -1 for test examples
                            for i in range(len(all_text)):
                                this_text = all_text[i]
                                this_input_feature = wrap_single_pair(
                                    tokenizer, example.question, this_text, -1, max_seq_length,
                                    cls_token, sep_token)
                                this_input_feature["filename_id"] = example.filename_id
                                this_input_feature["ind"] = "text_" + str(i)
                                features_neg.append(this_input_feature)
                                # table      
                            for this_table_id in range(len(example.table)):
                                this_table_row = example.table[this_table_id]
                                this_table_line = table_row_to_text(example.table[0], example.table[this_table_id])
                                this_input_feature = wrap_single_pair(
                                    tokenizer, example.question, this_table_line, -1, max_seq_length,
                                    cls_token, sep_token)
                                this_input_feature["filename_id"] = example.filename_id
                                this_input_feature["ind"] = "table_" + str(this_table_id)
                                features_neg.append(this_input_feature)

                        return pos_features, features_neg


                    def read_mathqa_entry(entry, tokenizer):

                        filename_id = entry["id"]
                        question = entry["qa"]["question"]
                        if "gold_inds" in entry["qa"]:
                            all_positive = entry["qa"]["gold_inds"]
                        else:
                            all_positive = []

                        pre_text = entry["pre_text"]
                        post_text = entry["post_text"]
                        table = entry["table"]

                        return MathQAExample(
                            filename_id=filename_id,
                            question=question,
                            all_positive=all_positive,
                            pre_text=pre_text,
                            post_text=post_text,
                            table=table)



                    # utils
                    def write_word(pred_list, save_dir, name):
                        ss = open(save_dir + name, "w+")
                        for item in pred_list:
                            ss.write(" ".join(item) + '\n')


                    def write_log(log_file, s):
                        pass
                        """"
                        print(s)
                        with open(log_file, 'a') as f:
                            f.write(s+'\n')
                        """


                    def _compute_softmax(scores):
                        """Compute softmax probability over raw logits."""
                        if not scores:
                            return []

                        max_score = None
                        for score in scores:
                            if max_score is None or score > max_score:
                                max_score = score

                        exp_scores = []
                        total_sum = 0.0
                        for score in scores:
                            x = math.exp(score - max_score)
                            exp_scores.append(x)
                            total_sum += x

                        probs = []
                        for score in exp_scores:
                            probs.append(score / total_sum)
                        return probs


                    def read_txt(input_path, log_file):
                        """Read a txt file into a list."""

                        #write_log(log_file, "Reading: %s" % input_path)
                        with open(input_path) as input_file:
                            input_data = input_file.readlines()
                        items = []
                        for line in input_data:
                            items.append(line.strip())
                        return items


                    def read_examples(input_path, tokenizer, op_list, const_list, log_file):
                        """Read a json file into a list of examples."""

                        #write_log(log_file, "Reading " + input_path)
                        #with open(input_path) as input_file:
                        #    input_data = json.load(input_file)
                        
                        input_data = input_path

                        examples = []
                        for entry in input_data:
                            examples.append(read_mathqa_entry(entry, tokenizer))

                        return input_data, examples, op_list, const_list


                    def convert_examples_to_features(examples,
                                                    tokenizer,
                                                    max_seq_length,
                                                    option,
                                                    is_training,
                                                    ):
                        """Converts a list of DropExamples into InputFeatures."""
                        res = []
                        res_neg = []
                        for (example_index, example) in tqdm(enumerate(examples)):
                            features, features_neg = example.convert_single_example(
                                tokenizer=tokenizer,
                                max_seq_length=max_seq_length,
                                option=option,
                                is_training=is_training,
                                cls_token=tokenizer.cls_token,
                                sep_token=tokenizer.sep_token)

                            res.extend(features)
                            res_neg.extend(features_neg)

                        return res, res_neg


                    def write_predictions(all_predictions, output_prediction_file):
                        """Writes final predictions in json format."""

                        with open(output_prediction_file, "w") as writer:
                            writer.write(json.dumps(all_predictions, indent=4) + "\n")


                    class DataLoader:
                        def __init__(self, is_training, data, batch_size=64, shuffle=True):
                            """
                            Main dataloader
                            """
                            self.data_pos = data[0]
                            self.data_neg = data[1]
                            self.batch_size = batch_size
                            self.is_training = is_training
                            
                            
                            if self.is_training:
                                random.shuffle(self.data_neg)
                                if conf.option == "tfidf":
                                    self.data = self.data_pos + self.data_neg
                                else:
                                    num_neg = len(self.data_pos) * conf.neg_rate
                                    self.data = self.data_pos + self.data_neg[:num_neg]
                            else:
                                self.data = self.data_pos + self.data_neg
                                
                                
                            self.data_size = len(self.data)
                            self.num_batches = int(self.data_size / batch_size) if self.data_size % batch_size == 0 \
                                else int(self.data_size / batch_size) + 1

                            self.count = 0

                        def __iter__(self):
                            return self

                        def __next__(self):
                            # drop last batch
                            if self.is_training:
                                bound = self.num_batches - 1
                            else:
                                bound = self.num_batches
                            if self.count < bound:
                                return self.get_batch()
                            else:
                                raise StopIteration

                        def __len__(self):
                            return self.num_batches

                        def reset(self):
                            self.count = 0
                            self.shuffle_all_data()

                        def shuffle_all_data(self):
                            if conf.option == "tfidf":
                                random.shuffle(self.data)
                            else:
                                random.shuffle(self.data_neg)
                                num_neg = len(self.data_pos) * conf.neg_rate
                                self.data = self.data_pos + self.data_neg[:num_neg]
                                random.shuffle(self.data)
                            return

                        def get_batch(self):
                            start_index = self.count * self.batch_size
                            end_index = min((self.count + 1) * self.batch_size, self.data_size)

                            self.count += 1
                            # print (self.count)
                            

                            batch_data = {"input_ids": [],
                                        "input_mask": [],
                                        "segment_ids": [],
                                        "filename_id": [],
                                        "label": [],
                                        "ind": []
                                        }
                            for each_data in self.data[start_index: end_index]:

                                batch_data["input_ids"].append(each_data["input_ids"])
                                batch_data["input_mask"].append(each_data["input_mask"])
                                batch_data["segment_ids"].append(each_data["segment_ids"])
                                batch_data["filename_id"].append(each_data["filename_id"])
                                batch_data["label"].append(each_data["label"])
                                batch_data["ind"].append(each_data["ind"])


                            return batch_data


                    def cleanhtml(raw_html):
                        cleanr = re.compile('<.*?>')
                        cleantext = re.sub(cleanr, '', raw_html)
                        return cleantext


                    def retrieve_evaluate(all_logits, all_filename_ids, all_inds, output_prediction_file, ori_file, topn):
                        '''
                        save results to file. calculate recall
                        '''
                        
                        res_filename = {}
                        res_filename_inds = {}
                        
                        for this_logit, this_filename_id, this_ind in zip(all_logits, all_filename_ids, all_inds):
                            
                            if this_filename_id not in res_filename:
                                res_filename[this_filename_id] = []
                                res_filename_inds[this_filename_id] = []
                            if this_ind not in res_filename_inds[this_filename_id]:
                                res_filename[this_filename_id].append({
                                    "score": this_logit[1],
                                    "ind": this_ind
                                })
                                res_filename_inds[this_filename_id].append(this_ind)
                                
                            
                            
                        data_all = ori_file
                            
                        # take top ten
                        all_recall = 0.0
                        all_recall_3 = 0.0
                        
                        for data in data_all:
                            this_filename_id = data["id"]
                            
                            this_res = res_filename[this_filename_id]
                            
                            sorted_dict = sorted(this_res, key=lambda kv: kv["score"], reverse=True)
                            
                            sorted_dict = sorted_dict[:topn]
                            
                            gold_inds = data["qa"]["gold_inds"]
                            
                            # table rows
                            table_retrieved = []
                            text_retrieved = []

                            # all retrieved
                            table_re_all = []
                            text_re_all = []
                            
                            
                            for tmp in sorted_dict[:topn]:
                                if "table" in tmp["ind"]:
                                    table_retrieved.append(tmp)
                                else:
                                    text_retrieved.append(tmp)


                            for tmp in sorted_dict:
                                if "table" in tmp["ind"]:
                                    table_re_all.append(tmp)
                                else:
                                    text_re_all.append(tmp)


                            data["table_retrieved"] = table_retrieved
                            data["text_retrieved"] = text_retrieved

                            data["table_retrieved_all"] = table_re_all
                            data["text_retrieved_all"] = text_re_all
                        
                        retriever_output = data_all
                        
                        return retriever_output
                                        
                        

                    # model
                    if conf.pretrained_model == "bert":
                        from transformers import BertModel
                    elif conf.pretrained_model == "roberta":
                        from transformers import RobertaModel


                    class Bert_model(nn.Module):

                        def __init__(self, hidden_size, dropout_rate):

                            super(Bert_model, self).__init__()

                            self.hidden_size = hidden_size

                            if conf.pretrained_model == "bert":
                                self.bert = BertModel.from_pretrained(
                                    conf.model_size, cache_dir=conf.cache_dir)
                            elif conf.pretrained_model == "roberta":
                                self.bert = RobertaModel.from_pretrained(
                                    conf.model_size, cache_dir=conf.cache_dir)

                            self.cls_prj = nn.Linear(hidden_size, hidden_size, bias=True)
                            self.cls_dropout = nn.Dropout(dropout_rate)

                            self.cls_final = nn.Linear(hidden_size, 2, bias=True)

                        def forward(self, is_training, input_ids, input_mask, segment_ids, device):

                            bert_outputs = self.bert(
                                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

                            bert_sequence_output = bert_outputs.last_hidden_state

                            bert_pooled_output = bert_sequence_output[:, 0, :]

                            pooled_output = self.cls_prj(bert_pooled_output)
                            pooled_output = self.cls_dropout(pooled_output)

                            logits = self.cls_final(pooled_output)

                            return logits



                    # run retriever
                    # test
                    # define the tokenizer and the model configurations
                    tokenizer = BertTokenizer.from_pretrained(conf.model_size)
                    model_config = BertConfig.from_pretrained(conf.model_size)


                    # output paths and log-file
                    saved_model_path = os.path.join(conf.output_path, conf.saved_model_path)
                    model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S") + \
                        "_" + conf.model_save_name
                    model_dir = os.path.join(
                        conf.output_path, 'inference_only_' + model_dir_name)
                    results_path = os.path.join(model_dir, "results")
                    os.makedirs(results_path, exist_ok=False)
                    log_file = os.path.join(results_path, 'log.txt')


                    # import opeartions and constants for the programm generation
                    op_list = read_txt("operation_list.txt", log_file)
                    op_list = [op + '(' for op in op_list]
                    op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
                    const_list = read_txt("constant_list.txt", log_file)
                    const_list = [const.lower().replace('.', '_') for const in const_list]
                    reserved_token_size = len(op_list) + len(const_list)

                    input_file = model_data  # json-file with question and text, changeable

                    test_data, test_examples, op_list, const_list = \
                        read_examples(input_path=input_file, tokenizer=tokenizer,  # conf.test_file: replace with new input
                                    op_list=op_list, const_list=const_list, log_file=log_file)
                        

                    kwargs = {"examples": test_examples,
                            "tokenizer": tokenizer,
                            "option": conf.option,
                            "is_training": False,
                            "max_seq_length": conf.max_seq_length,
                            }


                    test_features = convert_examples_to_features(**kwargs)  # calls the function convert_sinqle_mathqa_example, outputs positive and negative features


                    def generate(data_ori, data, model, ksave_dir, mode='valid'):

                        pred_list = []
                        pred_unk = []

                        ksave_dir_mode = os.path.join(ksave_dir, mode)
                        os.makedirs(ksave_dir_mode, exist_ok=True)

                        data_iterator = DataLoader(
                            is_training=False, data=data, batch_size=conf.batch_size_test, shuffle=False)

                        k = 0
                        all_logits = []
                        all_filename_id = []
                        all_ind = []
                        with torch.no_grad():
                            for x in tqdm(data_iterator):

                                input_ids = x['input_ids']
                                print("inputs_ids: ", input_ids)
                                input_mask = x['input_mask']
                                print("input_mask: ", input_mask)
                                segment_ids = x['segment_ids']
                                print("segment_ids: ", segment_ids)
                                label = x['label']
                                print("label: ", label)
                                filename_id = x["filename_id"]
                                print("filename_id: ", filename_id)
                                ind = x["ind"]
                                print("ind: ", ind)

                                ori_len = len(input_ids)
                                for each_item in [input_ids, input_mask, segment_ids]:
                                    if ori_len < conf.batch_size_test:
                                        each_len = len(each_item[0])
                                        pad_x = [0] * each_len
                                        each_item += [pad_x] * (conf.batch_size_test - ori_len)

                                input_ids = torch.tensor(input_ids).to(conf.device)
                                input_mask = torch.tensor(input_mask).to(conf.device)
                                segment_ids = torch.tensor(segment_ids).to(conf.device)

                                logits = model(True, input_ids, input_mask,
                                            segment_ids, device=conf.device)
                                
                                print(logits)

                                all_logits.extend(logits.tolist())
                                all_filename_id.extend(filename_id)
                                all_ind.extend(ind)

                        output_prediction_file = "retriever_output.json"  # only used if results are saved

                        if mode == "valid":
                            print_res = retrieve_evaluate(
                                all_logits, all_filename_id, all_ind, output_prediction_file, conf.valid_file, topn=conf.topn)
                        elif mode == "test":
                            retriever_output = retrieve_evaluate(
                                all_logits, all_filename_id, all_ind, output_prediction_file, input_file, topn=conf.topn)  # input_file defined earlier
                        else:
                            pass

                        #write_log(log_file, print_res)
                        #print(print_res)
                        return retriever_output


                    def generate_test():
                        model = Bert_model(hidden_size=model_config.hidden_size,
                                        dropout_rate=conf.dropout_rate,)

                        model = nn.DataParallel(model)
                        model.to(conf.device)
                        model.load_state_dict(torch.load("model_retriever.pt", map_location = torch.device('cpu')))  # changeable
                        model.eval()
                        retriever_output = generate(test_data, test_features, model, results_path, mode='test')
                        return retriever_output


                    retriever_output = generate_test()





                    # convert retriever-output to model-input
                    '''
                    convert retriever results to generator test input
                    '''

                    ### for single sent retrieve

                    def convert_test(data_input, topn, max_len):

                        data = data_input

                        for each_data in data:
                            table_retrieved = each_data["table_retrieved"]
                            text_retrieved = each_data["text_retrieved"]

                            pre_text = each_data["pre_text"]
                            post_text = each_data["post_text"]
                            all_text = pre_text + post_text

                            table = each_data["table"]

                            all_retrieved = each_data["table_retrieved"] + each_data["text_retrieved"]

                            sorted_dict = sorted(all_retrieved, key=lambda kv: kv["score"], reverse=True)

                            acc_len = 0
                            all_text_in = {}
                            all_table_in = {}

                            for tmp in sorted_dict:
                                if len(all_table_in) + len(all_text_in) >= topn:
                                    break
                                this_sent_ind = int(tmp["ind"].split("_")[1])

                                if "table" in tmp["ind"]:
                                    this_sent = table_row_to_text(table[0], table[this_sent_ind])
                                else:
                                    this_sent = all_text[this_sent_ind]

                                if acc_len + len(this_sent.split(" ")) < max_len:
                                    if "table" in tmp["ind"]:
                                        all_table_in[tmp["ind"]] = this_sent
                                    else:
                                        all_text_in[tmp["ind"]] = this_sent

                                    acc_len += len(this_sent.split(" "))
                                else:
                                    break

                            this_model_input = []

                            # sorted_dict = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
                            # this_model_input.extend(sorted_dict)

                            # sorted_dict = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
                            # this_model_input.extend(sorted_dict)

                            # original_order
                            sorted_dict_table = sorted(all_table_in.items(), key=lambda kv: int(kv[0].split("_")[1]))
                            sorted_dict_text = sorted(all_text_in.items(), key=lambda kv: int(kv[0].split("_")[1]))

                            for tmp in sorted_dict_text:
                                if int(tmp[0].split("_")[1]) < len(pre_text):
                                    this_model_input.append(tmp)

                            for tmp in sorted_dict_table:
                                this_model_input.append(tmp)

                            for tmp in sorted_dict_text:
                                if int(tmp[0].split("_")[1]) >= len(pre_text):
                                    this_model_input.append(tmp)

                            each_data["qa"]["model_input"] = this_model_input

                        
                        conversion_output = data

                        print(len(data))

                        return conversion_output



                    # run conversion
                    json_in = retriever_output 
                    #st.write("here retriever")
                    #st.write(retriever_output)

                    conversion_output = convert_test(json_in, topn=3, max_len=290)  # takes the top 3 (topn=3) retrieved facts (among table and text)





                    # program-generator
                    # config
                    class conf():

                        prog_name = "generator"

                        # set up your own path here
                        root_path = "C:/Users/siriv/Downloads/generator_test/"
                        output_path = "C:/Users/siriv/Downloads/generator_test/output_retriever_cuda/"
                        cache_dir = "C:/Users/siriv/Downloads/generator_test/other_cache_generator"

                        model_save_name = "bert-base-generator"

                        # train_file = root_path + "dataset/train.json"
                        # valid_file = root_path + "dataset/dev.json"
                        # test_file = root_path + "dataset/test.json"

                        ### files from the retriever results
                        train_file = root_path + "dataset_generator/generator_input_train.json"
                        valid_file = root_path + "dataset_generator/generator_input_dev.json"
                        test_file = conversion_output  # "dataset_generator/generator_input_test.json"

                        # infer table-only text-only
                        # test_file = root_path + "dataset/test_retrieve_7k_text_only.json"

                        op_list_file = "operation_list.txt"
                        const_list_file = "constant_list.txt"

                        # # model choice: bert, roberta, albert
                        pretrained_model = "bert"
                        model_size = "bert-base-uncased"

                        # model choice: bert, roberta, albert
                        # pretrained_model = "roberta"
                        # model_size = "roberta-large"

                        # # finbert
                        # pretrained_model = "finbert"
                        # model_size = root_path + "pre-trained-models/finbert/"

                        # pretrained_model = "longformer"
                        # model_size = "allenai/longformer-base-4096"

                        # single sent or sliding window
                        # single, slide, gold, none
                        retrieve_mode = "single"

                        # use seq program or nested program
                        program_mode = "seq"

                        # train, test, or private
                        # private: for testing private test data
                        device = "cpu"
                        mode = "test"  # train
                        saved_model_path = root_path + "model_generator.pt"    # "roberta-large-gold_20210713020324/saved_model/loads/119/model.pt"
                        build_summary = False

                        sep_attention = True
                        layer_norm = True
                        num_decoder_layers = 1

                        max_seq_length = 512 # 2k for longformer, 512 for others
                        max_program_length = 30
                        n_best_size = 20
                        dropout_rate = 0.1

                        batch_size = 16
                        batch_size_test = 16
                        epoch = 300
                        learning_rate = 2e-5

                        report = 300
                        report_loss = 100

                        max_step_ind = 11





                    # finqa_utils
                    """MathQA utils.
                    """
                    def str_to_num(text):
                        text = text.replace(",", "")
                        try:
                            num = int(text)
                        except ValueError:
                            try:
                                num = float(text)
                            except ValueError:
                                if text and text[-1] == "%":
                                    num = text
                                else:
                                    num = None
                        return num


                    def prog_token_to_indices(prog, numbers, number_indices, max_seq_length,
                                            op_list, op_list_size, const_list,
                                            const_list_size):
                        prog_indices = []
                        for i, token in enumerate(prog):
                            if token in op_list:
                                prog_indices.append(op_list.index(token))
                            elif token in const_list:
                                prog_indices.append(op_list_size + const_list.index(token))
                            else:
                                if token in numbers:
                                    cur_num_idx = numbers.index(token)
                                else:
                                    cur_num_idx = -1
                                    for num_idx, num in enumerate(numbers):
                                        if str_to_num(num) == str_to_num(token):
                                            cur_num_idx = num_idx
                                            break
                                assert cur_num_idx != -1
                                prog_indices.append(op_list_size + const_list_size +
                                                    number_indices[cur_num_idx])
                        return prog_indices


                    def indices_to_prog(program_indices, numbers, number_indices, max_seq_length,
                                        op_list, op_list_size, const_list, const_list_size):
                        prog = []
                        for i, prog_id in enumerate(program_indices):
                            if prog_id < op_list_size:
                                prog.append(op_list[prog_id])
                            elif prog_id < op_list_size + const_list_size:
                                prog.append(const_list[prog_id - op_list_size])
                            else:
                                prog.append(numbers[number_indices.index(prog_id - op_list_size
                                                                        - const_list_size)])
                        return prog


                    class MathQAExample(
                            collections.namedtuple(
                                "MathQAExample",
                                "id original_question question_tokens options answer \
                                numbers number_indices original_program program"
                            )):

                        def convert_single_example(self, *args, **kwargs):
                            return convert_single_mathqa_example(self, *args, **kwargs)


                    class InputFeatures(object):
                        """A single set of features of data."""

                        def __init__(self,
                                    unique_id,
                                    example_index,
                                    tokens,
                                    question,
                                    input_ids,
                                    input_mask,
                                    option_mask,
                                    segment_ids,
                                    options,
                                    answer=None,
                                    program=None,
                                    program_ids=None,
                                    program_weight=None,
                                    program_mask=None):
                            self.unique_id = unique_id
                            self.example_index = example_index
                            self.tokens = tokens
                            self.question = question
                            self.input_ids = input_ids
                            self.input_mask = input_mask
                            self.option_mask = option_mask
                            self.segment_ids = segment_ids
                            self.options = options
                            self.answer = answer
                            self.program = program
                            self.program_ids = program_ids
                            self.program_weight = program_weight
                            self.program_mask = program_mask


                    def tokenize(tokenizer, text, apply_basic_tokenization=False):
                        """Tokenizes text, optionally looking up special tokens separately.

                        Args:
                        tokenizer: a tokenizer from bert.tokenization.FullTokenizer
                        text: text to tokenize
                        apply_basic_tokenization: If True, apply the basic tokenization. If False,
                            apply the full tokenization (basic + wordpiece).

                        Returns:
                        tokenized text.

                        A special token is any text with no spaces enclosed in square brackets with no
                        space, so we separate those out and look them up in the dictionary before
                        doing actual tokenization.
                        """

                        if conf.pretrained_model in ["bert", "finbert"]:
                            _SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
                        elif conf.pretrained_model in ["roberta", "longformer"]:
                            _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

                        tokenize_fn = tokenizer.tokenize
                        if apply_basic_tokenization:
                            tokenize_fn = tokenizer.basic_tokenizer.tokenize

                        tokens = []
                        for token in text.split(" "):
                            if _SPECIAL_TOKENS_RE.match(token):
                                if token in tokenizer.get_vocab():
                                    tokens.append(token)
                                else:
                                    tokens.append(tokenizer.unk_token)
                            else:
                                tokens.extend(tokenize_fn(token))

                        return tokens


                    def _detokenize(tokens):
                        text = " ".join(tokens)

                        text = text.replace(" ##", "")
                        text = text.replace("##", "")

                        text = text.strip()
                        text = " ".join(text.split())
                        return text


                    def program_tokenization(original_program):
                        original_program = original_program.split(', ')
                        program = []
                        for tok in original_program:
                            cur_tok = ''
                            for c in tok:
                                if c == ')':
                                    if cur_tok != '':
                                        program.append(cur_tok)
                                        cur_tok = ''
                                cur_tok += c
                                if c in ['(', ')']:
                                    program.append(cur_tok)
                                    cur_tok = ''
                            if cur_tok != '':
                                program.append(cur_tok)
                        program.append('EOF')
                        return program


                    def convert_single_mathqa_example(example, is_training, tokenizer, max_seq_length,
                                                    max_program_length, op_list, op_list_size,
                                                    const_list, const_list_size,
                                                    cls_token, sep_token):
                        """Converts a single MathQAExample into an InputFeature."""
                        features = []
                        question_tokens = example.question_tokens
                        if len(question_tokens) >  max_seq_length - 2:
                            print("too long")
                            question_tokens = question_tokens[:max_seq_length - 2]
                        tokens = [cls_token] + question_tokens + [sep_token]
                        segment_ids = [0] * len(tokens)

                        input_ids = tokenizer.convert_tokens_to_ids(tokens)


                        input_mask = [1] * len(input_ids)
                        for ind, offset in enumerate(example.number_indices):
                            if offset < len(input_mask):
                                input_mask[offset] = 2
                            else:
                                if is_training == True:
                                    # print("\n")
                                    # print("################")
                                    # print("number not in input")
                                    # print(example.original_question)
                                    # print(tokens)
                                    # print(len(tokens))
                                    # print(example.numbers[ind])
                                    # print(offset)

                                    # invalid example, drop for training
                                    return features

                                # assert is_training == False



                        padding = [0] * (max_seq_length - len(input_ids))
                        input_ids.extend(padding)
                        input_mask.extend(padding)
                        segment_ids.extend(padding)

                        # print(len(input_ids))
                        assert len(input_ids) == max_seq_length
                        assert len(input_mask) == max_seq_length
                        assert len(segment_ids) == max_seq_length

                        number_mask = [tmp - 1 for tmp in input_mask]
                        for ind in range(len(number_mask)):
                            if number_mask[ind] < 0:
                                number_mask[ind] = 0
                        option_mask = [1, 0, 0, 1] + [1] * (len(op_list) + len(const_list) - 4)
                        option_mask = option_mask + number_mask
                        option_mask = [float(tmp) for tmp in option_mask]

                        for ind in range(len(input_mask)):
                            if input_mask[ind] > 1:
                                input_mask[ind] = 1

                        numbers = example.numbers
                        number_indices = example.number_indices
                        program = example.program
                        if program is not None and is_training:
                            program_ids = prog_token_to_indices(program, numbers, number_indices,
                                                                max_seq_length, op_list, op_list_size,
                                                                const_list, const_list_size)
                            program_mask = [1] * len(program_ids)
                            program_ids = program_ids[:max_program_length]
                            program_mask = program_mask[:max_program_length]
                            if len(program_ids) < max_program_length:
                                padding = [0] * (max_program_length - len(program_ids))
                                program_ids.extend(padding)
                                program_mask.extend(padding)
                        else:
                            program = ""
                            program_ids = [0] * max_program_length
                            program_mask = [0] * max_program_length
                        assert len(program_ids) == max_program_length
                        assert len(program_mask) == max_program_length
                        features.append(
                            InputFeatures(
                                unique_id=-1,
                                example_index=-1,
                                tokens=tokens,
                                question=example.original_question,
                                input_ids=input_ids,
                                input_mask=input_mask,
                                option_mask=option_mask,
                                segment_ids=segment_ids,
                                options=example.options,
                                answer=example.answer,
                                program=program,
                                program_ids=program_ids,
                                program_weight=1.0,
                                program_mask=program_mask))
                        return features


                    def read_mathqa_entry(entry, tokenizer):
                        
                        question = entry["qa"]["question"]
                        this_id = entry["id"]
                        context = ""


                        if conf.retrieve_mode == "single":
                            for ind, each_sent in entry["qa"]["model_input"]:
                                context += each_sent
                                context += " "
                        elif conf.retrieve_mode == "slide":
                            if len(entry["qa"]["pos_windows"]) > 0:
                                context = random.choice(entry["qa"]["pos_windows"])[0]
                            else:
                                context = entry["qa"]["neg_windows"][0][0]
                        elif conf.retrieve_mode == "gold":
                            for each_con in entry["qa"]["gold_inds"]:
                                context += entry["qa"]["gold_inds"][each_con]
                                context += " "

                        elif conf.retrieve_mode == "none":
                            # no retriever, use longformer
                            table = entry["table"]
                            table_text = ""
                            for row in table[1:]:
                                this_sent = table_row_to_text(table[0], row)
                                table_text += this_sent

                            context = " ".join(entry["pre_text"]) + " " + " ".join(entry["post_text"]) + " " + table_text

                        context = context.strip()
                        # process "." and "*" in text
                        context = context.replace(". . . . . .", "")
                        context = context.replace("* * * * * *", "")
                            
                        original_question = question + " " + tokenizer.sep_token + " " + context.strip()

                        if "exe_ans" in entry["qa"]:
                            options = entry["qa"]["exe_ans"]
                        else:
                            options = None

                        original_question_tokens = original_question.split(' ')

                        numbers = []
                        number_indices = []
                        question_tokens = []
                        for i, tok in enumerate(original_question_tokens):
                            num = str_to_num(tok)
                            if num is not None:
                                numbers.append(tok)
                                number_indices.append(len(question_tokens))
                                if tok[0] == '.':
                                    numbers.append(str(str_to_num(tok[1:])))
                                    number_indices.append(len(question_tokens) + 1)
                            tok_proc = tokenize(tokenizer, tok)
                            question_tokens.extend(tok_proc)

                        if "exe_ans" in entry["qa"]:
                            answer = entry["qa"]["exe_ans"]
                        else:
                            answer = None

                        # table headers
                        for row in entry["table"]:
                            tok = row[0]
                            if tok and tok in original_question:
                                numbers.append(tok)
                                tok_index = original_question.index(tok)
                                prev_tokens = original_question[:tok_index]
                                number_indices.append(len(tokenize(tokenizer, prev_tokens)) + 1)

                        if conf.program_mode == "seq":
                            if 'program' in entry["qa"]:
                                original_program = entry["qa"]['program']
                                program = program_tokenization(original_program)
                            else:
                                program = None
                                original_program = None
                                
                        elif conf.program_mode == "nest":
                            if 'program_re' in entry["qa"]:
                                original_program = entry["qa"]['program_re']
                                program = program_tokenization(original_program)
                            else:
                                program = None
                                original_program = None
                            
                        else:
                            program = None
                            original_program = None

                        return MathQAExample(
                            id=this_id,
                            original_question=original_question,
                            question_tokens=question_tokens,
                            options=options,
                            answer=answer,
                            numbers=numbers,
                            number_indices=number_indices,
                            original_program=original_program,
                            program=program)





                    # utils
                    # Progress bar

                    all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max",
                            "table_min", "table_sum", "table_average"]



                    def write_word(pred_list, save_dir, name):
                        ss = open(save_dir + name, "w+")
                        for item in pred_list:
                            ss.write(" ".join(item) + '\n')



                    def write_log(log_file, s):
                        pass
                        """"
                        print(s)
                        with open(log_file, 'a') as f:
                            f.write(s+'\n')
                        """


                    def _compute_softmax(scores):
                        """Compute softmax probability over raw logits."""
                        if not scores:
                            return []

                        max_score = None
                        for score in scores:
                            if max_score is None or score > max_score:
                                max_score = score

                        exp_scores = []
                        total_sum = 0.0
                        for score in scores:
                            x = math.exp(score - max_score)
                            exp_scores.append(x)
                            total_sum += x

                        probs = []
                        for score in exp_scores:
                            probs.append(score / total_sum)
                        return probs


                    def read_txt(input_path, log_file):
                        """Read a txt file into a list."""

                        #write_log(log_file, "Reading: %s" % input_path)
                        with open(input_path) as input_file:
                            input_data = input_file.readlines()
                        items = []
                        for line in input_data:
                            items.append(line.strip())
                        return items


                    def read_examples(data_input, tokenizer, op_list, const_list, log_file):
                        """Read a json file into a list of examples."""
                                              
                        #write_log(log_file, "Reading " + "conversion_output")
                        #with open(input_path) as input_file:
                        #    input_data = json.load(input_file)              

                        input_data = data_input

                        examples = []
                        for entry in tqdm(input_data):
                            examples.append(read_mathqa_entry(entry, tokenizer))
                            program = examples[-1].program
                            for tok in program:
                                if 'const_' in tok and not (tok in const_list):
                                    const_list.append(tok)
                                elif '(' in tok and not (tok in op_list):
                                    op_list.append(tok)
                        return input_data, examples, op_list, const_list


                    def convert_examples_to_features(examples,
                                                    tokenizer,
                                                    max_seq_length,
                                                    max_program_length,
                                                    is_training,
                                                    op_list,
                                                    op_list_size,
                                                    const_list,
                                                    const_list_size,
                                                    verbose=True):
                        """Converts a list of DropExamples into InputFeatures."""
                        unique_id = 1000000000
                        res = []
                        for (example_index, example) in enumerate(examples):
                            features = example.convert_single_example(
                                is_training=is_training,
                                tokenizer=tokenizer,
                                max_seq_length=max_seq_length,
                                max_program_length=max_program_length,
                                op_list=op_list,
                                op_list_size=op_list_size,
                                const_list=const_list,
                                const_list_size=const_list_size,
                                cls_token=tokenizer.cls_token,
                                sep_token=tokenizer.sep_token)

                            for feature in features:
                                feature.unique_id = unique_id
                                feature.example_index = example_index
                                res.append(feature)
                                unique_id += 1

                        return res


                    RawResult = collections.namedtuple(
                        "RawResult",
                        "unique_id logits loss")


                    def compute_prog_from_logits(logits, max_program_length, example,
                                                template=None):
                        pred_prog_ids = []
                        op_stack = []
                        loss = 0
                        for cur_step in range(max_program_length):
                            cur_logits = logits[cur_step]
                            cur_pred_softmax = _compute_softmax(cur_logits)
                            cur_pred_token = np.argmax(cur_logits)
                            loss -= np.log(cur_pred_softmax[cur_pred_token])
                            pred_prog_ids.append(cur_pred_token)
                            if cur_pred_token == 0:
                                break
                        return pred_prog_ids, loss


                    def compute_predictions(all_examples, all_features, all_results, n_best_size,
                                            max_program_length, tokenizer, op_list, op_list_size,
                                            const_list, const_list_size):
                        """Computes final predictions based on logits."""
                        example_index_to_features = collections.defaultdict(list)
                        for feature in all_features:
                            example_index_to_features[feature.example_index].append(feature)

                        unique_id_to_result = {}
                        for result in all_results:
                            unique_id_to_result[result.unique_id] = result

                        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                            "PrelimPrediction", [
                                "feature_index", "logits"
                            ])

                        all_predictions = collections.OrderedDict()
                        all_predictions["pred_programs"] = collections.OrderedDict()
                        all_predictions["ref_programs"] = collections.OrderedDict()
                        all_nbest = collections.OrderedDict()
                        for (example_index, example) in enumerate(all_examples):
                            features = example_index_to_features[example_index]
                            prelim_predictions = []
                            for (feature_index, feature) in enumerate(features):
                                result = unique_id_to_result[feature.unique_id]
                                logits = result.logits
                                prelim_predictions.append(
                                    _PrelimPrediction(
                                        feature_index=feature_index,
                                        logits=logits))

                            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                                "NbestPrediction", "options answer program_ids program")

                            nbest = []
                            for pred in prelim_predictions:
                                if len(nbest) >= n_best_size:
                                    break
                                program = example.program
                                pred_prog_ids, loss = compute_prog_from_logits(pred.logits,
                                                                            max_program_length,
                                                                            example)
                                pred_prog = indices_to_prog(pred_prog_ids,
                                                                        example.numbers,
                                                                        example.number_indices,
                                                                        conf.max_seq_length,
                                                                        op_list, op_list_size,
                                                                        const_list, const_list_size
                                                                        )
                                nbest.append(
                                    _NbestPrediction(
                                        options=example.options,
                                        answer=example.answer,
                                        program_ids=pred_prog_ids,
                                        program=pred_prog))

                            assert len(nbest) >= 1

                            nbest_json = []
                            for (i, entry) in enumerate(nbest):
                                output = collections.OrderedDict()
                                output["id"] = example.id
                                output["options"] = entry.options
                                output["ref_answer"] = entry.answer
                                output["pred_prog"] = [str(prog) for prog in entry.program]
                                output["ref_prog"] = example.program
                                output["question_tokens"] = example.question_tokens
                                output["numbers"] = example.numbers
                                output["number_indices"] = example.number_indices
                                nbest_json.append(output)

                            assert len(nbest_json) >= 1

                            all_predictions["pred_programs"][example_index] = nbest_json[0]["pred_prog"]
                            all_predictions["ref_programs"][example_index] = nbest_json[0]["ref_prog"]
                            all_nbest[example_index] = nbest_json

                        return all_predictions, all_nbest


                    def write_predictions(all_predictions, output_prediction_file):
                        """Writes final predictions in json format."""

                        with open(output_prediction_file, "w") as writer:
                            writer.write(json.dumps(all_predictions, indent=4) + "\n")


                    class DataLoader:
                        def __init__(self, is_training, data, reserved_token_size, batch_size=64, shuffle=True):
                            """
                            Main dataloader
                            """
                            self.data = data
                            self.batch_size = batch_size
                            self.is_training = is_training
                            self.data_size = len(data)
                            self.reserved_token_size = reserved_token_size
                            self.num_batches = int(self.data_size / batch_size) if self.data_size % batch_size == 0 \
                                else int(self.data_size / batch_size) + 1
                            if shuffle:
                                self.shuffle_all_data()
                            self.count = 0

                        def __iter__(self):
                            return self

                        def __next__(self):
                            # drop last batch
                            if self.is_training:
                                bound = self.num_batches - 1
                            else:
                                bound = self.num_batches
                            if self.count < bound:
                                return self.get_batch()
                            else:
                                raise StopIteration

                        def __len__(self):
                            return self.num_batches

                        def reset(self):
                            self.count = 0
                            self.shuffle_all_data()

                        def shuffle_all_data(self):
                            random.shuffle(self.data)
                            return

                        def get_batch(self):
                            start_index = self.count * self.batch_size
                            end_index = min((self.count + 1) * self.batch_size, self.data_size)

                            self.count += 1
                            # print (self.count)

                            batch_data = {"unique_id": [],
                                        "example_index": [],
                                        "tokens": [],
                                        "question": [],
                                        "input_ids": [],
                                        "input_mask": [],
                                        "option_mask": [],
                                        "segment_ids": [],
                                        "options": [],
                                        "answer": [],
                                        "program": [],
                                        "program_ids": [],
                                        "program_weight": [],
                                        "program_mask": []}
                            for each_data in self.data[start_index: end_index]:

                                batch_data["option_mask"].append(each_data.option_mask)
                                batch_data["input_mask"].append(each_data.input_mask)

                                batch_data["unique_id"].append(each_data.unique_id)
                                batch_data["example_index"].append(each_data.example_index)
                                batch_data["tokens"].append(each_data.tokens)
                                batch_data["question"].append(each_data.question)
                                batch_data["input_ids"].append(each_data.input_ids)
                                batch_data["segment_ids"].append(each_data.segment_ids)
                                batch_data["options"].append(each_data.options)
                                batch_data["answer"].append(each_data.answer)
                                batch_data["program"].append(each_data.program)
                                batch_data["program_ids"].append(each_data.program_ids)
                                batch_data["program_weight"].append(each_data.program_weight)
                                batch_data["program_mask"].append(each_data.program_mask)

                            return batch_data


                    def cleanhtml(raw_html):
                        cleanr = re.compile('<.*?>')
                        cleantext = re.sub(cleanr, '', raw_html)
                        return cleantext


                    def str_to_num(text):

                        text = text.replace(",", "")
                        try:
                            num = float(text)
                        except ValueError:
                            if "%" in text:
                                text = text.replace("%", "")
                                try:
                                    num = float(text)
                                    num = num / 100.0
                                except ValueError:
                                    num = "n/a"
                            elif "const" in text:
                                text = text.replace("const_", "")
                                if text == "m1":
                                    text = "-1"
                                num = float(text)
                            else:
                                num = "n/a"
                        return num


                    def process_row(row_in):

                        row_out = []
                        invalid_flag = 0

                        for num in row_in:
                            num = num.replace("$", "").strip()
                            num = num.split("(")[0].strip()

                            num = str_to_num(num)

                            if num == "n/a":
                                invalid_flag = 1
                                break

                            row_out.append(num)

                        if invalid_flag:
                            return "n/a"

                        return row_out


                    def reprog_to_seq(prog_in, is_gold):
                        '''
                        predicted recursive program to list program
                        ["divide(", "72", "multiply(", "6", "210", ")", ")"]
                        ["multiply(", "6", "210", ")", "divide(", "72", "#0", ")"]
                        '''

                        st = []
                        res = []

                        try:
                            num = 0
                            for tok in prog_in:
                                if tok != ")":
                                    st.append(tok)
                                else:
                                    this_step_vec = [")"]
                                    for _ in range(3):
                                        this_step_vec.append(st[-1])
                                        st = st[:-1]
                                    res.extend(this_step_vec[::-1])
                                    st.append("#" + str(num))
                                    num += 1
                        except:
                            if is_gold:
                                raise ValueError

                        return res





                    # Model_new
                    if conf.pretrained_model == "bert":
                        from transformers import BertModel
                    elif conf.pretrained_model == "roberta":
                        from transformers import RobertaModel
                    elif conf.pretrained_model == "finbert":
                        from transformers import BertModel
                    elif conf.pretrained_model == "longformer":
                        from transformers import LongformerModel


                    class Bert_model(nn.Module):

                        def __init__(self, num_decoder_layers, hidden_size, dropout_rate, input_length,
                                    program_length, op_list, const_list):

                            super(Bert_model, self).__init__()

                            self.op_list_size = len(op_list)
                            self.const_list_size = len(const_list)
                            self.reserved_token_size = self.op_list_size + self.const_list_size
                            self.program_length = program_length
                            self.hidden_size = hidden_size
                            self.const_list = const_list
                            self.op_list = op_list
                            self.input_length = input_length

                            self.reserved_ind = nn.Parameter(torch.arange(
                                0, self.reserved_token_size), requires_grad=False)
                            self.reserved_go = nn.Parameter(torch.arange(op_list.index(
                                'GO'), op_list.index('GO') + 1), requires_grad=False)

                            self.reserved_para = nn.Parameter(torch.arange(op_list.index(
                                ')'), op_list.index(')') + 1), requires_grad=False)

                            # masking for decoidng for test time
                            op_ones = nn.Parameter(torch.ones(
                                self.op_list_size), requires_grad=False)
                            op_zeros = nn.Parameter(torch.zeros(
                                self.op_list_size), requires_grad=False)
                            other_ones = nn.Parameter(torch.ones(
                                input_length + self.const_list_size), requires_grad=False)
                            other_zeros = nn.Parameter(torch.zeros(
                                input_length + self.const_list_size), requires_grad=False)
                            self.op_only_mask = nn.Parameter(
                                torch.cat((op_ones, other_zeros), 0), requires_grad=False)
                            self.seq_only_mask = nn.Parameter(
                                torch.cat((op_zeros, other_ones), 0), requires_grad=False)

                            # for ")"
                            para_before_ones = nn.Parameter(torch.ones(
                                op_list.index(')')), requires_grad=False)
                            para_after_ones = nn.Parameter(torch.ones(
                                input_length + self.reserved_token_size - op_list.index(')') - 1), requires_grad=False)
                            para_zero = nn.Parameter(torch.zeros(1), requires_grad=False)
                            self.para_mask = nn.Parameter(torch.cat(
                                (para_before_ones, para_zero, para_after_ones), 0), requires_grad=False)

                            # for step embedding
                            # self.step_masks = []
                            all_tmp_list = self.op_list + self.const_list
                            self.step_masks = nn.Parameter(torch.zeros(
                                conf.max_step_ind, input_length + self.reserved_token_size), requires_grad=False)
                            for i in range(conf.max_step_ind):
                                this_step_mask_ind = all_tmp_list.index("#" + str(i))
                                self.step_masks[i, this_step_mask_ind] = 1.0

                            # self.step_mask_eye = torch.eye(conf.max_step_ind)

                            if conf.pretrained_model == "bert":
                                self.bert = BertModel.from_pretrained(
                                    conf.model_size, cache_dir=conf.cache_dir)
                            elif conf.pretrained_model == "roberta":
                                self.bert = RobertaModel.from_pretrained(
                                    conf.model_size, cache_dir=conf.cache_dir)
                            elif conf.pretrained_model == "finbert":
                                self.bert = BertModel.from_pretrained(
                                    conf.model_size, cache_dir=conf.cache_dir)
                            elif conf.pretrained_model == "longformer":
                                self.bert = LongformerModel.from_pretrained(
                                    conf.model_size, cache_dir=conf.cache_dir)

                            self.cls_prj = nn.Linear(hidden_size, hidden_size, bias=True)
                            self.cls_dropout = nn.Dropout(dropout_rate)

                            self.seq_prj = nn.Linear(hidden_size, hidden_size, bias=True)
                            self.seq_dropout = nn.Dropout(dropout_rate)

                            self.reserved_token_embedding = nn.Embedding(
                                self.reserved_token_size, hidden_size)

                            # attentions
                            self.decoder_history_attn_prj = nn.Linear(
                                hidden_size, hidden_size, bias=True)
                            self.decoder_history_attn_dropout = nn.Dropout(dropout_rate)

                            self.question_attn_prj = nn.Linear(hidden_size, hidden_size, bias=True)
                            self.question_attn_dropout = nn.Dropout(dropout_rate)

                            self.question_summary_attn_prj = nn.Linear(
                                hidden_size, hidden_size, bias=True)
                            self.question_summary_attn_dropout = nn.Dropout(dropout_rate)

                            if conf.sep_attention:
                                self.input_embeddings_prj = nn.Linear(
                                    hidden_size*3, hidden_size, bias=True)
                            else:
                                self.input_embeddings_prj = nn.Linear(
                                    hidden_size*2, hidden_size, bias=True)
                            self.input_embeddings_layernorm = nn.LayerNorm([1, hidden_size])

                            self.option_embeddings_prj = nn.Linear(
                                hidden_size*2, hidden_size, bias=True)

                            # decoder lstm
                            self.rnn = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                                                    num_layers=conf.num_decoder_layers, batch_first=True)

                            # step vector
                            self.decoder_step_proj = nn.Linear(
                                3*hidden_size, hidden_size, bias=True)
                            self.decoder_step_proj_dropout = nn.Dropout(dropout_rate)

                            self.step_mix_proj = nn.Linear(
                                hidden_size*2, hidden_size, bias=True)

                        def forward(self, is_training, input_ids, input_mask, segment_ids, option_mask, program_ids, program_mask, device):

                            bert_outputs = self.bert(
                                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

                            bert_sequence_output = bert_outputs.last_hidden_state
                            bert_pooled_output = bert_sequence_output[:, 0, :]
                            batch_size, seq_length, bert_dim = list(bert_sequence_output.size())

                            split_program_ids = torch.split(program_ids, 1, dim=1)
                            # print(self.program_length)
                            # print(program_ids.size())
                            # print(split_program_ids[0].size())

                            pooled_output = self.cls_prj(bert_pooled_output)
                            pooled_output = self.cls_dropout(pooled_output)

                            option_size = self.reserved_token_size + seq_length

                            sequence_output = self.seq_prj(bert_sequence_output)
                            sequence_output = self.seq_dropout(sequence_output)

                            op_embeddings = self.reserved_token_embedding(self.reserved_ind)
                            op_embeddings = op_embeddings.repeat(batch_size, 1, 1)

                            logits = []

                            init_decoder_output = self.reserved_token_embedding(self.reserved_go)
                            decoder_output = init_decoder_output.repeat(batch_size, 1, 1)

                            # [batch, op + seq len, hidden]
                            initial_option_embeddings = torch.cat(
                                [op_embeddings, sequence_output], dim=1)

                            if conf.sep_attention:
                                decoder_history = decoder_output
                            else:
                                decoder_history = torch.unsqueeze(pooled_output, dim=-1)

                            decoder_state_h = torch.zeros(
                                1, batch_size, self.hidden_size, device=device)
                            decoder_state_c = torch.zeros(
                                1, batch_size, self.hidden_size, device=device)

                            float_input_mask = input_mask.float()
                            float_input_mask = torch.unsqueeze(float_input_mask, dim=-1)

                            this_step_new_op_emb = initial_option_embeddings

                            for cur_step in range(self.program_length):

                                # decoder history att
                                decoder_history_attn_vec = self.decoder_history_attn_prj(
                                    decoder_output)
                                decoder_history_attn_vec = self.decoder_history_attn_dropout(
                                    decoder_history_attn_vec)

                                decoder_history_attn_w = torch.matmul(
                                    decoder_history, torch.transpose(decoder_history_attn_vec, 1, 2))
                                decoder_history_attn_w = F.softmax(decoder_history_attn_w, dim=1)

                                decoder_history_ctx_embeddings = torch.matmul(
                                    torch.transpose(decoder_history_attn_w, 1, 2), decoder_history)

                                if conf.sep_attention:
                                    # input seq att
                                    question_attn_vec = self.question_attn_prj(decoder_output)
                                    question_attn_vec = self.question_attn_dropout(
                                        question_attn_vec)

                                    question_attn_w = torch.matmul(
                                        sequence_output, torch.transpose(question_attn_vec, 1, 2))
                                    question_attn_w -= 1e6 * (1 - float_input_mask)
                                    question_attn_w = F.softmax(question_attn_w, dim=1)

                                    question_ctx_embeddings = torch.matmul(
                                        torch.transpose(question_attn_w, 1, 2), sequence_output)

                                # another input seq att
                                question_summary_vec = self.question_summary_attn_prj(
                                    decoder_output)
                                question_summary_vec = self.question_summary_attn_dropout(
                                    question_summary_vec)

                                question_summary_w = torch.matmul(
                                    sequence_output, torch.transpose(question_summary_vec, 1, 2))
                                question_summary_w -= 1e6 * (1 - float_input_mask)
                                question_summary_w = F.softmax(question_summary_w, dim=1)

                                question_summary_embeddings = torch.matmul(
                                    torch.transpose(question_summary_w, 1, 2), sequence_output)

                                if conf.sep_attention:
                                    concat_input_embeddings = torch.cat([decoder_history_ctx_embeddings,
                                                                        question_ctx_embeddings,
                                                                        decoder_output], dim=-1)
                                else:
                                    concat_input_embeddings = torch.cat([decoder_history_ctx_embeddings,
                                                                        decoder_output], dim=-1)

                                input_embeddings = self.input_embeddings_prj(
                                    concat_input_embeddings)

                                if conf.layer_norm:
                                    input_embeddings = self.input_embeddings_layernorm(
                                        input_embeddings)

                                question_option_vec = this_step_new_op_emb * question_summary_embeddings
                                option_embeddings = torch.cat(
                                    [this_step_new_op_emb, question_option_vec], dim=-1)

                                option_embeddings = self.option_embeddings_prj(option_embeddings)
                                option_logits = torch.matmul(
                                    option_embeddings, torch.transpose(input_embeddings, 1, 2))
                                option_logits = torch.squeeze(
                                    option_logits, dim=2)  # [batch, op + seq_len]
                                option_logits -= 1e6 * (1 - option_mask)
                                logits.append(option_logits)

                                if is_training:
                                    program_index = torch.unsqueeze(
                                        split_program_ids[cur_step], dim=1)
                                else:
                                    # constrain decoding
                                    if cur_step % 4 == 0 or (cur_step + 1) % 4 == 0:
                                        # op round
                                        option_logits -= 1e6 * self.seq_only_mask
                                    else:
                                        # number round
                                        option_logits -= 1e6 * self.op_only_mask

                                    if (cur_step + 1) % 4 == 0:
                                        # ")" round
                                        option_logits -= 1e6 * self.para_mask
                                        # print(program_index)

                                    program_index = torch.argmax(
                                        option_logits, axis=-1, keepdim=True)

                                    program_index = torch.unsqueeze(
                                        program_index, dim=1
                                    )

                                if (cur_step + 1) % 4 == 0:

                                    # update op embeddings
                                    this_step_index = cur_step // 4
                                    this_step_list_index = (
                                        self.op_list + self.const_list).index("#" + str(this_step_index))
                                    this_step_mask = self.step_masks[this_step_index, :]

                                    decoder_step_vec = self.decoder_step_proj(
                                        concat_input_embeddings)
                                    decoder_step_vec = self.decoder_step_proj_dropout(
                                        decoder_step_vec)
                                    decoder_step_vec = torch.squeeze(decoder_step_vec)

                                    this_step_new_emb = decoder_step_vec  # [batch, hidden]

                                    this_step_new_emb = torch.unsqueeze(this_step_new_emb, 1)
                                    this_step_new_emb = this_step_new_emb.repeat(
                                        1, self.reserved_token_size+self.input_length, 1)  # [batch, op seq, hidden]

                                    this_step_mask = torch.unsqueeze(
                                        this_step_mask, 0)  # [1, op seq]
                                    # print(this_step_mask)

                                    this_step_mask = torch.unsqueeze(
                                        this_step_mask, 2)  # [1, op seq, 1]
                                    this_step_mask = this_step_mask.repeat(
                                        batch_size, 1, self.hidden_size)  # [batch, op seq, hidden]

                                    this_step_new_op_emb = torch.where(
                                        this_step_mask > 0, this_step_new_emb, initial_option_embeddings)

                                # print(program_index.size())
                                program_index = torch.repeat_interleave(
                                    program_index, self.hidden_size, dim=2)  # [batch, 1, hidden]

                                input_program_embeddings = torch.gather(
                                    option_embeddings, dim=1, index=program_index)

                                decoder_output, (decoder_state_h, decoder_state_c) = self.rnn(
                                    input_program_embeddings, (decoder_state_h, decoder_state_c))
                                decoder_history = torch.cat(
                                    [decoder_history, input_program_embeddings], dim=1)

                            logits = torch.stack(logits, dim=1)
                            return logits



                    #st.write(conversion_output)

                    # generator main
                    # test
                    #!/usr/bin/env python
                    # -*- coding: utf-8 -*-

                    if conf.pretrained_model == "bert":
                        print("Using bert")
                        from transformers import BertTokenizer
                        from transformers import BertConfig
                        tokenizer = BertTokenizer.from_pretrained(conf.model_size)
                        model_config = BertConfig.from_pretrained(conf.model_size)

                    elif conf.pretrained_model == "roberta":
                        print("Using roberta")
                        from transformers import RobertaTokenizer
                        from transformers import RobertaConfig
                        tokenizer = RobertaTokenizer.from_pretrained(conf.model_size)
                        model_config = RobertaConfig.from_pretrained(conf.model_size)

                    elif conf.pretrained_model == "finbert":
                        print("Using finbert")
                        from transformers import BertTokenizer
                        from transformers import BertConfig
                        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                        model_config = BertConfig.from_pretrained(conf.model_size)

                    elif conf.pretrained_model == "longformer":
                        print("Using longformer")
                        from transformers import LongformerTokenizer, LongformerConfig
                        tokenizer = LongformerTokenizer.from_pretrained(conf.model_size)
                        model_config = LongformerConfig.from_pretrained(conf.model_size)


                    saved_model_path = "C:/Users/siriv/Downloads/generator_test/"
                    # model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
                    model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S") + \
                        "_" + conf.model_save_name
                    model_dir = os.path.join(
                        conf.output_path, 'inference_only_' + model_dir_name)
                    results_path = os.path.join(model_dir, "results")
                    os.makedirs(results_path, exist_ok=False)
                    log_file = os.path.join(results_path, 'log.txt')


                    op_list = read_txt(conf.op_list_file, log_file)
                    op_list = [op + '(' for op in op_list]
                    op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
                    const_list = read_txt(conf.const_list_file, log_file)
                    const_list = [const.lower().replace('.', '_') for const in const_list]
                    reserved_token_size = len(op_list) + len(const_list)


                    test_data, test_examples, op_list, const_list = \
                        read_examples(conversion_output, tokenizer=tokenizer,
                                    op_list=op_list, const_list=const_list, log_file=log_file)


                    kwargs = {"examples": test_examples,
                            "tokenizer": tokenizer,
                            "max_seq_length": conf.max_seq_length,
                            "max_program_length": conf.max_program_length,
                            "is_training": False,
                            "op_list": op_list,
                            "op_list_size": len(op_list),
                            "const_list": const_list,
                            "const_list_size": len(const_list),
                            "verbose": True}


                    test_features = convert_examples_to_features(**kwargs)


                    def generate(data_ori, data, model, ksave_dir, mode='valid'):

                        pred_list = []
                        pred_unk = []

                        ksave_dir_mode = os.path.join(ksave_dir, mode)
                        os.makedirs(ksave_dir_mode, exist_ok=True)

                        data_iterator = DataLoader(
                            is_training=False, data=data, batch_size=conf.batch_size_test, reserved_token_size=reserved_token_size, shuffle=False)

                        k = 0
                        all_results = []
                        with torch.no_grad():
                            for x in tqdm(data_iterator):

                                input_ids = x['input_ids']
                                input_mask = x['input_mask']
                                segment_ids = x['segment_ids']
                                program_ids = x['program_ids']
                                program_mask = x['program_mask']
                                option_mask = x['option_mask']

                                ori_len = len(input_ids)
                                for each_item in [input_ids, input_mask, segment_ids, program_ids, program_mask, option_mask]:
                                    if ori_len < conf.batch_size_test:
                                        each_len = len(each_item[0])
                                        pad_x = [0] * each_len
                                        each_item += [pad_x] * (conf.batch_size_test - ori_len)

                                input_ids = torch.tensor(input_ids).to(conf.device)
                                input_mask = torch.tensor(input_mask).to(conf.device)
                                segment_ids = torch.tensor(segment_ids).to(conf.device)
                                program_ids = torch.tensor(program_ids).to(conf.device)
                                program_mask = torch.tensor(program_mask).to(conf.device)
                                option_mask = torch.tensor(option_mask).to(conf.device)

                                logits = model(False, input_ids, input_mask,
                                            segment_ids, option_mask, program_ids, program_mask, device=conf.device)

                                for this_logit, this_id in zip(logits.tolist(), x["unique_id"]):
                                    all_results.append(
                                        RawResult(
                                            unique_id=int(this_id),
                                            logits=this_logit,
                                            loss=None
                                        ))

                        output_prediction_file = "prediction.json" #os.path.join(ksave_dir_mode,
                                                            #"predictions.json")
                        output_nbest_file = "nbest_predictions.json"
                        output_eval_file = "evals.json"

                        all_predictions, all_nbest = compute_predictions(
                            data_ori,
                            data,
                            all_results,
                            n_best_size=conf.n_best_size,
                            max_program_length=conf.max_program_length,
                            tokenizer=tokenizer,
                            op_list=op_list,
                            op_list_size=len(op_list),
                            const_list=const_list,
                            const_list_size=len(const_list))
                        print("answer: ", all_predictions["pred_programs"])
                        #write_predictions(all_predictions, output_prediction_file)
                        #write_predictions(all_nbest, output_nbest_file)

                        model_output = all_predictions

                        return model_output


                    def generate_test():
                        model = Bert_model(num_decoder_layers=conf.num_decoder_layers,
                                        hidden_size=model_config.hidden_size,
                                        dropout_rate=conf.dropout_rate,
                                        program_length=conf.max_program_length,
                                        input_length=conf.max_seq_length,
                                        op_list=op_list,
                                        const_list=const_list)
                        model = nn.DataParallel(model)
                        model.to(conf.device)
                        model.load_state_dict(torch.load("model_generator.pt", map_location = torch.device('cpu')))  # changeable
                        model.eval()
                        model_output = generate(test_examples, test_features, model, results_path, mode='test')
                        
                        return model_output
                        


                    model_output = generate_test()

                    st.write(model_output["pred_programs"])

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
                graphic_no_find = Image.open("racoon/racoon_5.png")
                st.image(graphic_no_find)
    else:
        st.write("Please enter a query into the intended text-box to initiate a search.")
    done = 1
    if done == 1:
        # winter-theme if it is december
        if datetime.now().month == 12:
            st.snow()  # winter-theme
        # notification-sound (when computation finished)
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load("w_short.mp3")
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
graphic_1 = Image.open("vide_graphic_1.png")
st.image(graphic_1)


