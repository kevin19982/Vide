# page 3

# load packages
import streamlit as st
from PIL import Image

# load pictures
logo_3 = Image.open("logo_temp/vide_design_3.png")
logo_3_200px = Image.open("logo_temp/vide_design_3_200px.png")

# title (placed in center)
header_1, header_2, header_3 = st.columns(3)
with header_2:
    st.title("About")

# sidebar name of page
with st.sidebar:
    st.image(logo_3)
st.sidebar.markdown("# about")

# create two columns
col1, col2 = st.columns(2)

with col1:
    st.image(logo_3_200px)  # display logo
with col2:
    st.markdown("### Business inquiries:")  # display information
    st.write("Kevin Kopp: kevin.kopp1998@gmail.com")
    st.write("Felix Froschauer: felix.froschauer@gmx.de")


# empty spacese for spacing
st.write(" ")
st.write(" ")
st.write(" ")
st.write("This application is based on and the underlying model was trained on the FinQA dataset:")
st.write("Title: FinQA: A Dataset of Numerical Reasoning over Financial Data")
st.write("Author: Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan R and others")
st.write("Booktitle: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing")
st.write("Pages: 3697--3711")
st.write("Year: 2021")

st.write(" ")
st.write(" ")
st.write(" ")
st.write("The all-MiniLM-L6_v2 model is also used for the application.")

st.write(" ")
st.write(" ")
st.write(" ")
meme_1 = Image.open("logo_temp/meme_1.png")
st.image(meme_1)



