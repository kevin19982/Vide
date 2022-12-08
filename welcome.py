# pages

# load streamlit package
import streamlit as st

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

st.write("Welcome to Vide - Question and Answering, a platform to extract relevant information from financial documents. We aim to deliver fast and concise answers to any question you have regarding recent developments in the financial world.")

st.write("To use the app, please advance to the second page and follow the instructions displayed on the page.")

from PIL import Image
logo_1 = Image.open("logo_temp/vide_design_1.png")
logo_2 = Image.open("logo_temp/vide_design_2.png")
logo_3 = Image.open("logo_temp/vide_design_3.png")
logo_4 = Image.open("logo_temp/vide_design_4.png")
logo_5 = Image.open("logo_temp/vide_design_5.png")
logo_6 = Image.open("logo_temp/vide_design_6.png")
logo_7 = Image.open("logo_temp/vide_design_7.png")
logo_8 = Image.open("logo_temp/vide_design_8.png")

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.image(logo_1)
    st.image(logo_7)
with col2:
    st.image(logo_2)
    st.image(logo_8)
with col3:
    st.image(logo_3)
with col4:
    st.image(logo_4)
with col5:
    st.image(logo_5)
with col6:
    st.image(logo_6)
