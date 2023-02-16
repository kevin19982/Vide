# page 2

# load packages
import streamlit as st
from PIL import Image


# load pictures
logo_3 = Image.open("C:/Users/siriv/Downloads/vide_qa_model/graphics/logo_design_3.png")


# title
st.title("Tutorial")


# sidebar name of page
with st.sidebar:
    st.image(logo_3)
st.sidebar.markdown("# Tutorial")



st.write("Tutorial for how to use the app:")
st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


