# page 2

# load packages
import streamlit as st
from PIL import Image


# define base path
base_path = ""
# load pictures
logo_3 = Image.open(base_path + "graphis/logo_design_3.png")


# title
st.title("Tutorial")


# sidebar name of page
with st.sidebar:
    st.image(logo_3)
st.sidebar.markdown("# tutorial")



st.write("Tutorial for how to use the app:")
st.video("https://www.youtube.com/watch?v=_MCTX-l_tgg")

st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
graphic_1 = Image.open(base_path + "graphics/example_graphic.png")
st.image(graphic_1)
