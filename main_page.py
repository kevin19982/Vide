# pages

# load packages
import streamlit as st

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

from PIL import Image
im = Image.open("hello.jpg")
st.image(im)
