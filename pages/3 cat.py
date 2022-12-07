# page 3

import streamlit as st

st.markdown("# Page 3 🎉")
st.sidebar.markdown("# Page 3 🎉")

from PIL import Image
im = Image.open("hello.jpg")
st.image(im)
