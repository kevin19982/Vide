# page 2

import streamlit as st

st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# Page 2 ❄️")

from PIL import Image
im = Image.open("hello.jpg")
st.image(im)
