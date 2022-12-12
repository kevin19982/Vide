# page 2

# load packages
import streamlit as st
from PIL import Image

# load pictures
logo_3 = Image.open("C:/Users/siriv/Downloads/Data_Science_Project/scripts/logo_temp/vide_design_3.png")
im = Image.open("C:/Users/siriv/Downloads/Data_Science_Project/scripts/examples/hello.jpg")

# title
st.title("It's literally only a cat")

# sidebar name of page
with st.sidebar:
    st.image(logo_3)
st.sidebar.markdown("# Only a cat")

# display picture
st.image(im)
