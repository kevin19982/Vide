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
    st.title("Contacts")

# sidebar name of page
with st.sidebar:
    st.image(logo_3)
st.sidebar.markdown("# Contacts")

# create two columns
col1, col2 = st.columns(2)

with col1:
    st.image(logo_3_200px)  # display logo
with col2:
    st.markdown("### Business inquiries:")  # display information
    st.write("Kevin Kopp: kevin.kopp1998@gmail.com")
    st.write("Felix Froschauer: felix.froschauer@gmx.de")
    
