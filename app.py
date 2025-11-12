# app.py

import streamlit as st
from shared_utils import device # Import device for display

st.set_page_config(page_title="DetAny3D Streamlit Demo", layout="wide")

st.title("DetAny3D: Detect Anything 3D")
st.markdown(f"**Running on Device:** `{device}`") 

st.sidebar.header("Navigation")
st.sidebar.info("Select a dataset page to start detection.")

st.warning("Please navigate using the sidebar to select the nuScenes or NOA page.")