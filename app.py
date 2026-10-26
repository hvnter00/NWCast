from pathlib import Path

import streamlit as st
import about
import admin
import home
import prediksi


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Newcastle Coal Prediction",
    page_icon="📈",
    layout="wide",
)

st.html(BASE_DIR / "Assets" / "styles.css")

page = st.sidebar.radio(
    "Menu", ["Home", "Prediksi", "Admin", "About"], label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.caption("Newcastle Coal Prediction App (Proof of Concept) - 2026")

PAGES = {
    "Home": home.render,
    "Prediksi": prediksi.render,
    "Admin": admin.render,
    "About": about.render,
}
PAGES[page]()
