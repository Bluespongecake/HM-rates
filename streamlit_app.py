"""Main entry point for the HM multi-page Streamlit app."""

import streamlit as st

st.set_page_config(page_title="HM Discounts Explorer", page_icon="🏨", layout="wide")

st.title("HM Rates Explorer")
st.caption("Use the page selector in the sidebar to jump between hotel searches and event overviews.")

st.markdown(
    """
Welcome! Pick a page on the left to get started:

- **Hotels** lets you search Expedia for discounted properties around a point or event.
- **Events** summarises cached deals across all stored events.
"""
)
