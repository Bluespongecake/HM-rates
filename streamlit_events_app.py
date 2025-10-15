"""Legacy single-page entry point for the events dashboard."""

import streamlit as st

from pages_hm_rates_app.events import render


def main() -> None:
    st.set_page_config(page_title="Event Deals Overview", page_icon="🗺️", layout="wide")
    render()


if __name__ == "__main__":
    main()
