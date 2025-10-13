"""Streamlit interface for exploring hotel rates near a coordinate."""

from datetime import date, timedelta
from typing import Mapping

import pandas as pd
import streamlit as st

# Ensure python-dotenv absence does not break imports on Streamlit Cloud.
try:  # pragma: no cover - import guard for deployment environments
    import dotenv  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - handled in production
    import sys
    import types

    dotenv = types.ModuleType("dotenv")

    def _load_dotenv_stub(*args, **kwargs):  # type: ignore[unused-arg]
        return None

    dotenv.load_dotenv = _load_dotenv_stub  # type: ignore[attr-defined]
    sys.modules["dotenv"] = dotenv

from expedia.client import ExpediaAPIError, ExpediaClient
from expedia.helpers import fetch_rates_near_coordinate


DEFAULT_OCCUPANCY = 2
DEFAULT_RATE_TYPE = "mkt_prepay"  # public package rates
DEFAULT_STAY_NIGHTS = 1
SEARCH_POINT_DENSITY = 96  # polygon resolution for geography search
DEFAULT_LATITUDE = 52.370216
DEFAULT_LONGITUDE = 4.895168
DEFAULT_RADIUS_KM = 3
DEFAULT_CHECKIN_OFFSET_DAYS = 30
DEFAULT_CHECKIN_DATE = date.today() + timedelta(days=DEFAULT_CHECKIN_OFFSET_DAYS)
DEFAULT_CHECKOUT_DATE = DEFAULT_CHECKIN_DATE + timedelta(days=DEFAULT_STAY_NIGHTS)


def _load_expedia_credentials() -> Mapping[str, str]:
    """Read Expedia credentials from Streamlit secrets."""
    try:
        secrets = st.secrets["expedia"]
    except KeyError as exc:  # pragma: no cover - configuration error surfaced to user
        raise RuntimeError(
            "Missing 'expedia' section in Streamlit secrets. "
            "Add api_key, shared_secret, and api_base to deploy the app."
        ) from exc

    required = ("api_key", "shared_secret", "api_base")
    missing = [key for key in required if not secrets.get(key)]
    if missing:
        raise RuntimeError(
            f"Missing required Expedia secrets: {', '.join(missing)}. "
            "Populate them in .streamlit/secrets.toml or the Streamlit deployment settings."
        )
    return secrets


@st.cache_resource(show_spinner=False)
def get_client() -> ExpediaClient:
    """Instantiate a cached Expedia client using credentials from Streamlit secrets."""
    creds = _load_expedia_credentials()
    return ExpediaClient(
        creds["api_key"],
        creds["shared_secret"],
        creds["api_base"],
    )


st.set_page_config(page_title="Hotel Rates Explorer", page_icon="🏨", layout="wide")
st.title("Hotel Rates Near a Coordinate")
st.caption(
    "Enter a latitude, longitude, and radius (km) to look up the cheapest available room "
    "rate for each property returned by Expedia EPS Rapid."
)

# --- Session state defaults -------------------------------------------------
st.session_state.setdefault("latitude", DEFAULT_LATITUDE)
st.session_state.setdefault("longitude", DEFAULT_LONGITUDE)
st.session_state.setdefault("radius_km", DEFAULT_RADIUS_KM)
st.session_state.setdefault("checkin", DEFAULT_CHECKIN_DATE)
st.session_state.setdefault("checkout", DEFAULT_CHECKOUT_DATE)
st.session_state.setdefault("search_result", None)
st.session_state.setdefault("search_error", None)

st.subheader("Search Parameters")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.number_input(
        "Latitude",
        key="latitude",
        format="%.6f",
    )
with col2:
    st.number_input(
        "Longitude",
        key="longitude",
        format="%.6f",
    )
with col3:
    st.slider(
        "Radius (km)",
        min_value=1,
        max_value=50,
        step=1,
        key="radius_km",
    )

dates_col1, dates_col2 = st.columns(2)
with dates_col1:
    st.date_input("Check-in", key="checkin")
with dates_col2:
    st.date_input("Check-out", key="checkout")

submit = st.button("Find hotel prices", type="primary")

if submit:
    st.session_state["search_error"] = None
    checkin_date = st.session_state["checkin"]
    checkout_date = st.session_state["checkout"]

    if checkout_date <= checkin_date:
        st.session_state["search_error"] = "Check-out date must be after the check-in date."
        st.session_state["search_result"] = None
    else:
        stay_nights = (checkout_date - checkin_date).days
        with st.spinner("Contacting Expedia..."):
            try:
                client = get_client()
                result = fetch_rates_near_coordinate(
                    client,
                    center_lat=float(st.session_state["latitude"]),
                    center_lon=float(st.session_state["longitude"]),
                    radius_km=int(st.session_state["radius_km"]),
                    stay_nights=stay_nights,
                    occupancy=DEFAULT_OCCUPANCY,
                    rate_type=DEFAULT_RATE_TYPE,
                    n_points=SEARCH_POINT_DENSITY,
                    checkin=checkin_date.isoformat(),
                    checkout=checkout_date.isoformat(),
                )
                st.session_state["search_result"] = result
            except ExpediaAPIError as exc:
                st.session_state["search_error"] = f"Expedia API error: {exc}"
                st.session_state["search_result"] = None
            except Exception as exc:  # pragma: no cover - surfaced for visibility in UI
                st.session_state["search_error"] = str(exc)
                st.session_state["search_result"] = None

result = st.session_state.get("search_result")
error_message = st.session_state.get("search_error")

st.subheader("Map")
selected_point = pd.DataFrame(
    [
        {
            "latitude": float(st.session_state["latitude"]),
            "longitude": float(st.session_state["longitude"]),
            "label": "Selected coordinate",
        }
    ]
)

if result:
    hotels_geo = result["dataframe"].dropna(subset=["latitude", "longitude"])[["latitude", "longitude"]].copy()
    if not hotels_geo.empty:
        hotels_geo["label"] = "Hotel"
        map_points = pd.concat([selected_point, hotels_geo], ignore_index=True)
    else:
        map_points = selected_point
else:
    map_points = selected_point

st.map(map_points[["latitude", "longitude"]], use_container_width=True)

if error_message:
    st.error(error_message)
elif result:
    property_ids = result["property_ids"]
    if not property_ids:
        st.info("No properties found within that radius.")
    else:
        df = result["dataframe"]
        if df.empty:
            st.warning("Rates were not returned for the properties found.")
        else:
            st.success(f"Found {len(df)} priced properties.")

            st.subheader("Results")
            st.dataframe(df, use_container_width=True)

st.caption(
    "The app queries public package rates (`mkt_prepay`) for stays starting 30 days from today "
    f"and lasting {DEFAULT_STAY_NIGHTS} night(s). Adjust the script if you need different rate types "
    "or travel dates."
)
