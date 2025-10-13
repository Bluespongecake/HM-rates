"""Streamlit interface for exploring hotel rates near a coordinate."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Iterable, Tuple

import pandas as pd
import streamlit as st

from expedia.client import ExpediaAPIError, ExpediaClient
from expedia.helpers import circle_polygon_geojson, extract_cheapest_rates_by_id


DEFAULT_OCCUPANCY = 2
DEFAULT_RATE_TYPE = "mkt_prepay"  # public package rates
DEFAULT_STAY_NIGHTS = 1
SEARCH_POINT_DENSITY = 96  # polygon resolution for geography search


@st.cache_resource(show_spinner=False)
def get_client() -> ExpediaClient:
    """Instantiate a cached Expedia client using environment credentials."""
    return ExpediaClient.from_env()


def _property_name(data: Dict[str, Any]) -> str:
    """Best-effort extraction of the property name field."""
    name = data.get("name")
    if isinstance(name, dict):
        if "content" in name and isinstance(name["content"], str):
            return name["content"]
        if "value" in name and isinstance(name["value"], str):
            return name["value"]
    if isinstance(name, str):
        return name
    # Some responses nest the actual value inside content.value
    content = name.get("content") if isinstance(name, dict) else None
    if isinstance(content, dict) and isinstance(content.get("value"), str):
        return content["value"]
    return data.get("property_name", "")


def _safe_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _property_location(data: Dict[str, Any]) -> Tuple[float | None, float | None]:
    """Return latitude/longitude if available."""
    coords = (
        data.get("location", {})
        if isinstance(data.get("location"), dict)
        else {}
    )
    coordinates = coords.get("coordinates") if isinstance(coords, dict) else {}
    lat = _safe_float(coordinates.get("latitude"))
    lon = _safe_float(coordinates.get("longitude"))
    return (lat, lon)


def _batched_fetch(
    client: ExpediaClient,
    ids: Iterable[str],
    checkin: str,
    checkout: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fetch availability and property summaries in batches."""
    ids_list = list(ids)
    availability: Dict[str, Any] = {}
    summaries: Dict[str, Any] = {}

    def fetch_availability(batch):
        return client.fetch_availability(
            batch,
            checkin,
            checkout,
            DEFAULT_OCCUPANCY,
            DEFAULT_RATE_TYPE,
            rate_plan_count=1,
        )

    def fetch_summaries(batch):
        return client.fetch_property_summaries(
            batch,
            includes=("name", "location"),
        )

    client.run_batched(
        ids_list,
        jobs=[(fetch_availability, availability)],
        batch_size=250,
    )
    client.run_batched(
        ids_list,
        jobs=[(fetch_summaries, summaries)],
        batch_size=250,
    )
    return availability, summaries


st.set_page_config(page_title="Hotel Rates Explorer", page_icon="🏨", layout="wide")
st.title("Hotel Rates Near a Coordinate")
st.caption(
    "Enter a latitude, longitude, and radius (km) to look up the cheapest available room "
    "rate for each property returned by Expedia EPS Rapid."
)

with st.form("search_form"):
    lat = st.number_input("Latitude", value=52.370216, format="%.6f")
    lon = st.number_input("Longitude", value=4.895168, format="%.6f")
    radius_km = st.number_input("Radius (km)", min_value=1, max_value=50, value=3, step=1)
    submitted = st.form_submit_button("Find hotel prices")

if submitted:
    with st.spinner("Contacting Expedia..."):
        try:
            client = get_client()
            polygon = circle_polygon_geojson(
                center_lat=lat,
                center_lon=lon,
                radius_m=radius_km * 1000,
                n_points=SEARCH_POINT_DENSITY,
            )
            property_ids = client.search_geography(
                polygon,
                include="property_ids",
                supply_source="expedia",
            )

            if not property_ids:
                st.info("No properties found within that radius.")
            else:
                checkin_date = date.today() + timedelta(days=30)
                checkout_date = checkin_date + timedelta(days=DEFAULT_STAY_NIGHTS)
                checkin = checkin_date.isoformat()
                checkout = checkout_date.isoformat()

                availability, summaries = _batched_fetch(client, property_ids, checkin, checkout)
                rate_map = extract_cheapest_rates_by_id(
                    availability,
                    occupancy_key=str(DEFAULT_OCCUPANCY),
                    include_room_name=True,
                )
                if not rate_map:
                    st.warning("Rates were not returned for the properties found.")
                else:
                    rows = []
                    for pid, value in rate_map.items():
                        room_name, price = value if isinstance(value, tuple) else (None, value)
                        if price is None:
                            continue
                        summary = summaries.get(pid, {})
                        name = _property_name(summary) or pid
                        lat_lon = _property_location(summary)
                        rows.append(
                            {
                                "Property": name,
                                "Property ID": pid,
                                "Room": room_name or "",
                                "Price (request currency)": float(price),
                                "latitude": lat_lon[0],
                                "longitude": lat_lon[1],
                            }
                        )

                    if rows:
                        df = pd.DataFrame(rows).sort_values("Price (request currency)")
                        st.success(f"Found {len(df)} priced properties.")

                        st.subheader("Results")
                        st.dataframe(df, use_container_width=True)

                        st.subheader("Map view")
                        map_data = df.dropna(subset=["latitude", "longitude"])
                        if map_data.empty:
                            st.info("No coordinates available to plot.")
                        else:
                            st.map(map_data[["latitude", "longitude"]])

                    else:
                        st.warning("No priced offers were returned in this area.")

        except ExpediaAPIError as exc:
            st.error(f"Expedia API error: {exc}")
        except Exception as exc:  # pragma: no cover - surfaced for visibility in UI
            st.exception(exc)

st.caption(
    "The app queries public package rates (`mkt_prepay`) for stays starting 30 days from today "
    f"and lasting {DEFAULT_STAY_NIGHTS} night(s). Adjust the script if you need different rate types "
    "or travel dates."
)
