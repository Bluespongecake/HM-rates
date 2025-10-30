"""Hotels page for the HM multi-page Streamlit app."""

from datetime import date, timedelta
from pathlib import Path
from typing import List, Mapping

import pandas as pd
import pydeck as pdk
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

from events.event_helpers import load_events_catalog
from expedia.client import ExpediaAPIError, ExpediaClient
from expedia.helpers.rates_report import (
    CACHE_DISTANCE_TOLERANCE_KM,
    DEFAULT_OCCUPANCY,
    DEFAULT_RATE_TYPES,
    RATE_TYPE_LABELS,
    SEARCH_POINT_DENSITY,
    evaluate_deal_quality,
    find_cached_rates_dataframe,
    generate_rates_dataframe,
)


DEFAULT_STAY_NIGHTS = 1
DEFAULT_LATITUDE = 52.370216
DEFAULT_LONGITUDE = 4.895168
DEFAULT_RADIUS_KM = 3
DEFAULT_CHECKIN_OFFSET_DAYS = 30
DEFAULT_CHECKIN_DATE = date.today() + timedelta(days=DEFAULT_CHECKIN_OFFSET_DAYS)
DEFAULT_CHECKOUT_DATE = DEFAULT_CHECKIN_DATE + timedelta(days=DEFAULT_STAY_NIGHTS)
EVENT_DATA_DIRECTORIES: List[Path] = [Path("data"), Path("events/data")]
DEFAULT_MAP_STYLE = "mapbox://styles/mapbox/streets-v11"
CACHED_RATES_PATH = Path("reports/events_with_hotels.json")


@st.cache_data(show_spinner=False)
def get_events_catalog() -> pd.DataFrame:
    """Cached wrapper around event catalog loading."""
    return load_events_catalog(EVENT_DATA_DIRECTORIES)


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


def render() -> None:
    """Render the hotels page."""
    st.title("Non-contracted Hotel Discounts near an Event")
    st.caption(
        "Find an event in the drop-down search tool to find the highest dollar discounts. "
        "Or manually enter coordinates and dates below"
    )

    # --- Session state defaults -------------------------------------------------
    st.session_state.setdefault("latitude", DEFAULT_LATITUDE)
    st.session_state.setdefault("longitude", DEFAULT_LONGITUDE)
    st.session_state.setdefault("radius_km", DEFAULT_RADIUS_KM)
    st.session_state.setdefault("checkin", DEFAULT_CHECKIN_DATE)
    st.session_state.setdefault("checkout", DEFAULT_CHECKOUT_DATE)
    st.session_state.setdefault("search_result", None)
    st.session_state.setdefault("search_error", None)
    st.session_state.setdefault("deal_min_discount_pct", 10.0)
    st.session_state.setdefault("deal_min_savings", 100.0)
    st.session_state.setdefault("deal_max_total_cost", 2000.0)
    st.session_state.setdefault("selected_event_details", None)
    st.session_state.setdefault("map_filter_min_discount_pct", 5.0)
    st.session_state.setdefault("map_filter_max_total_cost", 0.0)
    st.session_state.setdefault("filter_hotels_only", False)

    events_catalog = get_events_catalog()

    if not events_catalog.empty:
        st.subheader("Event Search")
        with st.expander("Find an event to prefill search details", expanded=False):
            query = st.text_input(
                "Search by event name, city, map key, or venue ID",
                key="event_search_query",
                placeholder="Type to filter events…",
            )
            filtered_events = events_catalog
            if query:
                filtered_events = events_catalog[events_catalog["search_blob"].str.contains(query.lower(), na=False)]

            match_count = len(filtered_events)
            st.caption(f"{match_count} event{'s' if match_count != 1 else ''} found.")

            selected_event = None
            if match_count:
                option_labels = filtered_events["display_label"].tolist()
                selected_label = st.selectbox(
                    "Matching events",
                    option_labels,
                    key="event_selectbox",
                )
                selected_event = filtered_events.loc[
                    filtered_events["display_label"] == selected_label
                ].iloc[0]

                info_lines = [
                    f"- **Location:** {selected_event.get('city') or 'Unknown'}"
                    + (f", {selected_event.get('country')}" if selected_event.get("country") else ""),
                    f"- **Venue ID:** {selected_event.get('venue_id') or 'N/A'}",
                    f"- **Map key:** {selected_event.get('map_key') or 'N/A'}",
                    f"- **Dates:** {selected_event['date_start'].strftime('%d %b %Y')}"
                    + (
                        f" → {selected_event['date_end'].strftime('%d %b %Y')}"
                        if pd.notna(selected_event["date_end"])
                        else ""
                    ),
                ]
                st.markdown("\n".join(info_lines))

                if st.button("Use this event", key="use_selected_event"):
                    lat = selected_event.get("latitude")
                    lon = selected_event.get("longitude")
                    if pd.isna(lat) or pd.isna(lon):
                        st.warning("Selected event is missing latitude/longitude; please pick another event.")
                    else:
                        start_ts = selected_event["date_start"]
                        end_ts = selected_event["date_end"]
                        if pd.isna(start_ts):
                            st.warning("Selected event is missing a start date; please choose another event.")
                        else:
                            start_date = start_ts.date()
                            end_date = end_ts.date() if pd.notna(end_ts) else start_date
                            if end_date < start_date:
                                end_date = start_date

                            checkin_date = start_date - timedelta(days=1)
                            checkout_date = end_date
                            if checkout_date <= checkin_date:
                                checkout_date = checkin_date + timedelta(days=1)

                            st.session_state["latitude"] = float(lat)
                            st.session_state["longitude"] = float(lon)
                            st.session_state["checkin"] = checkin_date
                            st.session_state["checkout"] = checkout_date
                            st.session_state["selected_event_details"] = {
                                "title": selected_event.get("title"),
                                "map_key": selected_event.get("map_key"),
                                "venue_id": selected_event.get("venue_id"),
                                "city": selected_event.get("city"),
                                "country": selected_event.get("country"),
                                "latitude": float(lat),
                                "longitude": float(lon),
                                "checkin": checkin_date.isoformat(),
                                "checkout": checkout_date.isoformat(),
                            }
                            st.success(f"Loaded event '{selected_event['title']}'.")
                            st.rerun()
            else:
                st.info("No events match that search. Try a different keyword.")
    else:
        st.info("Add event data (JSON or Excel) to `data/` or `testing/data/` to enable quick event searches.")

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

    st.subheader("Deal Criteria")
    criteria_col1, criteria_col2, criteria_col3 = st.columns(3)
    with criteria_col1:
        st.number_input(
            "Min discount (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            key="deal_min_discount_pct",
            help="Minimum percentage discount required to consider the event a good deal.",
        )
    with criteria_col2:
        st.number_input(
            "Min savings ($)",
            min_value=0.0,
            step=10.0,
            key="deal_min_savings",
            help="Minimum dollar savings compared to the reference rate.",
        )
    with criteria_col3:
        st.number_input(
            "Max total cost ($)",
            min_value=0.0,
            step=50.0,
            key="deal_max_total_cost",
            help="Upper limit for the total price of the best deal.",
        )

    submit = st.button("Find hotel prices", type="primary")

    if submit:
        st.session_state["search_error"] = None
        checkin_date = st.session_state["checkin"]
        checkout_date = st.session_state["checkout"]

        if checkout_date <= checkin_date:
            st.session_state["search_error"] = "Check-out date must be after the check-in date."
            st.session_state["search_result"] = None
        else:
            checkin_str = checkin_date.isoformat()
            checkout_str = checkout_date.isoformat()
            cache_lookup_kwargs = {
                "cache_path": CACHED_RATES_PATH,
                "latitude": float(st.session_state["latitude"]),
                "longitude": float(st.session_state["longitude"]),
                "checkin": checkin_str,
                "checkout": checkout_str,
                "radius_km": float(st.session_state["radius_km"]),
                "coordinate_tolerance_km": CACHE_DISTANCE_TOLERANCE_KM,
            }
            selected_event_details = st.session_state.get("selected_event_details") or {}
            cache_lookup_kwargs.update(
                {
                    "map_key": selected_event_details.get("map_key"),
                    "venue_id": selected_event_details.get("venue_id"),
                    "title": selected_event_details.get("title"),
                }
            )

            cache_result = None
            try:
                cache_result = find_cached_rates_dataframe(**cache_lookup_kwargs)
            except Exception:
                cache_result = None

            if cache_result:
                dataframe, metadata = cache_result
                metadata_copy = dict(metadata)
                st.session_state["search_result"] = {
                    "dataframe": dataframe.copy(),
                    "rate_type_labels": metadata_copy.get("rate_type_labels", RATE_TYPE_LABELS),
                    "rate_types": metadata_copy.get("rate_types", list(DEFAULT_RATE_TYPES)),
                    "property_ids": metadata_copy.get("property_ids", []),
                    "checkin": metadata_copy.get("checkin", checkin_str),
                    "checkout": metadata_copy.get("checkout", checkout_str),
                    "metadata": metadata_copy,
                    "source": metadata_copy.get("source", "cache"),
                    "is_cached": True,
                }
                st.session_state["search_error"] = None
                st.session_state["last_checkin"] = checkin_str
                st.session_state["last_checkout"] = checkout_str
            else:
                with st.spinner("Contacting Expedia..."):
                    try:
                        client = get_client()
                        dataframe, metadata = generate_rates_dataframe(
                            client,
                            latitude=float(st.session_state["latitude"]),
                            longitude=float(st.session_state["longitude"]),
                            radius_km=int(st.session_state["radius_km"]),
                            checkin=checkin_str,
                            checkout=checkout_str,
                            occupancy=DEFAULT_OCCUPANCY,
                            rate_types=DEFAULT_RATE_TYPES,
                            n_points=SEARCH_POINT_DENSITY,
                            rate_type_labels=RATE_TYPE_LABELS,
                        )
                        st.session_state["search_result"] = {
                            "dataframe": dataframe,
                            "rate_type_labels": metadata.get("rate_type_labels", RATE_TYPE_LABELS),
                            "rate_types": metadata.get("rate_types", list(DEFAULT_RATE_TYPES)),
                            "property_ids": metadata.get("property_ids", []),
                            "checkin": metadata.get("checkin", checkin_str),
                            "checkout": metadata.get("checkout", checkout_str),
                            "metadata": metadata,
                            "source": metadata.get("source", "api"),
                            "is_cached": False,
                        }
                        st.session_state["last_checkin"] = checkin_str
                        st.session_state["last_checkout"] = checkout_str
                    except ExpediaAPIError as exc:
                        st.session_state["search_error"] = f"Expedia API error: {exc}"
                        st.session_state["search_result"] = None
                    except Exception as exc:  # pragma: no cover - surfaced for visibility in UI
                        st.session_state["search_error"] = str(exc)
                        st.session_state["search_result"] = None

    result = st.session_state.get("search_result")
    error_message = st.session_state.get("search_error")

    df = pd.DataFrame()
    rate_labels = {}
    rate_types_used: List[str] = []
    if result:
        df = result["dataframe"].copy()
        rate_labels = result.get("rate_type_labels", {})
        rate_types_used = result.get("rate_types", [])

    hotels_only = bool(st.session_state.get("filter_hotels_only"))
    if hotels_only and not df.empty:
        if "type" in df.columns:
            filtered = df["type"].astype(str).str.lower() == "hotel"
            df = df.loc[filtered].reset_index(drop=True)
        else:
            st.warning("Property type information is unavailable; displaying all properties.")
            hotels_only = False

    deal_assessment = None
    if not df.empty:
        deal_assessment = evaluate_deal_quality(
            df,
            rate_types=tuple(rate_types_used),
            rate_type_labels=rate_labels,
            discount_threshold_pct=float(st.session_state["deal_min_discount_pct"]),
            savings_threshold=float(st.session_state["deal_min_savings"]),
            max_total_cost=float(st.session_state["deal_max_total_cost"]),
        )

    if error_message:
        st.error(error_message)
    elif deal_assessment:
        st.subheader("Deal Evaluation")
        if deal_assessment["is_good_deal"]:
            st.success(deal_assessment["reason"])
        else:
            st.warning(deal_assessment["reason"])

        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

        qualifying_count = deal_assessment.get("qualifying_count")
        if qualifying_count is None or pd.isna(qualifying_count):
            qualifying_count_display = "0"
        else:
            qualifying_count_display = f"{int(qualifying_count):d}"
        metrics_col1.metric("Hotels Meeting Criteria", qualifying_count_display)

        best_discount = deal_assessment.get("best_discount_pct")
        metrics_col2.metric("Best Discount (%)", f"{best_discount:.1f}%" if best_discount is not None else "N/A")

        best_savings = deal_assessment.get("best_savings")
        metrics_col3.metric("Savings ($)", f"${best_savings:,.2f}" if best_savings is not None else "N/A")

        best_total_cost = deal_assessment.get("best_total_cost")
        metrics_col4.metric("Total Cost ($)", f"${best_total_cost:,.2f}" if best_total_cost is not None else "N/A")

        best_property = deal_assessment.get("best_property")
        if best_property:
            st.caption(f"Best deal property: **{best_property}**")
        qualifying_properties = deal_assessment.get("qualifying_properties") or []
        if qualifying_properties:
            qualifier_list = ", ".join(qualifying_properties[:5])
            if len(qualifying_properties) > 5:
                qualifier_list += ", ..."
            st.caption("Qualifying properties: " + qualifier_list)

    if result and result.get("source") == "cache":
        cached_radius = None
        requested_radius = None
        metadata = result.get("metadata") or {}
        cached_radius_value = metadata.get("cached_radius_km")
        requested_radius_value = metadata.get("requested_radius_km")
        if cached_radius_value is not None:
            cached_radius = float(cached_radius_value)
        if requested_radius_value is not None:
            requested_radius = float(requested_radius_value)

        # if cached_radius is not None and requested_radius is not None and cached_radius > requested_radius:
        #     st.info(
        #         "Showing cached rates filtered to the requested radius "
        #         f"({requested_radius:.1f} km) from a wider cached search ({cached_radius:.1f} km)."
        #     )
        # else:
        #     st.info("Showing cached rates from the saved database (no API call).")

    st.subheader("Map")

    discount_col = "Discount (%)"
    primary_price_col = None
    if rate_types_used:
        primary_label = rate_labels.get(rate_types_used[0], rate_types_used[0])
        candidate_col = f"Price ({primary_label})"
        if candidate_col in df.columns:
            primary_price_col = candidate_col
    if primary_price_col is None and "Price (request currency)" in df.columns:
        primary_price_col = "Price (request currency)"

    map_min_discount_filter = float(st.session_state.get("map_filter_min_discount_pct", 0.0) or 0.0)
    raw_map_max_total_cost = st.session_state.get("map_filter_max_total_cost", 0.0)
    try:
        map_max_total_cost_value = float(raw_map_max_total_cost)
    except (TypeError, ValueError):
        map_max_total_cost_value = 0.0
    map_max_total_cost_filter = map_max_total_cost_value if map_max_total_cost_value > 0 else None

    selected_point = pd.DataFrame(
        [
            {
                "latitude": float(st.session_state["latitude"]),
                "longitude": float(st.session_state["longitude"]),
                "category": "Event",
                "tooltip": "Selected event location",
            }
        ]
    )

    map_points = selected_point
    if result:
        hotel_columns = ["latitude", "longitude"]
        if "Property" in df.columns:
            hotel_columns.append("Property")
        if discount_col in df.columns:
            hotel_columns.append(discount_col)
        if primary_price_col and primary_price_col in df.columns:
            hotel_columns.append(primary_price_col)

        hotels_geo = df.dropna(subset=["latitude", "longitude"])[hotel_columns].copy()
        if not hotels_geo.empty:
            hotels_geo["category"] = "Hotel"
            if discount_col in hotels_geo.columns:
                hotels_geo[discount_col] = pd.to_numeric(hotels_geo[discount_col], errors="coerce")
            if primary_price_col and primary_price_col in hotels_geo.columns:
                hotels_geo[primary_price_col] = pd.to_numeric(hotels_geo[primary_price_col], errors="coerce")

            if discount_col in hotels_geo.columns and map_min_discount_filter > 0:
                hotels_geo = hotels_geo[hotels_geo[discount_col] >= map_min_discount_filter]
            if (
                primary_price_col
                and primary_price_col in hotels_geo.columns
                and map_max_total_cost_filter is not None
            ):
                hotels_geo = hotels_geo[hotels_geo[primary_price_col] <= map_max_total_cost_filter]

            if not hotels_geo.empty:
                def _format_tooltip(row):
                    parts = []
                    property_name = str(row.get("Property", "") or "").strip()
                    if property_name:
                        parts.append(property_name)
                    discount_value = row.get(discount_col)
                    if discount_value is not None and not pd.isna(discount_value):
                        parts.append(f"Discount: {discount_value:.1f}%")
                    if primary_price_col and primary_price_col in row and not pd.isna(row.get(primary_price_col)):
                        parts.append(f"Total: ${row[primary_price_col]:,.0f}")
                    if not parts:
                        parts.append(f"{row['latitude']}, {row['longitude']}")
                    return " • ".join(parts)

                hotels_geo["tooltip"] = hotels_geo.apply(_format_tooltip, axis=1)

                def _color_for_discount(value):
                    color_stops = [
                        (0.0, (220, 20, 60)),
                        (5.0, (255, 140, 0)),
                        (20.0, (34, 139, 34)),
                    ]
                    if value is None or pd.isna(value):
                        return (128, 128, 128)
                    if value <= color_stops[0][0]:
                        return color_stops[0][1]
                    for idx in range(1, len(color_stops)):
                        upper_threshold, upper_color = color_stops[idx]
                        lower_threshold, lower_color = color_stops[idx - 1]
                        if value <= upper_threshold:
                            span = upper_threshold - lower_threshold
                            fraction = 0.0 if span == 0 else (value - lower_threshold) / span
                            return tuple(
                                int(round(lower_channel + fraction * (upper_channel - lower_channel)))
                                for lower_channel, upper_channel in zip(lower_color, upper_color)
                            )
                    return color_stops[-1][1]

                if discount_col in hotels_geo.columns:
                    colors = hotels_geo[discount_col].apply(_color_for_discount)
                    color_df = pd.DataFrame(
                        colors.tolist(),
                        columns=["color_r", "color_g", "color_b"],
                        index=hotels_geo.index,
                    )
                    hotels_geo = hotels_geo.join(color_df)
                if not {"color_r", "color_g", "color_b"}.issubset(hotels_geo.columns):
                    hotels_geo[["color_r", "color_g", "color_b"]] = [220, 20, 60]

                map_points = pd.concat([selected_point, hotels_geo], ignore_index=True)

    if not map_points.empty:
        event_points = map_points[map_points["category"] == "Event"]
        hotel_points = map_points[map_points["category"] == "Hotel"]

        layers = []
        view_state = pdk.ViewState(
            latitude=map_points["latitude"].mean(),
            longitude=map_points["longitude"].mean(),
            zoom=12 if len(map_points) > 1 else 13,
            pitch=0,
        )

        if not event_points.empty:
            base_event_radius = 120
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=event_points,
                    get_position="[longitude, latitude]",
                    get_fill_color=[30, 144, 255],
                    get_radius=base_event_radius,
                    radius_scale=1,
                    radius_min_pixels=10,
                    radius_max_pixels=20,
                    pickable=True,
                )
            )

        if not hotel_points.empty:
            base_hotel_radius = 80
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=hotel_points,
                    get_position="[longitude, latitude]",
                    get_fill_color="[color_r, color_g, color_b]",
                    get_radius=base_hotel_radius,
                    radius_scale=1,
                    radius_min_pixels=4,
                    radius_max_pixels=10,
                    pickable=True,
                )
            )

        mapbox_token = ""
        if "mapbox_token" in st.secrets:
            mapbox_token = st.secrets.get("mapbox_token", "") or ""

        api_keys = {"mapbox": mapbox_token} if mapbox_token else None
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style=DEFAULT_MAP_STYLE if mapbox_token else "light",
            api_keys=api_keys,
            map_provider="mapbox" if mapbox_token else "carto",
            tooltip={"text": "{tooltip}"},
        )
        st.pydeck_chart(deck, use_container_width=True)
    else:
        st.map(map_points[["latitude", "longitude"]], use_container_width=True)

    with st.container():
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            st.number_input(
                "Hide hotels with total cost over ($)",
                min_value=0.0,
                step=50.0,
                key="map_filter_max_total_cost",
                help="Set to 0 to show all hotels regardless of total cost.",
            )
        with filter_col2:
            st.number_input(
                "Hide hotels with discount under (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key="map_filter_min_discount_pct",
                help="Set to 0 to show all hotels regardless of discount.",
            )
        st.checkbox(
            "Only show hotels",
            key="filter_hotels_only",
            help="Limit results to properties explicitly classified as hotels.",
        )

    if not error_message and result:
        property_ids = result["property_ids"]
        if not property_ids:
            st.info("No properties found within that radius.")
        else:
            if df.empty:
                if hotels_only:
                    st.warning("No hotel properties match the current filters.")
                else:
                    st.warning("Rates were not returned for the properties found.")
            else:
                st.success(f"Found {len(df)} priced properties.")

                st.subheader("Results")
                st.dataframe(df, use_container_width=True)

    st.caption(
        "A basic app to find deals hotels are offering us. Compares market rates (`mkt_prepay`) and private package rates (`priv_pkg`) "
        "output is pretty basic right now but it is a proof of concept "
        "Made by Max McCormack"
    )
