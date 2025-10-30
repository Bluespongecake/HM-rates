"""Events page for the HM multi-page Streamlit app."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import pandas as pd
import pydeck as pdk
import streamlit as st

from expedia.helpers.rates_report import (
    DEFAULT_RATE_TYPES,
    RATE_TYPE_LABELS,
    evaluate_deal_quality,
)

CACHE_PATH = Path("reports/events_with_hotels.json")
DEFAULT_MIN_DISCOUNT = 10.0
DEFAULT_MIN_SAVINGS = 100.0
DEFAULT_MAX_TOTAL = 1000.0
DEFAULT_RADIUS_FILTER_KM = None
MAP_CENTER = (20.0, 0.0)
ALL_COUNTRIES_OPTION = "All countries"


def _normalize_country(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else "Unknown"
    cleaned = str(value).strip()
    return cleaned if cleaned else "Unknown"


def _load_cached_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(data, list):
        return [dict(record) for record in data if isinstance(record, Mapping)]
    if isinstance(data, Mapping):
        return [dict(data)]
    return []


def _parse_date(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def _event_length_days(checkin: Optional[datetime], checkout: Optional[datetime]) -> Optional[int]:
    if checkin is None or checkout is None:
        return None
    delta = (checkout.date() - checkin.date()).days
    if delta < 1:
        return 1
    return delta


def _evaluate_event(
    event: Mapping[str, Any],
    *,
    discount_threshold_pct: float,
    savings_threshold: float,
    max_total_cost: float,
    radius_filter_km: Optional[float] = None,
) -> Dict[str, Any]:
    hotels = list(event.get("hotels") or [])
    df = pd.DataFrame(hotels)

    if radius_filter_km is not None and not df.empty and "Distance to Center (km)" in df.columns:
        distances = pd.to_numeric(df["Distance to Center (km)"], errors="coerce")
        df = df.loc[distances <= float(radius_filter_km)].reset_index(drop=True)

    rate_meta = event.get("rates_search") or {}
    rate_types: Sequence[str] = tuple(rate_meta.get("rate_types") or DEFAULT_RATE_TYPES)
    rate_type_labels: Mapping[str, str] = {**RATE_TYPE_LABELS, **(rate_meta.get("rate_type_labels") or {})}

    if df.empty:
        return {
            "is_good_deal": False,
            "reason": "No hotels in cache within the selected radius." if radius_filter_km is not None else "No cached hotels.",
            "best_property": None,
            "best_discount_pct": None,
            "best_savings": None,
            "best_total_cost": None,
            "qualifying_count": 0,
            "qualifying_properties": [],
            "rate_types": list(rate_types),
            "rate_type_labels": dict(rate_type_labels),
            "filtered_hotel_count": 0,
        }

    evaluation = evaluate_deal_quality(
        df,
        rate_types=tuple(rate_types),
        rate_type_labels=rate_type_labels,
        discount_threshold_pct=float(discount_threshold_pct),
        savings_threshold=float(savings_threshold),
        max_total_cost=float(max_total_cost),
    )
    evaluation.update(
        {
            "rate_types": list(rate_types),
            "rate_type_labels": dict(rate_type_labels),
            "filtered_hotel_count": len(df),
        }
    )
    return evaluation


def _build_event_rows(
    events: Iterable[Mapping[str, Any]],
    *,
    discount_threshold_pct: float,
    savings_threshold: float,
    max_total_cost: float,
    radius_filter_km: Optional[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in events:
        latitude = event.get("latitude")
        longitude = event.get("longitude")
        if pd.isna(latitude) or pd.isna(longitude):
            continue

        evaluation = _evaluate_event(
            event,
            discount_threshold_pct=discount_threshold_pct,
            savings_threshold=savings_threshold,
            max_total_cost=max_total_cost,
            radius_filter_km=radius_filter_km,
        )

        start_date = _parse_date(event.get("checkin"))
        end_date = _parse_date(event.get("checkout"))
        duration_days = _event_length_days(start_date, end_date)

        rows.append(
            {
                "title": event.get("title"),
                "city": event.get("city"),
                "country": _normalize_country(event.get("country")),
                "map_key": event.get("map_key"),
                "venue_id": event.get("venue_id"),
                "latitude": float(latitude),
                "longitude": float(longitude),
                "checkin": start_date.strftime("%Y-%m-%d") if start_date else None,
                "checkout": end_date.strftime("%Y-%m-%d") if end_date else None,
                "duration_days": duration_days,
                "is_good_deal": evaluation["is_good_deal"],
                "reason": evaluation["reason"],
                "best_property": evaluation.get("best_property"),
                "best_discount_pct": evaluation.get("best_discount_pct"),
                "best_savings": evaluation.get("best_savings"),
                "best_total_cost": evaluation.get("best_total_cost"),
                "qualifying_count": evaluation.get("qualifying_count", 0),
                "qualifying_properties": evaluation.get("qualifying_properties", []),
                "filtered_hotel_count": evaluation.get("filtered_hotel_count", 0),
            }
        )
    return rows


def _build_map(rows: Sequence[Mapping[str, Any]]) -> Optional[pdk.Deck]:
    if not rows:
        return None
    map_df = pd.DataFrame(rows)
    if map_df.empty:
        return None

    def _color(is_good: bool) -> List[int]:
        return [34, 139, 34] if is_good else [220, 20, 60]

    map_df["color"] = map_df["is_good_deal"].apply(_color)
    map_df["tooltip"] = map_df.apply(
        lambda row: f"{row['title'] or 'Event'}\nBest: {row['best_property'] or 'N/A'}", axis=1
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius=400,
        pickable=True,
        radius_scale=1,
        radius_min_pixels=5,
        radius_max_pixels=15,
    )
    view_state = pdk.ViewState(
        latitude=map_df["latitude"].mean() if not map_df["latitude"].isna().all() else MAP_CENTER[0],
        longitude=map_df["longitude"].mean() if not map_df["longitude"].isna().all() else MAP_CENTER[1],
        zoom=2.5,
    )
    mapbox_token = ""
    if "mapbox_token" in st.secrets:
        mapbox_token = st.secrets.get("mapbox_token", "") or ""

    deck_kwargs = {
        "layers": [layer],
        "initial_view_state": view_state,
        "tooltip": {"text": "{tooltip}"},
    }
    if mapbox_token:
        deck_kwargs.update(
            {
                "map_style": "mapbox://styles/mapbox/light-v10",
                "api_keys": {"mapbox": mapbox_token},
                "map_provider": "mapbox",
            }
        )
    else:
        deck_kwargs.update({"map_style": "light", "map_provider": "carto"})

    return pdk.Deck(**deck_kwargs)


def render() -> None:
    """Render the events page."""
    st.title("Event Deals Overview")
    st.caption("All North Am events in HM system")

    events = _load_cached_events(CACHE_PATH)
    if not events:
        st.warning(
            "No cached events found. Run `update_event_rates_cache` first to populate "
            f"{CACHE_PATH} with hotel data."
        )
        st.stop()

    available_countries: Set[str] = {_normalize_country(event.get("country")) for event in events}
    country_options = sorted(available_countries) or ["Unknown"]
    country_selector_options: List[str] = [ALL_COUNTRIES_OPTION, *country_options]
    selected_countries = st.sidebar.multiselect(
        "Countries",
        options=country_selector_options,
        default=[ALL_COUNTRIES_OPTION],
        help="Keep 'All countries' selected to include everything, or remove it to pick specific countries.",
        placeholder="Select countries…",
    )

    if not selected_countries or ALL_COUNTRIES_OPTION in selected_countries:
        selected_countries = country_options

    if not selected_countries:
        st.info("Select at least one country to display events.")
        st.stop()

    st.sidebar.header("Event Duration")
    min_duration, max_duration = st.sidebar.slider(
        "Length of event (days)",
        min_value=1,
        max_value=30,
        value=(1, 7),
        step=1,
    )

    st.sidebar.header("Deal Criteria")
    min_discount = st.sidebar.slider(
        "Minimum discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=DEFAULT_MIN_DISCOUNT,
        step=1.0,
    )
    min_savings = st.sidebar.number_input(
        "Minimum savings ($)",
        min_value=0.0,
        value=DEFAULT_MIN_SAVINGS,
        step=25.0,
    )
    max_total = st.sidebar.number_input(
        "Maximum total cost ($)",
        min_value=0.0,
        value=DEFAULT_MAX_TOTAL,
        step=25.0,
    )
    radius_filter = st.sidebar.slider(
        "Radius filter (km)",
        min_value=0.0,
        value=DEFAULT_RADIUS_FILTER_KM if DEFAULT_RADIUS_FILTER_KM is not None else 0.0,
        step=0.5,
        max_value=10.0,
        help="Only consider hotels within this distance of the event. Set to 0 to disable.",
    )
    radius_filter_km = None if radius_filter == 0 else float(radius_filter)

    filtered_events = [
        event for event in events if _normalize_country(event.get("country")) in set(selected_countries)
    ]

    if not filtered_events:
        st.info("No events match the selected countries.")
        st.stop()

    rows = _build_event_rows(
        filtered_events,
        discount_threshold_pct=min_discount,
        savings_threshold=min_savings,
        max_total_cost=max_total,
        radius_filter_km=radius_filter_km,
    )

    duration_filtered_rows = [
        row
        for row in rows
        if row.get("duration_days") is None
        or (row["duration_days"] >= min_duration and row["duration_days"] <= max_duration)
    ]

    rows = duration_filtered_rows

    if not rows:
        st.info("No events match the filters or have cached hotels.")
        st.stop()

    deck = _build_map(rows)
    if deck is not None:
        st.pydeck_chart(deck, use_container_width=True)

    total_events = len(rows)
    qualified_events = sum(1 for row in rows if row.get("is_good_deal"))
    percent_qualified = (qualified_events / total_events * 100.0) if total_events else 0.0
    percent_display = f"{percent_qualified:.0f}" if percent_qualified >= 1 else f"{percent_qualified:.1f}"
    st.markdown(
        f"**{qualified_events} of {total_events} ({percent_display}%)** of events have at least one suitable deal"
    )

    st.subheader("Event Details")
    table_df = pd.DataFrame(rows)
    table_df["Deal status"] = table_df["is_good_deal"].map({True: "Good", False: "Needs review"})
    table_df["Qualifying properties"] = table_df["qualifying_properties"].apply(lambda values: ", ".join(values[:5]))
    display_cols = [
        "title",
        "city",
        "country",
        "checkin",
        "checkout",
        "Deal status",
        "best_property",
        "best_discount_pct",
        "best_savings",
        "best_total_cost",
        "qualifying_count",
        "Qualifying properties",
        "reason",
    ]
    st.dataframe(table_df[display_cols], use_container_width=True)
