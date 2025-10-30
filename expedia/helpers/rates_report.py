"""Utility functions to build rate comparison tables outside the Streamlit app."""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd

from ..client import ExpediaAPIError, ExpediaClient
from ..geo_helpers import haversine_distance
from . import fetch_rates_near_coordinate

DEFAULT_OCCUPANCY = 2
DEFAULT_RATE_TYPES: Sequence[str] = ("mkt_prepay", "priv_pkg")
RATE_TYPE_LABELS: Mapping[str, str] = {
    "mkt_prepay": "Public rate",
    "priv_pkg": "Private package",
}
SEARCH_POINT_DENSITY = 96
CACHE_DISTANCE_TOLERANCE_KM = 0.5

def _try_parse_date(value: Any) -> Optional[date]:
    """Convert a date/ISO string to a date object when possible."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value.split("T")[0], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _normalize_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped.lower() if stripped else None
    return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_date(value: date | str, *, field: str) -> Tuple[date, str]:
    """Ensure we have both a date object and ISO string representation."""
    if isinstance(value, date):
        return value, value.isoformat()
    if isinstance(value, str):
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
        return parsed, value
    raise TypeError(f"{field} must be a date or YYYY-MM-DD string.")


def _normalize_column_name(name: Any) -> Optional[str]:
    """Normalize a column label to support matching with minor formatting differences."""
    if name is None:
        return None
    if isinstance(name, str):
        return re.sub(r"[^a-z0-9]+", "", name.lower())
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _resolve_dataframe_column(df: pd.DataFrame, target: str) -> Optional[str]:
    """
    Return the actual dataframe column whose normalized form matches the supplied target label.
    """
    if target in df.columns:
        return target

    normalized_target = _normalize_column_name(target)
    if not normalized_target:
        return None

    for column in df.columns:
        if _normalize_column_name(column) == normalized_target:
            return column
    return None


def _apply_discount_columns(df: pd.DataFrame, rate_labels: Mapping[str, str], rate_types: Sequence[str]) -> pd.DataFrame:
    """Mirror the Streamlit discount calculation between the first two rate types."""
    if len(rate_types) < 2:
        return df

    primary_rate = rate_types[0]
    secondary_rate = rate_types[1]
    primary_label = rate_labels.get(primary_rate, primary_rate)
    secondary_label = rate_labels.get(secondary_rate, secondary_rate)
    primary_price_col = f"Price ({primary_label})"
    secondary_price_col = f"Price ({secondary_label})"
    primary_price_field = _resolve_dataframe_column(df, primary_price_col)
    secondary_price_field = _resolve_dataframe_column(df, secondary_price_col)

    if primary_price_field is None or secondary_price_field is None:
        return df

    delta_col = f"Delta ({primary_label} - {secondary_label})"
    delta_field = _resolve_dataframe_column(df, delta_col) or delta_col
    discount_col = "Discount (%)"
    discount_field = _resolve_dataframe_column(df, discount_col) or discount_col

    df = df.copy()
    primary_prices = pd.to_numeric(df[primary_price_field], errors="coerce")
    secondary_prices = pd.to_numeric(df[secondary_price_field], errors="coerce")
    df[delta_field] = primary_prices - secondary_prices
    df[discount_field] = (df[delta_field] / primary_prices) * 100
    df.loc[~(primary_prices > 0), discount_field] = None
    df = df.sort_values(by=discount_field, ascending=False, na_position="last").reset_index(drop=True)
    return df


def generate_rates_dataframe(
    client: ExpediaClient,
    *,
    latitude: float,
    longitude: float,
    radius_km: float,
    checkin: date | str,
    checkout: date | str,
    occupancy: int = DEFAULT_OCCUPANCY,
    rate_types: Optional[Sequence[str]] = None,
    n_points: int = SEARCH_POINT_DENSITY,
    rate_type_labels: Optional[Mapping[str, str]] = None,
) -> Tuple[pd.DataFrame, MutableMapping[str, object]]:
    """
    Fetch Expedia rates for the supplied coordinate window and build the comparison dataframe.

    Parameters mirror the Streamlit flow so notebook scripts can call this helper and then display
    or export the resulting dataframe.
    """
    checkin_date, checkin_iso = _coerce_date(checkin, field="checkin")
    checkout_date, checkout_iso = _coerce_date(checkout, field="checkout")
    stay_nights = (checkout_date - checkin_date).days
    if stay_nights <= 0:
        raise ValueError("checkout must be after checkin.")

    selected_rate_types = tuple(rate_types or DEFAULT_RATE_TYPES)
    if not selected_rate_types:
        raise ValueError("At least one rate type must be supplied.")

    rate_labels = dict(RATE_TYPE_LABELS)
    if rate_type_labels:
        rate_labels.update({k: v for k, v in rate_type_labels.items() if isinstance(v, str)})

    result = fetch_rates_near_coordinate(
        client,
        center_lat=float(latitude),
        center_lon=float(longitude),
        radius_km=float(radius_km),
        stay_nights=stay_nights,
        occupancy=int(occupancy),
        rate_type=selected_rate_types,
        n_points=int(n_points),
        checkin=checkin_iso,
        checkout=checkout_iso,
        rate_type_labels=rate_labels,
    )

    df = result["dataframe"].copy()
    df = _apply_discount_columns(
        df,
        result.get("rate_type_labels", rate_labels),
        result.get("rate_types", selected_rate_types),
    )

    distance_col = "Distance to Center (km)"
    if {"latitude", "longitude"}.issubset(df.columns):
        center_lat = float(latitude)
        center_lon = float(longitude)
        latitudes = pd.to_numeric(df["latitude"], errors="coerce")
        longitudes = pd.to_numeric(df["longitude"], errors="coerce")

        distances_km = [
            haversine_distance(center_lat, center_lon, lat, lon) / 1000.0
            if pd.notna(lat) and pd.notna(lon)
            else None
            for lat, lon in zip(latitudes, longitudes)
        ]
        df[distance_col] = distances_km
    else:
        df[distance_col] = None

    metadata: MutableMapping[str, object] = {
        "checkin": result.get("checkin", checkin_iso),
        "checkout": result.get("checkout", checkout_iso),
        "rate_types": result.get("rate_types", list(selected_rate_types)),
        "rate_type_labels": result.get("rate_type_labels", rate_labels),
        "property_ids": result.get("property_ids", []),
        "polygon": result.get("polygon"),
    }
    return df, metadata


def generate_rates_payload(
    client: ExpediaClient,
    *,
    latitude: float,
    longitude: float,
    radius_km: float,
    checkin: date | str,
    checkout: date | str,
    occupancy: int = DEFAULT_OCCUPANCY,
    rate_types: Optional[Sequence[str]] = None,
    n_points: int = SEARCH_POINT_DENSITY,
    rate_type_labels: Optional[Mapping[str, str]] = None,
    event_details: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """
    Build a JSON-serialisable payload containing the rate dataframe and search metadata.
    """
    df, metadata = generate_rates_dataframe(
        client,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        checkin=checkin,
        checkout=checkout,
        occupancy=occupancy,
        rate_types=rate_types,
        n_points=n_points,
        rate_type_labels=rate_type_labels,
    )

    payload = {
        "event": dict(event_details) if event_details is not None else None,
        "search": {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "radius_km": float(radius_km),
            "occupancy": int(occupancy),
            "checkin": metadata.get("checkin"),
            "checkout": metadata.get("checkout"),
            "rate_types": metadata.get("rate_types"),
            "rate_type_labels": metadata.get("rate_type_labels"),
        },
        "properties": df.to_dict(orient="records"),
    }
    return payload


def export_rates_to_excel(df: pd.DataFrame, destination: Path | str) -> Path:
    """Persist the rates dataframe to an Excel workbook for sharing."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return path


def export_rates_to_json(payload: Mapping[str, Any], destination: Path | str, *, pretty: bool = True) -> Path:
    """Write the generated payload to a JSON file."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    return path


def _is_rate_limit_error(exc: ExpediaAPIError) -> bool:
    """Check whether an Expedia error appears to be due to rate limiting."""
    message = str(exc)
    lowered = message.lower()
    return "429" in message or "rate limit" in lowered


def generate_rates_payload_with_retry(
    client: ExpediaClient,
    *,
    latitude: float,
    longitude: float,
    radius_km: float,
    checkin: date | str,
    checkout: date | str,
    occupancy: int = DEFAULT_OCCUPANCY,
    rate_types: Optional[Sequence[str]] = None,
    n_points: int = SEARCH_POINT_DENSITY,
    rate_type_labels: Optional[Mapping[str, str]] = None,
    event_details: Optional[Mapping[str, Any]] = None,
    max_attempts: int = 4,
    initial_backoff_seconds: float = 5.0,
    backoff_multiplier: float = 2.0,
    post_success_sleep_seconds: float = 0.0,
) -> Mapping[str, Any]:
    """
    Call generate_rates_payload with retry/backoff when the API indicates rate limiting.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    for attempt in range(1, max_attempts + 1):
        try:
            payload = generate_rates_payload(
                client,
                latitude=latitude,
                longitude=longitude,
                radius_km=radius_km,
                checkin=checkin,
                checkout=checkout,
                occupancy=occupancy,
                rate_types=rate_types,
                n_points=n_points,
                rate_type_labels=rate_type_labels,
                event_details=event_details,
            )
            if post_success_sleep_seconds > 0:
                time.sleep(post_success_sleep_seconds)
            return payload
        except ExpediaAPIError as exc:
            is_rate_limit = _is_rate_limit_error(exc)
            is_last_attempt = attempt == max_attempts
            if not is_rate_limit or is_last_attempt:
                raise
            delay = max(initial_backoff_seconds, 0.0) * (backoff_multiplier ** (attempt - 1))
            if delay > 0:
                time.sleep(delay)
    # The loop will either return or raise; this is a safety net.
    raise ExpediaAPIError("Failed to retrieve rates after retrying rate limit errors.")


def load_cached_events(path: Path | str) -> Sequence[Mapping[str, Any]]:
    """
    Load cached event rate records from disk. Returns an empty list if the file is missing or invalid.
    """
    cache_path = Path(path)
    if not cache_path.exists():
        return []
    try:
        data = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        return []
    if isinstance(data, list):
        return [record for record in data if isinstance(record, Mapping)]
    if isinstance(data, Mapping):
        return [data]
    return []


def _build_cached_dataframe(record: Mapping[str, Any]) -> Tuple[pd.DataFrame, MutableMapping[str, object]]:
    hotels = list(record.get("hotels") or [])
    df = pd.DataFrame(hotels)
    metadata: MutableMapping[str, object] = {}

    search_info = record.get("rates_search")
    if isinstance(search_info, Mapping):
        metadata.update(
            {
                "latitude": search_info.get("latitude"),
                "longitude": search_info.get("longitude"),
                "radius_km": search_info.get("radius_km"),
                "occupancy": search_info.get("occupancy"),
            }
        )
        metadata["rate_types"] = list(search_info.get("rate_types") or DEFAULT_RATE_TYPES)
        metadata["rate_type_labels"] = dict(search_info.get("rate_type_labels") or RATE_TYPE_LABELS)
    else:
        metadata["rate_types"] = list(DEFAULT_RATE_TYPES)
        metadata["rate_type_labels"] = dict(RATE_TYPE_LABELS)

    metadata["checkin"] = record.get("checkin")
    metadata["checkout"] = record.get("checkout")
    metadata["cached_radius_km"] = _safe_float((search_info or {}).get("radius_km")) if isinstance(search_info, Mapping) else None
    metadata["event"] = {
        "title": record.get("title"),
        "map_key": record.get("map_key"),
        "venue_id": record.get("venue_id"),
        "city": record.get("city"),
        "country": record.get("country"),
        "latitude": record.get("latitude"),
        "longitude": record.get("longitude"),
    }
    metadata["property_ids"] = [
        str(hotel["Property ID"])
        for hotel in hotels
        if isinstance(hotel, Mapping) and hotel.get("Property ID") is not None
    ]
    metadata["polygon"] = None
    metadata["source"] = "cache"
    return df, metadata


def find_cached_rates_dataframe(
    *,
    cache: Optional[Iterable[Mapping[str, Any]]] = None,
    cache_path: Optional[Path | str] = None,
    map_key: Optional[str] = None,
    venue_id: Optional[str | int | float] = None,
    title: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    checkin: date | str | None = None,
    checkout: date | str | None = None,
    radius_km: float | None = None,
    coordinate_tolerance_km: float = CACHE_DISTANCE_TOLERANCE_KM,
) -> Optional[Tuple[pd.DataFrame, MutableMapping[str, object]]]:
    """
    Search cached event rate data for a matching record and return a dataframe + metadata.
    """
    if cache is None:
        if cache_path is None:
            raise ValueError("Either cache or cache_path must be supplied.")
        cache = load_cached_events(cache_path)

    target_radius = float(radius_km) if radius_km is not None else None
    target_checkin = _try_parse_date(checkin)
    target_checkout = _try_parse_date(checkout)
    norm_map_key = _normalize_string(map_key)
    norm_title = _normalize_string(title)
    norm_venue_id = _normalize_string(venue_id)

    best_record: Optional[Mapping[str, Any]] = None
    best_record_radius: Optional[float] = None
    best_score = -1

    for record in cache:
        if not isinstance(record, Mapping):
            continue

        score = 0

        record_map_key = _normalize_string(record.get("map_key"))
        record_title = _normalize_string(record.get("title"))
        record_venue_id = _normalize_string(record.get("venue_id"))
        record_lat = record.get("latitude")
        record_lon = record.get("longitude")
        record_checkin = _try_parse_date(record.get("checkin"))
        record_checkout = _try_parse_date(record.get("checkout"))
        record_radius = None
        search_info = record.get("rates_search")
        if isinstance(search_info, Mapping):
            try:
                record_radius = float(search_info.get("radius_km")) if search_info.get("radius_km") is not None else None
            except (TypeError, ValueError):
                record_radius = None

        if norm_map_key is not None:
            if record_map_key is None or record_map_key != norm_map_key:
                continue
            score += 8

        if norm_venue_id is not None:
            if record_venue_id is None or record_venue_id != norm_venue_id:
                continue
            score += 4

        if norm_title is not None and record_title is not None:
            if record_title == norm_title:
                score += 2

        if target_checkin is not None:
            if record_checkin is None or record_checkin != target_checkin:
                continue
            score += 1

        if target_checkout is not None:
            if record_checkout is None or record_checkout != target_checkout:
                continue
            score += 1

        if target_radius is not None:
            if record_radius is None or record_radius < target_radius:
                continue
            score += 1

        if latitude is not None and longitude is not None:
            if record_lat is None or record_lon is None:
                continue
            try:
                dist_km = haversine_distance(float(latitude), float(longitude), float(record_lat), float(record_lon)) / 1000.0
            except (TypeError, ValueError):
                continue
            if coordinate_tolerance_km is not None and dist_km > float(coordinate_tolerance_km):
                continue
            score += 2

        if score > best_score and score >= 0:
            best_score = score
            best_record = record
            best_record_radius = record_radius

    if best_record is None:
        return None

    df, metadata = _build_cached_dataframe(best_record)
    metadata["cached_radius_km"] = best_record_radius if best_record_radius is not None else metadata.get("cached_radius_km")
    metadata["requested_radius_km"] = target_radius

    if target_radius is not None and best_record_radius is not None and best_record_radius >= target_radius:
        distance_col = "Distance to Center (km)"
        if distance_col in df.columns:
            distances = pd.to_numeric(df[distance_col], errors="coerce")
            mask = distances <= float(target_radius)
            mask = mask.fillna(False)
            df = df.loc[mask].reset_index(drop=True)
    metadata["property_ids"] = [
        str(row.get("Property ID"))
        for _, row in df.iterrows()
        if row.get("Property ID") is not None
    ]
    return df, metadata


def evaluate_deal_quality(
    df: pd.DataFrame,
    *,
    rate_types: Sequence[str],
    rate_type_labels: Mapping[str, str],
    discount_threshold_pct: float,
    savings_threshold: float,
    max_total_cost: float,
) -> Mapping[str, Any]:
    """
    Analyse the rate comparison dataframe and determine whether the event offers a good deal.

    The evaluation uses the property with the highest percentage discount between the primary
    (first) and secondary (second) rate types.
    """
    discount_col = "Discount (%)"
    discount_field = _resolve_dataframe_column(df, discount_col)
    if df.empty:
        return {
            "is_good_deal": False,
            "best_property": None,
            "best_discount_pct": None,
            "best_savings": None,
            "best_total_cost": None,
            "reason": "No rates available for this search.",
            "discount_threshold_pct": float(discount_threshold_pct),
            "savings_threshold": float(savings_threshold),
            "max_total_cost": float(max_total_cost),
            "qualifying_count": 0,
            "qualifying_properties": [],
        }

    if discount_field is None:
        return {
            "is_good_deal": False,
            "best_property": None,
            "best_discount_pct": None,
            "best_savings": None,
            "best_total_cost": None,
            "reason": "Discount data is unavailable.",
            "discount_threshold_pct": float(discount_threshold_pct),
            "savings_threshold": float(savings_threshold),
            "max_total_cost": float(max_total_cost),
            "qualifying_count": 0,
            "qualifying_properties": [],
        }

    if len(rate_types) < 2:
        return {
            "is_good_deal": False,
            "best_property": None,
            "best_discount_pct": None,
            "best_savings": None,
            "best_total_cost": None,
            "reason": "At least two rate types are required to evaluate savings.",
            "discount_threshold_pct": float(discount_threshold_pct),
            "savings_threshold": float(savings_threshold),
            "max_total_cost": float(max_total_cost),
            "qualifying_count": 0,
            "qualifying_properties": [],
        }

    primary_rate = rate_types[0]
    secondary_rate = rate_types[1]

    def _clean_rate_label(label: Any) -> str:
        if isinstance(label, str):
            stripped = label.strip()
            lowered = stripped.lower()
            for prefix in ("price (", "room ("):
                if lowered.startswith(prefix):
                    inner = stripped[len(prefix) :]
                    if inner.endswith(")"):
                        inner = inner[:-1]
                    return inner.strip()
            return stripped
        return str(label)

    primary_label_raw = rate_type_labels.get(primary_rate, primary_rate)
    secondary_label_raw = rate_type_labels.get(secondary_rate, secondary_rate)
    primary_label = _clean_rate_label(primary_label_raw)
    secondary_label = _clean_rate_label(secondary_label_raw)
    primary_price_col = f"Price ({primary_label})"
    secondary_price_col = f"Price ({secondary_label})"
    delta_col = f"Delta ({primary_label} - {secondary_label})"
    primary_price_field = _resolve_dataframe_column(df, primary_price_col)
    if primary_price_field is None:
        for candidate in (primary_label_raw, primary_rate):
            primary_price_field = _resolve_dataframe_column(df, candidate)
            if primary_price_field is not None:
                break
    secondary_price_field = _resolve_dataframe_column(df, secondary_price_col)
    if secondary_price_field is None:
        for candidate in (secondary_label_raw, secondary_rate):
            secondary_price_field = _resolve_dataframe_column(df, candidate)
            if secondary_price_field is not None:
                break
    delta_field = _resolve_dataframe_column(df, delta_col)

    discounts = pd.to_numeric(df[discount_field], errors="coerce")
    if discounts.isna().all():
        return {
            "is_good_deal": False,
            "best_property": None,
            "best_discount_pct": None,
            "best_savings": None,
            "best_total_cost": None,
            "reason": "No valid discount percentages were returned.",
            "discount_threshold_pct": float(discount_threshold_pct),
            "savings_threshold": float(savings_threshold),
            "max_total_cost": float(max_total_cost),
            "qualifying_count": 0,
            "qualifying_properties": [],
        }

    def _coerce_numeric(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if delta_field is not None:
        savings_series = pd.to_numeric(df[delta_field], errors="coerce")
    elif primary_price_field is not None and secondary_price_field is not None:
        primary_prices = pd.to_numeric(df[primary_price_field], errors="coerce")
        secondary_prices = pd.to_numeric(df[secondary_price_field], errors="coerce")
        savings_series = primary_prices - secondary_prices
    else:
        savings_series = pd.Series([float("nan")] * len(df), index=df.index)

    if secondary_price_field is not None:
        total_cost_series = pd.to_numeric(df[secondary_price_field], errors="coerce")
    elif primary_price_field is not None:
        total_cost_series = pd.to_numeric(df[primary_price_field], errors="coerce")
    else:
        total_cost_series = pd.Series([float("nan")] * len(df), index=df.index)

    discount_threshold = float(discount_threshold_pct)
    savings_threshold_value = float(savings_threshold)
    max_total_cost_value = float(max_total_cost)

    property_field = _resolve_dataframe_column(df, "Property")

    qualifying_mask = (
        discounts >= discount_threshold
    ) & (savings_series >= savings_threshold_value) & (total_cost_series <= max_total_cost_value)
    qualifying_mask = qualifying_mask & ~(discounts.isna() | savings_series.isna() | total_cost_series.isna())
    qualifying_count = int(qualifying_mask.sum())
    if property_field is not None:
        qualifying_properties = df.loc[qualifying_mask, property_field].dropna().astype(str).tolist()
    else:
        qualifying_properties = []

    sorted_indices = discounts.sort_values(ascending=False, na_position="last").index

    best_index = None
    best_discount = None
    best_savings = None
    best_total_cost = None
    best_property = None
    is_good_deal = False

    fallback_index = None
    fallback_discount = None
    fallback_savings = None
    fallback_total_cost = None
    fallback_property = None

    for idx in sorted_indices:
        discount_value = _coerce_numeric(discounts.loc[idx])
        savings_value = _coerce_numeric(savings_series.loc[idx])
        total_cost_value = _coerce_numeric(total_cost_series.loc[idx])
        row = df.loc[idx]
        property_name = row.get(property_field) if property_field is not None else None
        if property_name is not None and pd.isna(property_name):
            property_name = None

        if fallback_index is None:
            fallback_index = idx
            fallback_discount = None if discount_value is None or pd.isna(discount_value) else float(discount_value)
            fallback_savings = None if savings_value is None or pd.isna(savings_value) else float(savings_value)
            fallback_total_cost = None if total_cost_value is None or pd.isna(total_cost_value) else float(total_cost_value)
            fallback_property = property_name

        qualifies = (
            discount_value is not None
            and not pd.isna(discount_value)
            and savings_value is not None
            and not pd.isna(savings_value)
            and total_cost_value is not None
            and not pd.isna(total_cost_value)
            and float(discount_value) >= discount_threshold
            and float(savings_value) >= savings_threshold_value
            and float(total_cost_value) <= max_total_cost_value
        )

        if qualifies:
            best_index = idx
            best_discount = float(discount_value)
            best_savings = float(savings_value)
            best_total_cost = float(total_cost_value)
            best_property = property_name
            is_good_deal = True
            break

    if best_index is None:
        best_index = fallback_index
        best_discount = fallback_discount
        best_savings = fallback_savings
        best_total_cost = fallback_total_cost
        best_property = fallback_property

    if best_property is not None:
        if pd.isna(best_property):
            best_property = None
        else:
            best_property = str(best_property)

    if not is_good_deal:
        reasons = []
        if best_discount is None or pd.isna(best_discount):
            reasons.append("No valid discount identified.")
        elif float(best_discount) < discount_threshold:
            reasons.append(f"Best discount ({float(best_discount):.1f}%) is below the threshold ({discount_threshold:.1f}%).")
        if best_savings is None or pd.isna(best_savings):
            reasons.append("Unable to determine savings for the best discount.")
        elif float(best_savings) < savings_threshold_value:
            reasons.append(
                f"Best savings ({float(best_savings):.2f}) are below the threshold (${savings_threshold_value:.2f})."
            )
        if best_total_cost is None or pd.isna(best_total_cost):
            reasons.append("Unable to determine total cost for the best discount.")
        elif float(best_total_cost) > max_total_cost_value:
            reasons.append(f"Total cost ({float(best_total_cost):.2f}) exceeds the limit (${max_total_cost_value:.2f}).")
        reason = " ".join(reasons) or "Deal does not meet the configured thresholds."
    else:
        reason = "Best available deal meets the configured thresholds."

    return {
        "is_good_deal": bool(is_good_deal),
        "best_property": best_property,
        "best_discount_pct": None if best_discount is None or pd.isna(best_discount) else float(best_discount),
        "best_savings": None if best_savings is None or pd.isna(best_savings) else float(best_savings),
        "best_total_cost": None if best_total_cost is None or pd.isna(best_total_cost) else float(best_total_cost),
        "reason": reason,
        "discount_threshold_pct": discount_threshold,
        "savings_threshold": savings_threshold_value,
        "max_total_cost": max_total_cost_value,
        "qualifying_count": qualifying_count,
        "qualifying_properties": qualifying_properties,
    }


__all__ = [
    "generate_rates_dataframe",
    "generate_rates_payload",
    "generate_rates_payload_with_retry",
    "export_rates_to_excel",
    "export_rates_to_json",
    "DEFAULT_OCCUPANCY",
    "DEFAULT_RATE_TYPES",
    "RATE_TYPE_LABELS",
    "SEARCH_POINT_DENSITY",
    "evaluate_deal_quality",
    "load_cached_events",
    "find_cached_rates_dataframe",
    "CACHE_DISTANCE_TOLERANCE_KM",
]
