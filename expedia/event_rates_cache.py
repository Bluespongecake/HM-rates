"""Helpers for building and refreshing the cached event → hotel rate database."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd

from .client import ExpediaClient
from .helpers.rates_report import (
    DEFAULT_OCCUPANCY,
    DEFAULT_RATE_TYPES,
    RATE_TYPE_LABELS,
    generate_rates_payload_with_retry,
)

DEFAULT_CACHE_PATH = Path("reports/events_with_hotels.json")


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _normalize_string(value: Any) -> Optional[str]:
    if value is None or _is_nan(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped.lower() if stripped else None
    return str(value).strip().lower()


def _event_identity(record: Mapping[str, Any]) -> Optional[Tuple[str, str, Any, Any]]:
    checkin = record.get("checkin")
    checkout = record.get("checkout")

    map_key = _normalize_string(record.get("map_key"))
    if map_key:
        return ("map_key", map_key, checkin, checkout)

    venue_id = record.get("venue_id")
    if venue_id is not None and not _is_nan(venue_id):
        venue_key = _normalize_string(venue_id)
        if venue_key:
            return ("venue_id", venue_key, checkin, checkout)

    title = _normalize_string(record.get("title"))
    city = _normalize_string(record.get("city"))
    if title:
        return ("title", title, city, checkin, checkout)

    return None


def _hotel_key(hotel: Mapping[str, Any]) -> Optional[str]:
    property_id = hotel.get("Property ID")
    if property_id is not None and not _is_nan(property_id):
        return str(property_id)
    return _normalize_string(hotel.get("Property"))


def _merge_hotels(existing: Sequence[Mapping[str, Any]], new: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def _insert(hotel: Mapping[str, Any]) -> None:
        key = _hotel_key(hotel)
        if key is None:
            key = f"__anonymous_{len(order)}"
        if key not in order:
            order.append(key)
        merged[key] = dict(hotel)

    for hotel in existing or []:
        _insert(hotel)
    for hotel in new or []:
        _insert(hotel)

    return [merged[key] for key in order]


def _safe_float(value: Any) -> Optional[float]:
    if value is None or _is_nan(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_widest_search(existing: Mapping[str, Any] | None, new: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(existing, Mapping):
        return dict(new or {})
    if not isinstance(new, Mapping):
        return dict(existing)

    existing_radius = _safe_float(existing.get("radius_km"))
    new_radius = _safe_float(new.get("radius_km"))

    if new_radius is not None and (existing_radius is None or new_radius > existing_radius):
        merged = dict(existing)
        merged.update(new)
        merged["radius_km"] = new_radius
        return merged

    merged = dict(new)
    merged.update(existing)
    if existing_radius is not None:
        merged["radius_km"] = existing_radius
    return merged


def _load_cache(path: Path) -> List[Dict[str, Any]]:
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


def _save_cache(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [dict(record) for record in records]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def update_event_rates_cache(
    events_catalog: pd.DataFrame,
    client: ExpediaClient,
    *,
    output_path: Path | str = DEFAULT_CACHE_PATH,
    limit: Optional[int] = None,
    default_radius_km: float = 3.0,
    retry_kwargs: Optional[Mapping[str, Any]] = None,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch rates for upcoming events and merge them into the cached database.

    - New events are appended.
    - Existing events merge hotel data by property ID (overwriting existing entries and adding new ones).
    - The stored ``rates_search`` metadata keeps the largest search radius seen for that event.
    """
    output_path = Path(output_path)
    cache_records = _load_cache(output_path)

    records_by_key: Dict[Tuple[str, str, Any, Any], Dict[str, Any]] = {}
    for record in cache_records:
        identity = _event_identity(record)
        if identity is not None:
            records_by_key[identity] = record

    retry_config = dict(
        max_attempts=5,
        initial_backoff_seconds=8.0,
        backoff_multiplier=1.5,
        post_success_sleep_seconds=2.0,
    )
    if retry_kwargs:
        retry_config.update(retry_kwargs)

    processed = 0
    iterable = events_catalog.iterrows()
    for _, event in iterable:
        if limit is not None and processed >= limit:
            break

        lat = event.get("latitude")
        lon = event.get("longitude")
        start = event.get("date_start")
        if pd.isna(lat) or pd.isna(lon) or pd.isna(start):
            continue

        checkin_date = start.date()
        end_ts = event.get("date_end")
        if pd.notna(end_ts):
            checkout_date = end_ts.date()
        else:
            checkout_date = checkin_date + pd.Timedelta(days=1)

        radius_value = event.get("radius_km")
        if pd.isna(radius_value):
            radius_value = default_radius_km
        search_radius = float(radius_value)

        venue_id = event.get("venue_id")
        if pd.isna(venue_id):
            venue_id = None
        elif isinstance(venue_id, float) and venue_id.is_integer():
            venue_id = int(venue_id)

        map_key = event.get("map_key")
        if isinstance(map_key, str):
            map_key = map_key.strip() or None
        elif pd.isna(map_key):
            map_key = None

        event_info: Dict[str, Any] = {
            "title": event.get("title"),
            "map_key": map_key,
            "venue_id": venue_id,
            "city": event.get("city"),
            "country": event.get("country"),
            "latitude": float(lat),
            "longitude": float(lon),
            "checkin": checkin_date.isoformat(),
            "checkout": checkout_date.isoformat(),
        }

        event_details = {
            "title": event_info["title"],
            "map_key": map_key,
            "venue_id": venue_id,
        }

        payload = generate_rates_payload_with_retry(
            client,
            latitude=event_info["latitude"],
            longitude=event_info["longitude"],
            radius_km=search_radius,
            checkin=event_info["checkin"],
            checkout=event_info["checkout"],
            occupancy=DEFAULT_OCCUPANCY,
            rate_types=DEFAULT_RATE_TYPES,
            rate_type_labels=RATE_TYPE_LABELS,
            event_details=event_details,
            **retry_config,
        )

        new_hotels = [dict(hotel) for hotel in payload.get("properties") or []]
        new_search = dict(payload.get("search") or {})
        new_search.setdefault("radius_km", search_radius)

        record_key = _event_identity(event_info)

        if record_key and record_key in records_by_key:
            existing_record = records_by_key[record_key]

            for field, value in event_info.items():
                if value is not None:
                    existing_record[field] = value

            existing_hotels = existing_record.get("hotels") or []
            existing_record["hotels"] = _merge_hotels(existing_hotels, new_hotels)
            existing_record["rates_search"] = _select_widest_search(existing_record.get("rates_search"), new_search)
        else:
            event_info["hotels"] = new_hotels
            event_info["rates_search"] = new_search
            cache_records.append(event_info)
            if record_key:
                records_by_key[record_key] = event_info

        processed += 1
        if show_progress:
            prefix = f"[{processed}]"
            title = event_info.get("title") or "Unnamed event"
            print(f"{prefix} Cached rates for {title} (radius {search_radius} km)")

    _save_cache(output_path, cache_records)
    return cache_records


__all__ = ["update_event_rates_cache", "DEFAULT_CACHE_PATH"]
