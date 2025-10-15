"""Helper utilities for loading and normalising event data."""

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


def normalise_event_record(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Standardise the structure of an event record regardless of source schema."""
    event = raw.get("event") if isinstance(raw.get("event"), Mapping) else raw
    title = event.get("title") or raw.get("title")
    city = event.get("city") or raw.get("city")
    country = event.get("country") or raw.get("country")
    latitude = event.get("latitude") or event.get("lat")
    longitude = event.get("longitude") or event.get("lon")
    venue_id = event.get("venue_id") or raw.get("venue_id")
    map_key = event.get("map_key") or raw.get("map_key")
    date_start = event.get("date_start") or event.get("start_date") or raw.get("date_start")
    date_end = event.get("date_end") or event.get("end_date") or raw.get("date_end")

    return {
        "title": title,
        "city": city,
        "country": country,
        "latitude": latitude,
        "longitude": longitude,
        "venue_id": venue_id,
        "map_key": map_key,
        "date_start": date_start,
        "date_end": date_end,
    }


def load_events_from_json(path: Path) -> List[Dict[str, Any]]:
    """Load and normalise event records from a JSON file."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    items: List[Mapping[str, Any]] = []
    if isinstance(data, list):
        items = [item for item in data if isinstance(item, Mapping)]
    elif isinstance(data, Mapping):
        items = [data]
    else:
        return []

    out: List[Dict[str, Any]] = []
    for item in items:
        normalised = normalise_event_record(item)
        normalised["source"] = str(path)
        out.append(normalised)
    return out


def load_events_from_excel(path: Path) -> List[Dict[str, Any]]:
    """Load and normalise event records from an Excel file."""
    try:
        df = pd.read_excel(path)
    except Exception:
        return []

    records: List[Dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        if not isinstance(record, Mapping):
            continue
        normalised = normalise_event_record(record)
        normalised["source"] = str(path)
        # Include latitude/longitude if present under alternative keys.
        for alt_key, target in (("lat", "latitude"), ("lon", "longitude"), ("lng", "longitude")):
            if normalised.get(target) is None and record.get(alt_key) is not None:
                normalised[target] = record.get(alt_key)
        records.append(normalised)
    return records


def load_events_from_file(path: Path) -> List[Dict[str, Any]]:
    """Load events based on file extension, returning an empty list if unsupported."""
    suffix = path.suffix.lower()
    if suffix in {".json", ".geojson"}:
        return load_events_from_json(path)
    if suffix in {".xlsx", ".xls"}:
        return load_events_from_excel(path)
    return []


def load_events_catalog(
    directories: Iterable[Path],
    patterns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build an events catalog dataframe from one or more directories."""
    patterns = patterns or ("*events*.json", "*event_*.json", "*events*.xlsx", "*events*.xls")

    records: List[Dict[str, Any]] = []
    sources: List[Path] = []

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        for pattern in patterns:
            for event_path in sorted(dir_path.glob(pattern)):
                records.extend(load_events_from_file(event_path))
                if event_path not in sources:
                    sources.append(event_path)

    if not records:
        return pd.DataFrame()

    if sources:
        print("Loaded event sources:")
        for source in sources:
            print(f"  - {source}")

    df = pd.DataFrame(records)

    for col in ("latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("date_start", "date_end"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["date_end"] = df.apply(
        lambda row: row["date_start"] if pd.isna(row["date_end"]) else row["date_end"],
        axis=1,
    )

    df = df.dropna(subset=["title", "latitude", "longitude", "date_start"])

    today_ts = pd.to_datetime(date.today())
    df = df[df["date_end"] >= today_ts]

    df = df.drop_duplicates(subset=["map_key", "venue_id", "title", "date_start"], keep="first")

    def _format_label(row: pd.Series) -> str:
        start_str = row["date_start"].strftime("%d %b %Y") if pd.notna(row["date_start"]) else "Unknown"
        end_str = row["date_end"].strftime("%d %b %Y") if pd.notna(row["date_end"]) else start_str
        location = row["city"] or row["country"] or "Unknown location"
        return f"{row['title']} • {location} • {start_str} → {end_str}"

    df["display_label"] = df.apply(_format_label, axis=1)
    df["search_blob"] = (
        df[["title", "city", "country", "map_key", "venue_id"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    return df.reset_index(drop=True)


__all__ = [
    "normalise_event_record",
    "load_events_from_json",
    "load_events_from_excel",
    "load_events_from_file",
    "load_events_catalog",
]
