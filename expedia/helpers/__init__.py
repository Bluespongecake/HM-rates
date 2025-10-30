from datetime import date, datetime, timedelta
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from ..geo_helpers import (
    circle_polygon_geojson,
    destination_point,
    haversine_distance,
    property_coordinates,
    to_geojson_string,
)


def build_rates_dataframe(
    jsons,
    labels,
    occupancy_key: str = "2",
    *,
    total_fields: Optional[Iterable[str]] = None,
):
    """
    Parameters
    ----------
    jsons : list of dict
        Each dict should have the same structure you posted (contains rooms with 'room_name' and 'rates').
    labels : list of str
        Names to assign to each JSON's prices column. Must match length of jsons.
    occupancy_key : str
        Occupancy key to read from the availability payload (defaults to "2").
    total_fields : iterable of str, optional
        Which totals fields to extract from each rate payload. Defaults to ("inclusive",).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'room_name' from the first JSON and one column per label/total field
        showing values derived from the cheapest rate for that room.
    """
    if len(jsons) != len(labels):
        raise ValueError("`jsons` and `labels` must have the same length.")

    totals_to_extract: Sequence[str]
    if total_fields is None:
        totals_to_extract = ("inclusive",)
    else:
        totals_to_extract = tuple(total_fields)
        if not totals_to_extract:
            raise ValueError("`total_fields` must include at least one key.")

    ranking_key = "inclusive" if "inclusive" in totals_to_extract else totals_to_extract[0]

    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.replace(",", ""))
            except ValueError:
                return None
        return None

    def _extract_totals_from_rate(rate: Mapping[str, Any]) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        occupancy_data = (
            ((rate or {}).get("occupancy_pricing") or {})
            .get(occupancy_key, {})
        )
        totals_node = occupancy_data.get("totals")
        if not isinstance(totals_node, Mapping):
            return totals

        for field in totals_to_extract:
            field_node = totals_node.get(field)
            value: Optional[float] = None
            if isinstance(field_node, Mapping):
                request_currency = field_node.get("request_currency")
                if isinstance(request_currency, Mapping):
                    value = _coerce_float(request_currency.get("value"))
                if value is None:
                    value = _coerce_float(field_node.get("value"))
            else:
                value = _coerce_float(field_node)

            if value is not None:
                totals[field] = value
        return totals

    def _select_totals_for_room(room: Mapping[str, Any]) -> Dict[str, float]:
        best_totals: Optional[Dict[str, float]] = None
        best_rank_value: Optional[float] = None

        rates = room.get("rates", []) or []
        for rate in rates:
            totals = _extract_totals_from_rate(rate)
            if not totals:
                continue

            rank_value = totals.get(ranking_key)
            if rank_value is None:
                # Keep candidate so we have fallback data if all rates miss the ranking key.
                if best_totals is None:
                    best_totals = totals
                continue

            if best_rank_value is None or rank_value < best_rank_value:
                best_rank_value = rank_value
                best_totals = totals

        return best_totals or {}

    # --- Helper to extract {room_name: cheapest_price} mapping ---
    def extract_room_prices(data):
        mapping: Dict[str, Dict[str, float]] = {}
        for room in data.get("rooms", []):
            room_name = room.get("room_name")
            if not room_name:
                continue

            totals = _select_totals_for_room(room)
            mapping[room_name] = totals
        return mapping
    
    # --- Build initial DataFrame from first JSON ---
    first_json = jsons[0]
    rooms = sorted(first_json.get("rooms", []), key=lambda r: (r.get("room_name") or "").lower())
    df = pd.DataFrame([{"room_name": r.get("room_name")} for r in rooms])

    # Add columns for each JSON/label
    for data, label in zip(jsons, labels):
        price_map = extract_room_prices(data)
        for field in totals_to_extract:
            column_name = label
            if not (len(totals_to_extract) == 1 and field == "inclusive"):
                column_name = f"{label}_{field}"
            df[column_name] = df["room_name"].map(lambda name: price_map.get(name, {}).get(field))
    
    return df


def cheapest_total_inclusive(
    property_data: Dict[str, Any],
    occupancy_key: str = "2",
    *,
    include_room_name: bool = False,
) -> Optional[Union[Tuple[Optional[str], float], float]]:
    best_price: Optional[float] = None
    best_room: Optional[str] = None
    for room in property_data.get("rooms", []) or []:
        name = room.get("room_name")
        for rate in room.get("rates", []) or []:
            try:
                price = float(
                    rate["occupancy_pricing"][occupancy_key]["totals"]["inclusive"]["request_currency"]["value"]
                )
            except (KeyError, TypeError, ValueError):
                continue
            if best_price is None or price < best_price:
                best_price = price
                best_room = name

    if include_room_name:
        return (best_room, best_price) if best_price is not None else None
    return best_price


def extract_cheapest_rates_by_id(
    rates_by_id: Dict[str, Any],
    occupancy_key: str = "2",
    *,
    include_room_name: bool = False,
) -> Dict[str, Union[float, Tuple[Optional[str], float]]]:
    out: Dict[str, Union[float, Tuple[Optional[str], float]]] = {}
    for pid, pdata in (rates_by_id or {}).items():
        result = cheapest_total_inclusive(
            pdata,
            occupancy_key,
            include_room_name=include_room_name,
        )
        if result is None:
            continue
        key = str(pid)
        out[key] = result
    return out


def property_display_name(summary: Dict[str, Any], fallback: str = "") -> str:
    """
    Best-effort extraction of a property's name from the summary payload returned by EPS Rapid.
    """
    name = summary.get("name")
    if isinstance(name, str):
        return name
    if isinstance(name, dict):
        # Common shapes: {"content": "Hotel"}, {"value": "Hotel"}, {"content": {"value": "Hotel"}}
        for key in ("content", "value"):
            value = name.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                nested_value = value.get("value")
                if isinstance(nested_value, str):
                    return nested_value
    property_name = summary.get("property_name")
    if isinstance(property_name, str):
        return property_name
    return fallback


def property_category_label(summary: Mapping[str, Any]) -> Optional[str]:
    """
    Extract a human-readable property category name from a summary payload.
    """
    category = summary.get("category")
    if isinstance(category, Mapping):
        name = category.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        ident = category.get("id")
        if isinstance(ident, (str, int, float)):
            ident_str = str(ident).strip()
            if ident_str:
                return ident_str
    elif isinstance(category, str) and category.strip():
        return category.strip()
    return None


def _parse_int_like(value: Any) -> Optional[int]:
    """
    Try to coerce a value that looks numeric into an integer.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return int(round(float(value)))
    if isinstance(value, str):
        match = re.search(r"\d+", value.replace(",", ""))
        if match:
            try:
                return int(match.group())
            except ValueError:
                return None
    return None


def property_total_room_count(summary: Mapping[str, Any]) -> Optional[int]:
    """
    Extract the total number of rooms for a property from the statistics payload.
    """
    stats = summary.get("statistics")
    if isinstance(stats, Mapping):
        items = stats.values()
    elif isinstance(stats, list):
        items = stats
    else:
        items = []

    for entry in items:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        value = entry.get("value")
        identifier = entry.get("id")
        parsed_value = _parse_int_like(value)
        if parsed_value is None:
            parsed_value = _parse_int_like(name)
        if parsed_value is None:
            continue
        name_text = name.lower() if isinstance(name, str) else ""
        if identifier == "52" or "total number of rooms" in name_text:
            return parsed_value
    return None



def generate_dates(start: str, end: str, step: Union[int, timedelta]) -> List[str]:
    """
    Generate a list of dates between start and end (inclusive) with a given step.
    
    Parameters:
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
        step (int or timedelta): Step size between dates. 
                                 If int, interpreted as number of days.
                                 If timedelta, used directly.
    
    Returns:
        List[str]: List of dates in 'YYYY-MM-DD' format.
    """
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    
    if isinstance(step, int):
        step = timedelta(days=step)
    elif not isinstance(step, timedelta):
        raise ValueError("Step must be either an integer (days) or a timedelta object.")
    
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += step
    
    return dates


def add_days(date_str: str, n: int) -> str:
    """
    Add n days to a given date.

    Parameters:
        date_str (str): The date in 'YYYY-MM-DD' format.
        n (int): Number of days to add (can be negative to subtract days).

    Returns:
        str: New date in 'YYYY-MM-DD' format.
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    new_date = date_obj + timedelta(days=n)
    return new_date.strftime("%Y-%m-%d")


def fetch_rates_near_coordinate(
    client: Any,
    *,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    checkin: Optional[str] = None,
    checkout: Optional[str] = None,
    occupancy: Union[int, str] = 2,
    rate_type: Union[str, Iterable[str]] = "mkt_prepay",
    stay_nights: int = 1,
    days_ahead: int = 30,
    n_points: int = 96,
    supply_source: str = "expedia",
    include_room_name: bool = True,
    sort_results: bool = True,
    rate_type_labels: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """
    Retrieve the cheapest room offers for properties near a coordinate.

    Args:
        client: An ExpediaClient (or API-compatible object) used to perform calls.
        center_lat: Search latitude in degrees.
        center_lon: Search longitude in degrees.
        radius_km: Search radius in kilometres.
        checkin: Optional check-in date (YYYY-MM-DD). If omitted, defaults to today + days_ahead.
        checkout: Optional check-out date (YYYY-MM-DD). If omitted, defaults to checkin + stay_nights.
        occupancy: Occupancy requested when fetching availability.
        rate_type: Single rate type key, or iterable of keys, from ExpediaClient.RATE_PRESETS.
        stay_nights: Length of stay (used when checkout omitted). Must be >= 1.
        days_ahead: Number of days from today to use when checkin omitted.
        n_points: Polygon vertex count used when constructing the search area.
        supply_source: Supply source sent to the geography search endpoint.
        include_room_name: If true, rows include the cheapest room name when available.
        sort_results: If true, results are sorted by ascending price.
        rate_type_labels: Optional mapping of rate type -> friendly display label.

    Returns:
        Dict containing:
            - dataframe: pandas.DataFrame of priced properties.
            - property_ids: List of property IDs returned by the geography search.
            - availability: Raw availability payload keyed by property ID for the first rate type.
            - availability_by_rate_type: Mapping of rate type -> availability payloads.
            - summaries: Property content payload keyed by property ID.
            - rate_map: Mapping of property IDs to cheapest rate info for the first rate type.
            - rate_maps: Mapping of rate type -> cheapest rate info.
            - polygon: GeoJSON polygon used in the search.
            - checkin / checkout: ISO formatted stay dates.
            - rate_type_labels: Mapping of rate type to display label.
    """
    if stay_nights < 1:
        raise ValueError("stay_nights must be >= 1")

    occupancy_key = str(occupancy)

    if isinstance(rate_type, str):
        rate_types = [rate_type]
    else:
        rate_types = list(rate_type)

    if not rate_types:
        raise ValueError("At least one rate_type must be provided.")

    primary_rate_type = rate_types[0]
    label_mapping: Dict[str, str] = {}
    if rate_type_labels:
        label_mapping.update({key: val for key, val in rate_type_labels.items() if isinstance(val, str)})
    for rt in rate_types:
        label_mapping.setdefault(rt, rt)

    if checkin is None:
        checkin_date = date.today() + timedelta(days=days_ahead)
        checkin = checkin_date.isoformat()
    else:
        checkin_date = datetime.strptime(checkin, "%Y-%m-%d").date()

    if checkout is None:
        checkout_date = checkin_date + timedelta(days=stay_nights)
        checkout = checkout_date.isoformat()
    else:
        checkout_date = datetime.strptime(checkout, "%Y-%m-%d").date()

    polygon = circle_polygon_geojson(
        center_lat=center_lat,
        center_lon=center_lon,
        radius_m=radius_km * 1000,
        n_points=n_points,
    )

    property_ids = client.search_geography(
        polygon,
        include="property_ids",
        supply_source=supply_source,
        checkin=checkin,
        checkout=checkout,
    )

    availability_by_rate_type: Dict[str, Dict[str, Any]] = {rt: {} for rt in rate_types}
    summaries: Dict[str, Any] = {}

    if property_ids:
        ids_list = list(property_ids)

        def fetch_summaries(batch: List[str]) -> Mapping[Any, Any]:
            return client.fetch_property_summaries(
                batch,
                includes=("name", "location", "statistics", "category"),
            )

        for rt in rate_types:
            sink = availability_by_rate_type[rt]

            def fetch_factory(rate_type_key: str):
                def _fetch(batch: List[str]) -> Mapping[Any, Any]:
                    return client.fetch_availability(
                        batch,
                        checkin,
                        checkout,
                        occupancy_key,
                        rate_type_key,
                        rate_plan_count=1,
                    )

                return _fetch

            client.run_batched(
                ids_list,
                jobs=[(fetch_factory(rt), sink)],
                batch_size=250,
            )

        client.run_batched(
            ids_list,
            jobs=[(fetch_summaries, summaries)],
            batch_size=250,
        )

    rate_maps: Dict[str, Dict[str, Union[float, Tuple[Optional[str], float]]]] = {}
    for rt, availability in availability_by_rate_type.items():
        rate_maps[rt] = extract_cheapest_rates_by_id(
            availability,
            occupancy_key=occupancy_key,
            include_room_name=include_room_name,
        )

    rows: List[Dict[str, Any]] = []
    all_property_ids: List[str] = sorted(
        {str(pid) for rates in rate_maps.values() for pid in rates.keys()}
    )

    for pid in all_property_ids:
        row: Dict[str, Any] = {}
        summary = summaries.get(pid, {})
        name = property_display_name(summary, fallback=pid)
        lat_lon = property_coordinates(summary)
        total_rooms = property_total_room_count(summary)
        row.update(
            {
                "Property": name,
                "Property ID": pid,
                "Total rooms": total_rooms,
                "type": property_category_label(summary),
                "latitude": lat_lon[0],
                "longitude": lat_lon[1],
            }
        )

        primary_price: Optional[float] = None
        primary_room: Optional[str] = None

        for rt in rate_types:
            label = label_mapping.get(rt, rt)
            value = rate_maps.get(rt, {}).get(pid)
            room_name: Optional[str] = None
            price_value: Optional[float] = None
            if include_room_name and isinstance(value, tuple):
                room_name, price_value = value
            elif isinstance(value, tuple):
                _, price_value = value
            elif value is not None:
                price_value = float(value) if not isinstance(value, tuple) else None
            if price_value is not None:
                try:
                    price_value = float(price_value)
                except (TypeError, ValueError):
                    price_value = None

            row[f"Price ({label})"] = price_value
            if include_room_name:
                row[f"Room ({label})"] = room_name or ""

            if rt == primary_rate_type:
                primary_price = price_value
                primary_room = room_name

        row["Price (request currency)"] = primary_price
        if include_room_name:
            row["Room"] = primary_room or ""

        rows.append(row)

    base_columns = [
        "Property",
        "Property ID",
        "Total rooms",
        "type",
        "Room",
        "Price (request currency)",
        "latitude",
        "longitude",
    ]

    if rows:
        df = pd.DataFrame(rows)
        if include_room_name:
            room_cols = [col for col in df.columns if col.startswith("Room (")]
        else:
            room_cols = []
        price_cols = [col for col in df.columns if col.startswith("Price (")]
        extra_cols = [col for col in df.columns if col not in set(base_columns + room_cols + price_cols)]
        ordered_cols = (
            ["Property", "Property ID"]
            + room_cols
            + price_cols
            + [col for col in base_columns if col not in {"Property", "Property ID", "Room", "Price (request currency)"}]
            + extra_cols
        )
        df = df[ordered_cols]
        if sort_results and "Price (request currency)" in df.columns:
            df = df.sort_values("Price (request currency)", ignore_index=True)
    else:
        df = pd.DataFrame(columns=base_columns)

    return {
        "dataframe": df,
        "property_ids": list(property_ids) if property_ids else [],
        "availability": availability_by_rate_type[primary_rate_type],
        "availability_by_rate_type": availability_by_rate_type,
        "summaries": summaries,
        "rate_map": rate_maps[primary_rate_type],
        "rate_maps": rate_maps,
        "polygon": polygon,
        "checkin": checkin,
        "checkout": checkout,
        "rate_type_labels": label_mapping,
        "rate_types": rate_types,
    }
