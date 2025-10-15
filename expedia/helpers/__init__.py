from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import pandas as pd

from ..geo_helpers import (
    circle_polygon_geojson,
    destination_point,
    haversine_distance,
    property_coordinates,
    to_geojson_string,
)


def build_rates_dataframe(jsons, labels, occupancy_key: str = "2"):
    """
    Parameters
    ----------
    jsons : list of dict
        Each dict should have the same structure you posted (contains rooms with 'room_name' and 'rates').
    labels : list of str
        Names to assign to each JSON's prices column. Must match length of jsons.
    occupancy_key : str
        Occupancy key to read from the availability payload (defaults to "2").
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'room_name' from the first JSON and one column per label
        showing the cheapest 'inclusive' total price (request_currency.value) for that room.
    """
    if len(jsons) != len(labels):
        raise ValueError("`jsons` and `labels` must have the same length.")
    
    def _extract_price_from_rate(rate):
        """Safely pull the inclusive request-currency value (as float) for the chosen occupancy from a rate."""
        try:
            return float(
                rate["occupancy_pricing"][occupancy_key]["totals"]["inclusive"]["request_currency"]["value"]
            )
        except (KeyError, TypeError, ValueError):
            return None

    # --- Helper to extract {room_name: cheapest_price} mapping ---
    def extract_room_prices(data):
        mapping = {}
        for room in data.get("rooms", []):
            room_name = room.get("room_name")
            if not room_name:
                continue

            prices = []
            for rate in room.get("rates", []) or []:
                price = _extract_price_from_rate(rate)
                if price is not None:
                    prices.append(price)

            mapping[room_name] = min(prices) if prices else None
        return mapping
    
    # --- Build initial DataFrame from first JSON ---
    first_json = jsons[0]
    rooms = sorted(first_json.get("rooms", []), key=lambda r: (r.get("room_name") or "").lower())
    df = pd.DataFrame([{"room_name": r.get("room_name")} for r in rooms])

    # Add columns for each JSON/label
    for data, label in zip(jsons, labels):
        price_map = extract_room_prices(data)
        df[label] = df["room_name"].map(price_map)
    
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
                includes=("name", "location"),
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
        row.update(
            {
                "Property": name,
                "Property ID": pid,
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

    base_columns = ["Property", "Property ID", "Room", "Price (request currency)", "latitude", "longitude"]

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
