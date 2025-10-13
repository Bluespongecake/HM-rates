
import json
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import pandas as pd


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


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def property_coordinates(summary: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract latitude/longitude from a property summary payload.
    """
    location = summary.get("location")
    coordinates: Dict[str, Any] = {}
    if isinstance(location, dict):
        coordinates = location.get("coordinates") or {}
    if not isinstance(coordinates, dict):
        coordinates = {}
    lat = _safe_float(coordinates.get("latitude"))
    lon = _safe_float(coordinates.get("longitude"))
    return (lat, lon)


'''generate a circle of points on the earth surface of radius 
   this can be viewed by pasting the results of to_geojson_string into geojson.io'''

def destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    EARTH_RADIUS_M = 6371008.8  # mean Earth radius (meters)
    """
    Spherical Earth 'destination point' calculation. 
    I.e if I travel 1000m from (lat, long), around the surface of a sphere in a given direction, where do I end up?
    Probably not super importatnt for the case of a few km, but prudent nonetheless
    Returns (lat_deg, lon_deg).
    TODO: run through the maths and figure out how it actually works
    """
    φ1 = math.radians(lat_deg)
    λ1 = math.radians(lon_deg)
    θ = math.radians(bearing_deg)
    δ = distance_m / EARTH_RADIUS_M  # angular distance

    sinφ1, cosφ1 = math.sin(φ1), math.cos(φ1)
    sinδ, cosδ = math.sin(δ), math.cos(δ)
    sinθ, cosθ = math.sin(θ), math.cos(θ)

    sinφ2 = sinφ1 * cosδ + cosφ1 * sinδ * cosθ
    φ2 = math.asin(max(-1.0, min(1.0, sinφ2)))  # clamp

    y = sinθ * sinδ * cosφ1
    x = cosδ - sinφ1 * sinφ2
    λ2 = λ1 + math.atan2(y, x)

    # Normalize lon to [-180, 180)
    lon2 = (math.degrees(λ2) + 540) % 360 - 180
    lat2 = math.degrees(φ2)
    return lat2, lon2

def circle_polygon_geojson(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    n_points: int = 64
) -> Dict[str, Any]:
    """
    Generate a GeoJSON Polygon

    Args:
        center_lat: Latitude in degrees.
        center_lon: Longitude in degrees.
        radius_m: Radius in meters.
        n_points: Number of vertices around the circle (3..200). The polygon
                  will be closed by repeating the first coordinate at the end.

    Returns:
        A dict GeoJSON Polygon with coordinates as [ [ [lon, lat], ... ] ].
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be > 0")
    n = max(3, min(200, int(n_points)))

    coords: List[List[float]] = []
    # Sample bearings 0..360 (exclusive), then close ring by repeating first point
    for i in range(n):
        bearing = (360.0 * i) / n
        lat, lon = destination_point(center_lat, center_lon, bearing, radius_m)
        coords.append([lon, lat])  # GeoJSON is [lon, lat]

    # Close the polygon
    coords.append(coords[0])

    return {
        "type": "Polygon",
        "coordinates": [coords],
    }

def to_geojson_string(geo: Dict[str, Any], pretty: bool = True) -> str:
    """
    Convert a Python GeoJSON-like dict into a JSON string ready to paste
    into geojson.io for debugging etc
    """
    if pretty:
        return json.dumps(geo, indent=2)
    return json.dumps(geo, separators=(",", ":"))

# Example usage:
# polygon = circle_polygon_geojson(37.208957, -93.292000, radius_m=5000, n_points=120)
# ids = polygon_property_ids(polygon, include="property_ids", supply_source="expedia")
# print(ids)

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
    rate_type: str = "mkt_prepay",
    stay_nights: int = 1,
    days_ahead: int = 30,
    n_points: int = 96,
    supply_source: str = "expedia",
    include_room_name: bool = True,
    sort_results: bool = True,
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
        rate_type: Key lookup into ExpediaClient.RATE_PRESETS; defaults to public package rate.
        stay_nights: Length of stay (used when checkout omitted). Must be >= 1.
        days_ahead: Number of days from today to use when checkin omitted.
        n_points: Polygon vertex count used when constructing the search area.
        supply_source: Supply source sent to the geography search endpoint.
        include_room_name: If true, rows include the cheapest room name when available.
        sort_results: If true, results are sorted by ascending price.

    Returns:
        Dict containing:
            - dataframe: pandas.DataFrame of priced properties.
            - property_ids: List of property IDs returned by the geography search.
            - availability: Raw availability payload keyed by property ID.
            - summaries: Property content payload keyed by property ID.
            - rate_map: Mapping of property IDs to cheapest rate info.
            - polygon: GeoJSON polygon used in the search.
            - checkin / checkout: ISO formatted stay dates.
    """
    if stay_nights < 1:
        raise ValueError("stay_nights must be >= 1")

    occupancy_key = str(occupancy)

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
    )

    availability: Dict[str, Any] = {}
    summaries: Dict[str, Any] = {}

    if property_ids:
        ids_list = list(property_ids)

        def fetch_availability(batch: List[str]) -> Mapping[Any, Any]:
            return client.fetch_availability(
                batch,
                checkin,
                checkout,
                occupancy_key,
                rate_type,
                rate_plan_count=1,
            )

        def fetch_summaries(batch: List[str]) -> Mapping[Any, Any]:
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

    rate_map = extract_cheapest_rates_by_id(
        availability,
        occupancy_key=occupancy_key,
        include_room_name=include_room_name,
    )

    rows: List[Dict[str, Any]] = []
    for pid, value in rate_map.items():
        if value is None:
            continue
        if include_room_name and isinstance(value, tuple):
            room_name, price = value
        else:
            room_name, price = ("", value if value is not None else None)

        if price is None:
            continue

        summary = summaries.get(pid, {})
        name = property_display_name(summary, fallback=pid)
        lat_lon = property_coordinates(summary)
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

    columns = [
        "Property",
        "Property ID",
        "Room",
        "Price (request currency)",
        "latitude",
        "longitude",
    ]
    if rows:
        df = pd.DataFrame(rows)
        if sort_results and "Price (request currency)" in df.columns:
            df = df.sort_values("Price (request currency)", ignore_index=True)
    else:
        df = pd.DataFrame(columns=columns)

    return {
        "dataframe": df,
        "property_ids": list(property_ids) if property_ids else [],
        "availability": availability,
        "summaries": summaries,
        "rate_map": rate_map,
        "polygon": polygon,
        "checkin": checkin,
        "checkout": checkout,
    }
