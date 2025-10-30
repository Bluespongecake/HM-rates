"""Geospatial helper functions shared across the Expedia toolkit."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    """
    Compute the lat/lon reached when travelling a distance along a bearing on a spherical Earth.
    """
    earth_radius_m = 6378137.0  # WGS84 equatorial radius (meters)
    φ1 = math.radians(lat_deg)
    λ1 = math.radians(lon_deg)
    θ = math.radians(bearing_deg)
    δ = distance_m / earth_radius_m  # angular distance

    sinφ1, cosφ1 = math.sin(φ1), math.cos(φ1)
    sinδ, cosδ = math.sin(δ), math.cos(δ)
    sinθ, cosθ = math.sin(θ), math.cos(θ)

    sinφ2 = sinφ1 * cosδ + cosφ1 * sinδ * cosθ
    φ2 = math.asin(max(-1.0, min(1.0, sinφ2)))  # clamp for numerical safety

    y = sinθ * sinδ * cosφ1
    x = cosδ - sinφ1 * sinφ2
    λ2 = λ1 + math.atan2(y, x)

    lon2 = (math.degrees(λ2) + 540) % 360 - 180  # normalize to [-180, 180)
    lat2 = math.degrees(φ2)
    return lat2, lon2


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    *,
    radius_m: float = 6378137.0,
) -> float:
    """
    Compute great-circle distance between two points on Earth using the haversine formula.
    """
    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)

    sin_dφ = math.sin(dφ / 2.0)
    sin_dλ = math.sin(dλ / 2.0)

    a = sin_dφ ** 2 + math.cos(φ1) * math.cos(φ2) * sin_dλ ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return radius_m * c


def circle_polygon_geojson(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    n_points: int = 64,
) -> Dict[str, Any]:
    """
    Generate a GeoJSON Polygon approximating a circle around a coordinate.
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be > 0")
    n = max(3, min(200, int(n_points)))

    coords: List[List[float]] = []
    for i in range(n):
        bearing = (360.0 * i) / n
        lat, lon = destination_point(center_lat, center_lon, bearing, radius_m)
        coords.append([lon, lat])  # GeoJSON expects [lon, lat]

    coords.append(coords[0])  # close the polygon

    return {
        "type": "Polygon",
        "coordinates": [coords],
    }


def to_geojson_string(geo: Dict[str, Any], pretty: bool = True) -> str:
    """Serialise a GeoJSON dictionary to a JSON string."""
    if pretty:
        return json.dumps(geo, indent=2)
    return json.dumps(geo, separators=(",", ":"))


__all__ = [
    "property_coordinates",
    "destination_point",
    "haversine_distance",
    "circle_polygon_geojson",
    "to_geojson_string",
]
