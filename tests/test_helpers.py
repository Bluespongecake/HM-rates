import pytest

import expedia.helpers


@pytest.fixture
def availability_payload():
    # Two rooms, each with multiple rates. Second room has missing pricing info for one rate.
    return {
        "rooms": [
            {
                "room_name": "Standard Room",
                "rates": [
                    _rate("120.00"),
                    _rate("95.50"),
                ],
            },
            {
                "room_name": "Suite",
                "rates": [
                    _rate("250.00"),
                    {"occupancy_pricing": {}},  # malformed entry should be ignored
                    _rate("210.75"),
                ],
            },
        ]
    }


def _rate(value: str):
    return {
        "occupancy_pricing": {
            "2": {
                "totals": {
                    "inclusive": {
                        "request_currency": {
                            "value": value,
                        }
                    }
                }
            }
        }
    }


def test_cheapest_total_inclusive_returns_min_price(availability_payload):
    price = expedia.helpers.cheapest_total_inclusive(availability_payload)
    assert price == pytest.approx(95.50)


def test_cheapest_total_inclusive_includes_room_name(availability_payload):
    room_name, price = expedia.helpers.cheapest_total_inclusive(
        availability_payload,
        include_room_name=True,
    )
    assert room_name == "Standard Room"
    assert price == pytest.approx(95.50)


def test_extract_cheapest_rates_by_id_supports_room_names(availability_payload):
    rates_by_id = {"123": availability_payload}
    result = expedia.helpers.extract_cheapest_rates_by_id(
        rates_by_id,
        include_room_name=True,
    )
    assert result == {"123": ("Standard Room", pytest.approx(95.50))}


def test_build_rates_dataframe_creates_columns(availability_payload):
    other_payload = {
        "rooms": [
            {
                "room_name": "Standard Room",
                "rates": [_rate("88.00")],
            },
            {
                "room_name": "Suite",
                "rates": [_rate("205.00")],
            },
        ]
    }

    df = expedia.helpers.build_rates_dataframe(
        [availability_payload, other_payload],
        labels=["first_rate", "second_rate"],
    )

    assert list(df["room_name"]) == ["Standard Room", "Suite"]
    assert df["first_rate"].tolist() == [pytest.approx(95.50), pytest.approx(210.75)]
    assert df["second_rate"].tolist() == [pytest.approx(88.00), pytest.approx(205.00)]


def test_cheapest_total_inclusive_handles_missing_rates():
    payload = {"rooms": [{"room_name": "Empty Room", "rates": []}]}
    assert expedia.helpers.cheapest_total_inclusive(payload) is None
    assert (
        expedia.helpers.cheapest_total_inclusive(payload, include_room_name=True) is None
    )


def test_extract_cheapest_rates_by_id_empty_input():
    assert expedia.helpers.extract_cheapest_rates_by_id({}) == {}


def test_build_rates_dataframe_validates_input_lengths(availability_payload):
    with pytest.raises(ValueError):
        expedia.helpers.build_rates_dataframe([availability_payload], labels=["only", "extra"])


def test_destination_point_moves_north():
    # Move 1000m north from the equator; expect latitude increase ~0.009 degrees
    lat, lon = expedia.helpers.destination_point(0.0, 0.0, bearing_deg=0.0, distance_m=1000.0)
    assert lat == pytest.approx(0.00899, rel=1e-3)
    assert lon == pytest.approx(0.0, abs=1e-6)


def test_circle_polygon_geojson_structure():
    polygon = expedia.helpers.circle_polygon_geojson(10.0, 20.0, radius_m=500, n_points=8)
    assert polygon["type"] == "Polygon"
    coords = polygon["coordinates"][0]
    assert len(coords) == 9  # 8 points + closing point
    assert coords[0] == coords[-1]


def test_to_geojson_string_roundtrip():
    polygon = {"type": "Polygon", "coordinates": [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]}
    json_str = expedia.helpers.to_geojson_string(polygon, pretty=False)
    assert '"type":"Polygon"' in json_str


def test_generate_dates_and_add_days():
    dates = expedia.helpers.generate_dates("2024-01-01", "2024-01-03", step=1)
    assert dates == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert expedia.helpers.add_days("2024-01-01", 5) == "2024-01-06"


def test_property_display_name_prefers_nested_content():
    summary = {"name": {"content": {"value": "Hotel Nested"}}}
    assert expedia.helpers.property_display_name(summary) == "Hotel Nested"


def test_property_display_name_fallback():
    summary = {"something_else": "ignored"}
    assert expedia.helpers.property_display_name(summary, fallback="Fallback Hotel") == "Fallback Hotel"


def test_property_coordinates_extracts_lat_lon():
    summary = {"location": {"coordinates": {"latitude": "52.1", "longitude": "4.89"}}}
    assert expedia.helpers.property_coordinates(summary) == (pytest.approx(52.1), pytest.approx(4.89))


def test_property_coordinates_handles_missing_values():
    assert expedia.helpers.property_coordinates({}) == (None, None)
