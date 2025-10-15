import json
from datetime import date

import pandas as pd
import pytest

from expedia.geo_helpers import haversine_distance
from expedia.helpers import rates_report


class DummyClient:
    pass  # client instance is only passed through to the fetch hook


def test_generate_rates_dataframe_applies_discount(monkeypatch):
    called_kwargs = {}

    def fake_fetch(client, **kwargs):
        called_kwargs.update(kwargs)
        df = pd.DataFrame(
            [
                {"Property": "Hotel A", "Price (Public Package)": 200.0, "Price (Private Prepay)": 150.0},
                {"Property": "Hotel B", "Price (Public Package)": 240.0, "Price (Private Prepay)": 240.0},
            ]
        )
        return {
            "dataframe": df,
            "rate_type_labels": {"mkt_prepay": "Public Package", "priv_pkg": "Private Prepay"},
            "rate_types": ["mkt_prepay", "priv_pkg"],
            "property_ids": ["1", "2"],
            "polygon": {"type": "Polygon"},
            "checkin": "2024-09-01",
            "checkout": "2024-09-02",
        }

    monkeypatch.setattr(rates_report, "fetch_rates_near_coordinate", fake_fetch)
    client = DummyClient()

    df, meta = rates_report.generate_rates_dataframe(
        client,
        latitude=52.0,
        longitude=4.9,
        radius_km=3,
        checkin="2024-09-01",
        checkout="2024-09-02",
    )

    assert "Discount (%)" in df.columns
    assert df.loc[0, "Property"] == "Hotel A"
    assert df.loc[0, "Discount (%)"] == pytest.approx(25.0)
    assert called_kwargs["center_lat"] == 52.0
    assert meta["property_ids"] == ["1", "2"]


def test_generate_rates_dataframe_rejects_invalid_dates():
    client = DummyClient()
    with pytest.raises(ValueError):
        rates_report.generate_rates_dataframe(
            client,
            latitude=0.0,
            longitude=0.0,
            radius_km=1,
            checkin=date(2024, 1, 5),
            checkout=date(2024, 1, 4),
        )


def test_export_rates_to_excel(tmp_path, monkeypatch):
    pytest.importorskip("openpyxl")
    df = pd.DataFrame({"Property": ["A"], "Price (Public Package)": [100.0]})
    output = tmp_path / "report.xlsx"
    saved_path = rates_report.export_rates_to_excel(df, output)
    assert saved_path == output
    assert output.exists()


def test_generate_rates_payload_returns_jsonable_structure(monkeypatch):
    fake_df = pd.DataFrame(
        [
            {"Property": "Hotel A", "Price (Public Package)": 200.0, "Price (Private Prepay)": 180.0},
        ]
    )
    fake_meta = {
        "checkin": "2024-09-01",
        "checkout": "2024-09-02",
        "rate_types": ["mkt_prepay", "priv_pkg"],
        "rate_type_labels": {"mkt_prepay": "Public Package", "priv_pkg": "Private Prepay"},
        "property_ids": ["1"],
        "polygon": {"type": "Polygon"},
    }

    def fake_generate(*args, **kwargs):
        return fake_df, fake_meta

    monkeypatch.setattr(rates_report, "generate_rates_dataframe", fake_generate)

    payload = rates_report.generate_rates_payload(
        DummyClient(),
        latitude=1.0,
        longitude=2.0,
        radius_km=3.0,
        checkin="2024-09-01",
        checkout="2024-09-02",
        event_details={"title": "Sample Event"},
    )

    assert payload["event"] == {"title": "Sample Event"}
    assert payload["search"]["rate_types"] == ["mkt_prepay", "priv_pkg"]
    assert payload["properties"][0]["Property"] == "Hotel A"


def test_export_rates_to_json(tmp_path):
    payload = {"hello": "world"}
    output = tmp_path / "payload.json"
    saved = rates_report.export_rates_to_json(payload, output, pretty=True)
    assert saved == output
    content = json.loads(output.read_text())
    assert content == payload


def test_evaluate_deal_quality_identifies_good_deal():
    df = pd.DataFrame(
        [
            {
                "Property": "Hotel A",
                "Price (Public Package)": 200.0,
                "Price (Private Prepay)": 150.0,
                "Delta (Public Package - Private Prepay)": 50.0,
                "Discount (%)": 25.0,
            },
            {
                "Property": "Hotel B",
                "Price (Public Package)": 220.0,
                "Price (Private Prepay)": 210.0,
                "Delta (Public Package - Private Prepay)": 10.0,
                "Discount (%)": 4.5,
            },
        ]
    )
    result = rates_report.evaluate_deal_quality(
        df,
        rate_types=["mkt_prepay", "priv_pkg"],
        rate_type_labels={"mkt_prepay": "Public Package", "priv_pkg": "Private Prepay"},
        discount_threshold_pct=20.0,
        savings_threshold=40.0,
        max_total_cost=180.0,
    )

    assert result["is_good_deal"] is True
    assert result["best_property"] == "Hotel A"
    assert result["best_discount_pct"] == pytest.approx(25.0)
    assert result["best_savings"] == pytest.approx(50.0)
    assert result["best_total_cost"] == pytest.approx(150.0)
    assert result["qualifying_count"] == 1
    assert result["qualifying_properties"] == ["Hotel A"]


def test_evaluate_deal_quality_flags_high_cost():
    df = pd.DataFrame(
        [
            {
                "Property": "Hotel A",
                "Price (Public Package)": 200.0,
                "Price (Private Prepay)": 150.0,
                "Delta (Public Package - Private Prepay)": 50.0,
                "Discount (%)": 25.0,
            }
        ]
    )
    result = rates_report.evaluate_deal_quality(
        df,
        rate_types=["mkt_prepay", "priv_pkg"],
        rate_type_labels={"mkt_prepay": "Public Package", "priv_pkg": "Private Prepay"},
        discount_threshold_pct=20.0,
        savings_threshold=40.0,
        max_total_cost=140.0,
    )

    assert result["is_good_deal"] is False
    assert "exceeds the limit" in result["reason"]
    assert result["qualifying_count"] == 0
    assert result["qualifying_properties"] == []


def test_evaluate_deal_quality_chooses_next_best_qualifying_property():
    df = pd.DataFrame(
        [
            {
                "Property": "Hotel A",
                "Price (Public Package)": 200.0,
                "Price (Private Prepay)": 160.0,
                "Delta (Public Package - Private Prepay)": 40.0,
                "Discount (%)": 20.0,
            },
            {
                "Property": "Hotel B",
                "Price (Public Package)": 220.0,
                "Price (Private Prepay)": 140.0,
                "Delta (Public Package - Private Prepay)": 80.0,
                "Discount (%)": 36.36,
            },
        ]
    )

    # Set thresholds so Hotel A fails on savings, Hotel B qualifies
    result = rates_report.evaluate_deal_quality(
        df,
        rate_types=["mkt_prepay", "priv_pkg"],
        rate_type_labels={"mkt_prepay": "Public Package", "priv_pkg": "Private Prepay"},
        discount_threshold_pct=30.0,
        savings_threshold=60.0,
        max_total_cost=150.0,
    )

    assert result["is_good_deal"] is True
    assert result["best_property"] == "Hotel B"
    assert result["best_savings"] == pytest.approx(80.0)
    assert result["best_total_cost"] == pytest.approx(140.0)


def test_generate_rates_dataframe_adds_distance_column(monkeypatch):
    called_kwargs = {}

    def fake_fetch(client, **kwargs):
        called_kwargs.update(kwargs)
        df = pd.DataFrame(
            [
                {
                    "Property": "Hotel A",
                    "Price (Public Package)": 200.0,
                    "Price (Private Prepay)": 150.0,
                    "latitude": 52.0,
                    "longitude": 4.9,
                },
                {
                    "Property": "Hotel B",
                    "Price (Public Package)": 210.0,
                    "Price (Private Prepay)": 160.0,
                    "latitude": 52.01,
                    "longitude": 4.92,
                },
            ]
        )
        return {
            "dataframe": df,
            "rate_type_labels": {"mkt_prepay": "Public Package", "priv_pkg": "Private Prepay"},
            "rate_types": ["mkt_prepay", "priv_pkg"],
            "property_ids": ["1", "2"],
            "polygon": {"type": "Polygon"},
            "checkin": "2024-09-01",
            "checkout": "2024-09-02",
        }

    monkeypatch.setattr(rates_report, "fetch_rates_near_coordinate", fake_fetch)
    client = DummyClient()

    df, _ = rates_report.generate_rates_dataframe(
        client,
        latitude=52.0,
        longitude=4.9,
        radius_km=3,
        checkin="2024-09-01",
        checkout="2024-09-02",
    )

    assert "Distance to Center (km)" in df.columns
    df_by_property = df.set_index("Property")
    assert df_by_property.loc["Hotel A", "Distance to Center (km)"] == pytest.approx(0.0, abs=1e-6)

    expected_distance = haversine_distance(52.0, 4.9, 52.01, 4.92) / 1000.0
    assert df_by_property.loc["Hotel B", "Distance to Center (km)"] == pytest.approx(expected_distance, rel=1e-6)


def test_generate_rates_payload_with_retry_retries_on_rate_limit(monkeypatch):
    attempts = {"count": 0}

    def fake_generate(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise rates_report.ExpediaAPIError("Expedia API error (429) rate limit exceeded")
        return {"properties": [], "search": {}}

    sleeps = []

    monkeypatch.setattr(rates_report, "generate_rates_payload", fake_generate)
    monkeypatch.setattr(rates_report.time, "sleep", lambda seconds: sleeps.append(seconds))

    payload = rates_report.generate_rates_payload_with_retry(
        DummyClient(),
        latitude=1.0,
        longitude=2.0,
        radius_km=3.0,
        checkin="2024-09-01",
        checkout="2024-09-02",
        max_attempts=3,
        initial_backoff_seconds=1.0,
        backoff_multiplier=2.0,
    )

    assert payload == {"properties": [], "search": {}}
    assert attempts["count"] == 2
    assert sleeps == [1.0]


def test_generate_rates_payload_with_retry_raises_after_exhausting(monkeypatch):
    def fake_generate(*args, **kwargs):
        raise rates_report.ExpediaAPIError("Expedia API error (500) unexpected")

    monkeypatch.setattr(rates_report, "generate_rates_payload", fake_generate)
    monkeypatch.setattr(rates_report.time, "sleep", lambda seconds: (_ for _ in ()).throw(AssertionError("sleep should not be called")))

    with pytest.raises(rates_report.ExpediaAPIError):
        rates_report.generate_rates_payload_with_retry(
            DummyClient(),
            latitude=1.0,
            longitude=2.0,
            radius_km=3.0,
            checkin="2024-09-01",
            checkout="2024-09-02",
            max_attempts=2,
            initial_backoff_seconds=1.0,
        )


def test_find_cached_rates_dataframe_matches_map_key(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_data = [
        {
            "title": "Sample Event",
            "map_key": "ABC123",
            "venue_id": 42,
            "latitude": 10.0,
            "longitude": 20.0,
            "checkin": "2025-09-01",
            "checkout": "2025-09-03",
            "hotels": [
                {"Property": "Hotel A", "Property ID": "1", "Price (Public Package)": 200.0},
                {"Property": "Hotel B", "Property ID": "2", "Price (Public Package)": 250.0},
            ],
            "rates_search": {
                "rate_types": ["mkt_prepay", "priv_pkg"],
                "rate_type_labels": {"mkt_prepay": "Public Package", "priv_pkg": "Private Prepay"},
                "latitude": 10.0,
                "longitude": 20.0,
                "radius_km": 3.0,
                "occupancy": 2,
            },
        }
    ]
    cache_path.write_text(json.dumps(cache_data))

    df, metadata = rates_report.find_cached_rates_dataframe(
        cache_path=cache_path,
        map_key="abc123",
        latitude=10.0,
        longitude=20.0,
        checkin="2025-09-01",
        checkout="2025-09-03",
        radius_km=3.0,
    )

    assert set(df["Property"]) == {"Hotel A", "Hotel B"}
    assert metadata["source"] == "cache"
    assert metadata["rate_types"] == ["mkt_prepay", "priv_pkg"]
    assert metadata["property_ids"] == ["1", "2"]


def test_find_cached_rates_dataframe_returns_none_when_missing(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("[]")

    result = rates_report.find_cached_rates_dataframe(
        cache_path=cache_path,
        map_key="missing",
        checkin="2025-01-01",
        checkout="2025-01-02",
        radius_km=3.0,
    )

    assert result is None


def test_find_cached_rates_dataframe_skips_mismatched_radius(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_data = [
        {
            "title": "Sample Event",
            "map_key": "ABC123",
            "latitude": 10.0,
            "longitude": 20.0,
            "checkin": "2025-09-01",
            "checkout": "2025-09-03",
            "hotels": [],
            "rates_search": {"radius_km": 3.0},
        }
    ]
    cache_path.write_text(json.dumps(cache_data))

    result = rates_report.find_cached_rates_dataframe(
        cache_path=cache_path,
        map_key="abc123",
        latitude=10.0,
        longitude=20.0,
        checkin="2025-09-01",
        checkout="2025-09-03",
        radius_km=5.0,
    )

    assert result is None


def test_find_cached_rates_dataframe_filters_by_distance(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_data = [
        {
            "title": "Sample Event",
            "map_key": "ABC123",
            "latitude": 10.0,
            "longitude": 20.0,
            "checkin": "2025-09-01",
            "checkout": "2025-09-03",
            "hotels": [
                {"Property": "Near Hotel", "Property ID": "1", "Distance to Center (km)": 2.5},
                {"Property": "Far Hotel", "Property ID": "2", "Distance to Center (km)": 4.5},
            ],
            "rates_search": {"radius_km": 5.0},
        }
    ]
    cache_path.write_text(json.dumps(cache_data))

    df, metadata = rates_report.find_cached_rates_dataframe(
        cache_path=cache_path,
        map_key="abc123",
        latitude=10.0,
        longitude=20.0,
        checkin="2025-09-01",
        checkout="2025-09-03",
        radius_km=3.0,
    )

    assert list(df["Property"]) == ["Near Hotel"]
    assert metadata["cached_radius_km"] == pytest.approx(5.0)
    assert metadata["requested_radius_km"] == pytest.approx(3.0)
