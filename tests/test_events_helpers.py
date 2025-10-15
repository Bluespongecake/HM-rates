import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from events.event_helpers import (
    load_events_catalog,
    load_events_from_file,
    load_events_from_json,
    load_events_from_excel,
)


def _write_event_json(path: Path, **overrides):
    record = {
        "title": "Sample Conference",
        "city": "Amsterdam",
        "country": "NL",
        "latitude": 52.37,
        "longitude": 4.89,
        "date_start": (date.today() + timedelta(days=10)).isoformat(),
        "date_end": (date.today() + timedelta(days=12)).isoformat(),
        "map_key": "abc123",
        "venue_id": "venue-1",
    }
    record.update(overrides)
    path.write_text(json.dumps(record))


def test_load_events_from_file_json(tmp_path):
    json_path = tmp_path / "events.json"
    _write_event_json(json_path)

    records = load_events_from_file(json_path)
    assert len(records) == 1
    assert records[0]["title"] == "Sample Conference"
    assert records[0]["source"] == str(json_path)


def test_load_events_catalog_deduplicates_and_formats(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    future_start = date.today() + timedelta(days=30)
    future_end = future_start + timedelta(days=2)

    json_path = data_dir / "events_1.json"
    json_records = [
        {
            "title": "Event One",
            "city": "Berlin",
            "country": "DE",
            "latitude": 52.5,
            "longitude": 13.4,
            "date_start": future_start.isoformat(),
            "date_end": future_end.isoformat(),
            "map_key": "MK1",
            "venue_id": "V1",
        },
        {
            "title": "Event One",  # duplicate should be dropped
            "city": "Berlin",
            "country": "DE",
            "latitude": 52.5,
            "longitude": 13.4,
            "date_start": future_start.isoformat(),
            "date_end": future_end.isoformat(),
            "map_key": "MK1",
            "venue_id": "V1",
        },
    ]
    json_path.write_text(json.dumps(json_records))

    excel_path = data_dir / "events_extra.xlsx"
    pytest.importorskip("openpyxl")
    excel_df = pd.DataFrame(
        [
            {
                "title": "Event Two",
                "city": "Paris",
                "country": "FR",
                "lat": 48.8566,
                "lon": 2.3522,
                "date_start": (future_start + timedelta(days=5)),
                "date_end": (future_end + timedelta(days=5)),
                "map_key": "MK2",
                "venue_id": "V2",
            }
        ]
    )
    excel_df.to_excel(excel_path, index=False)

    df = load_events_catalog([data_dir])
    assert len(df) == 2  # duplicate removed
    assert {"Event One", "Event Two"} == set(df["title"])
    assert all(df["date_end"] >= pd.Timestamp(date.today()))
    assert df["display_label"].str.contains("Event One").any()
    assert df["search_blob"].str.contains("mk2").any()


@pytest.mark.parametrize(
    "suffix,loader",
    [
        (".json", load_events_from_json),
        (".xlsx", load_events_from_excel),
    ],
)
def test_load_events_from_file_dispatch(tmp_path, suffix, loader, monkeypatch):
    path = tmp_path / f"event{suffix}"
    if suffix == ".json":
        _write_event_json(path)
    else:
        pytest.importorskip("openpyxl")
        pd.DataFrame(
            [
                {
                    "title": "Excel Event",
                    "latitude": 1.0,
                    "longitude": 2.0,
                    "date_start": (date.today() + timedelta(days=1)),
                }
            ]
        ).to_excel(path, index=False)

    spy = []

    def wrapped(path_arg):
        spy.append(path_arg)
        return loader(path_arg)

    monkeypatch.setattr(
        "events.event_helpers.load_events_from_json" if suffix == ".json" else "events.event_helpers.load_events_from_excel",
        wrapped,
    )

    load_events_from_file(path)
    assert spy and spy[0] == path
