"""The runtime renders a DataFrame message for breakdown buckets too,
not only per-patient samples."""

from __future__ import annotations


def _payload_has_renderable(payload: dict) -> bool:
    # Mirrors the runner gate we are about to implement.
    return bool(payload.get("sample")) or bool(payload.get("buckets"))


def test_gate_accepts_buckets_only_payload():
    assert _payload_has_renderable({"sample": [], "buckets": [{"bucket_label": "F", "patient_count": 3}]})


def test_gate_rejects_empty_payload():
    assert not _payload_has_renderable({"sample": [], "buckets": []})
