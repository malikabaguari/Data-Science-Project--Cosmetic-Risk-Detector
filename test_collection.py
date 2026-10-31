"""Unit tests for src.collection. All HTTP behaviour is supplied by fakes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src import collection


class FakeResponse:
    def __init__(self, status_code: int, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.closed = False

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def close(self):
        self.closed = True


class FakeSession:
    """A strict fake: no real HTTP implementation exists in these tests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected HTTP call")
        return self.responses.pop(0)


class CollectionTests(unittest.TestCase):
    def test_labeled_csv_keeps_leading_zero_barcode_as_a_string(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "products.csv"
            csv_path.write_text("code,product_name\n001234,Sample\n42,Other\n", encoding="utf-8")
            self.assertEqual(collection.read_labeled_barcodes(csv_path, limit=2), ["001234", "42"])

    def test_preferred_csv_selection_uses_named_ingredient_bearing_record_without_casting(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "products.csv"
            csv_path.write_text(
                "code,product_name,ingredients_text\n"
                "14,Short legacy,Aqua\n"
                "001234,Complete product,Aqua\n"
                "9999,Name only,\n",
                encoding="utf-8",
            )
            self.assertEqual(collection.read_labeled_product_barcodes(csv_path), ["001234", "14", "9999"])

    def test_fetch_retries_429_then_records_actual_success_json(self):
        session = FakeSession(
            [
                FakeResponse(429, {"status": 0}, {"Retry-After": "0"}),
                FakeResponse(200, {"code": "001234", "status": 1, "product": {"product_name": "Sample"}}),
            ]
        )
        sleeps = []
        record = collection.fetch_product(
            "001234",
            session=session,
            sleep=sleeps.append,
            random_fn=lambda: 0.0,
        )

        self.assertEqual(record["outcome"], "ok")
        self.assertEqual(record["status_code"], 200)
        self.assertEqual(record["attempts"], 2)
        self.assertEqual(record["response"]["code"], "001234")
        self.assertEqual(sleeps, [0.0])
        self.assertEqual(
            session.calls[0][0],
            "https://world.openbeautyfacts.org/api/v2/product/001234.json",
        )
        self.assertIn("IronhackCosmeticRiskDetector", session.calls[0][1]["headers"]["User-Agent"])

    def test_http_200_with_api_status_zero_is_an_honest_not_found_diagnostic(self):
        session = FakeSession([FakeResponse(200, {"code": "00000014", "status": 0})])
        record = collection.fetch_product("14", session=session, sleep=lambda _: None)

        self.assertEqual(record["outcome"], "api_not_found")
        self.assertEqual(record["status_code"], 200)
        self.assertEqual(record["response"], {"code": "00000014", "status": 0})
        self.assertEqual(record["error"], "Open Beauty Facts API status 0")
        self.assertNotIn("label", record)
        self.assertNotIn("risk", record)

    def test_snapshot_reports_http_error_without_inventing_a_product_or_label(self):
        session = FakeSession([FakeResponse(404, {"status": 0, "status_verbose": "product not found"})])
        snapshot = collection.collect_product_snapshot(
            ["does-not-exist"],
            session=session,
            sleep=lambda _: None,
        )

        result = snapshot["results"][0]
        self.assertEqual(result["outcome"], "http_error")
        self.assertEqual(result["status_code"], 404)
        self.assertEqual(result["response"], {"status": 0, "status_verbose": "product not found"})
        self.assertNotIn("label", result)
        self.assertNotIn("risk", result)

    def test_snapshot_refuses_more_than_small_serial_cap(self):
        with self.assertRaises(collection.CollectionError):
            collection.collect_product_snapshot(["1", "2", "3", "4"], session=FakeSession([]))


if __name__ == "__main__":
    unittest.main()
