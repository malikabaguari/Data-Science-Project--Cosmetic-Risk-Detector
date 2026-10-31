"""Collect a small, auditable Open Beauty Facts API sample or an explicit bulk CSV.

Run from the repository root, for example:
    python -m src.collection --from-labeled-csv /home/ubuntu/upload/products_labeled.csv

The default operation never crawls the API: it requires explicit barcode input and
caps live product requests at a small serial sample.  The bulk download is opt-in.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import quote

import requests

DEFAULT_BASE_URL = "https://world.openbeautyfacts.org"
PRODUCT_PATH_TEMPLATE = "/api/v2/product/{barcode}.json"
DEFAULT_BULK_CSV_URL = (
    "https://static.openbeautyfacts.org/data/"
    "en.openbeautyfacts.org.products.csv.gz"
)
DEFAULT_OUTPUT = Path("data/api/product_snapshot.json")
DEFAULT_MAX_PRODUCTS = 3
DEFAULT_TIMEOUT: tuple[float, float] = (5.0, 30.0)
DEFAULT_USER_AGENT = (
    "IronhackCosmeticRiskDetector/1.0 "
    "(educational collector; set a reachable contact before production use)"
)

# Provenance supplied for the project CSV.  It describes that existing source, not
# the scope of a product's country fields or the timestamp of a fresh API response.
LABELED_CSV_PROVENANCE = {
    "original_download_date": "2026-10-10",
    "original_source": "fr.openbeautyfacts.org export",
    "scope_note": (
        "An export sourced from fr.openbeautyfacts.org is not evidence that its "
        "products are France-only."
    ),
}

_BARCODE_RE = re.compile(r"^[0-9A-Za-z._-]{1,64}$")
_RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class CollectionError(RuntimeError):
    """Raised for local input or download errors that should stop the CLI."""


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalise_base_url(base_url: str) -> str:
    """Return an HTTPS base URL without a trailing slash."""
    base_url = base_url.strip().rstrip("/")
    if not base_url.startswith("https://"):
        raise CollectionError("--base-url must use https://")
    return base_url


def product_url(base_url: str, barcode: str) -> str:
    """Build a product endpoint URL while preserving the barcode text unchanged."""
    if not isinstance(barcode, str) or not _BARCODE_RE.fullmatch(barcode):
        raise CollectionError(
            "Barcode must be a non-empty string of up to 64 letters, digits, dot, underscore, or hyphen."
        )
    return normalise_base_url(base_url) + PRODUCT_PATH_TEMPLATE.format(
        barcode=quote(barcode, safe="")
    )


def read_labeled_barcodes(csv_path: str | Path, limit: int = DEFAULT_MAX_PRODUCTS) -> list[str]:
    """Read up to *limit* valid codes from a comma-delimited labeled CSV.

    CSV values remain strings: no integer conversion, zero-padding, or other
    barcode normalization is performed.
    """
    if limit < 1:
        raise CollectionError("limit must be at least 1")

    path = Path(csv_path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "code" not in reader.fieldnames:
                raise CollectionError(f"{path} must contain a comma-delimited 'code' column")
            barcodes: list[str] = []
            for row in reader:
                # Keep the CSV text verbatim. Whitespace-only values are ignored.
                barcode = row.get("code")
                if barcode is None or not barcode.strip():
                    continue
                if not _BARCODE_RE.fullmatch(barcode):
                    continue
                barcodes.append(barcode)
                if len(barcodes) >= limit:
                    break
    except OSError as exc:
        raise CollectionError(f"Could not read labeled CSV {path}: {exc}") from exc

    if not barcodes:
        raise CollectionError(f"No valid barcode strings found in {path}")
    return barcodes


def read_labeled_product_barcodes(csv_path: str | Path, limit: int = DEFAULT_MAX_PRODUCTS) -> list[str]:
    """Prefer named, ingredient-bearing records when sampling the labeled CSV.

    Very short legacy identifiers are retained by :func:`read_labeled_barcodes`,
    but this API-sample helper prefers the stronger local evidence available in
    the project CSV: a code of at least four characters plus product name and
    ingredient text. It never pads, casts, or otherwise changes the code string.
    If too few preferred records exist, valid codes are used as a fallback.
    """
    if limit < 1:
        raise CollectionError("limit must be at least 1")

    path = Path(csv_path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "code" not in reader.fieldnames:
                raise CollectionError(f"{path} must contain a comma-delimited 'code' column")
            preferred: list[str] = []
            fallback: list[str] = []
            for row in reader:
                barcode = row.get("code")
                if barcode is None or not barcode.strip() or not _BARCODE_RE.fullmatch(barcode):
                    continue
                if barcode not in fallback:
                    fallback.append(barcode)
                if (
                    len(barcode) >= 4
                    and bool((row.get("product_name") or "").strip())
                    and bool((row.get("ingredients_text") or "").strip())
                    and barcode not in preferred
                ):
                    preferred.append(barcode)
                    if len(preferred) >= limit:
                        break
    except OSError as exc:
        raise CollectionError(f"Could not read labeled CSV {path}: {exc}") from exc

    selected = (preferred + [barcode for barcode in fallback if barcode not in preferred])[:limit]
    if not selected:
        raise CollectionError(f"No valid barcode strings found in {path}")
    return selected


def _retry_delay(response: Any | None, attempt_index: int, random_fn: Callable[[], float]) -> float:
    """Return a capped exponential delay, honoring integer Retry-After when present."""
    retry_after: str | None = None
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), 60.0)
        except (TypeError, ValueError):
            pass
    # 0.5, 1, 2 seconds plus small jitter; bounded so a CLI cannot stall indefinitely.
    return min((0.5 * (2**attempt_index)) + (0.25 * random_fn()), 30.0)


def _get_with_retry(
    session: requests.Session,
    url: str,
    *,
    user_agent: str,
    timeout: tuple[float, float],
    max_retries: int,
    stream: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] = random.random,
) -> tuple[Any | None, int, str | None]:
    """Perform one GET with bounded retry on network errors, 429, and selected 5xx."""
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response = session.get(
                url,
                headers={"User-Agent": user_agent, "Accept": "application/json"},
                timeout=timeout,
                stream=stream,
            )
        except requests.RequestException as exc:
            response = None
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code not in _RETRIABLE_STATUS_CODES:
                return response, attempt + 1, None
            last_error = f"HTTP {response.status_code}"

        if attempt < max_retries:
            sleep(_retry_delay(response, attempt, random_fn))
            if response is not None and hasattr(response, "close"):
                response.close()

    return response, max_retries + 1, last_error


def fetch_product(
    barcode: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    session: requests.Session | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    max_retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] = random.random,
) -> dict[str, Any]:
    """Fetch one product and return its response or an honest diagnostic record."""
    if max_retries < 0:
        raise CollectionError("max_retries cannot be negative")

    endpoint = product_url(base_url, barcode)
    client = session if session is not None else requests.Session()
    response, attempts, request_error = _get_with_retry(
        client,
        endpoint,
        user_agent=user_agent,
        timeout=timeout,
        max_retries=max_retries,
        sleep=sleep,
        random_fn=random_fn,
    )

    record: dict[str, Any] = {
        "barcode": barcode,
        "request_url": endpoint,
        "retrieved_at_utc": utc_now(),
        "attempts": attempts,
        "status_code": getattr(response, "status_code", None),
        "outcome": "network_error" if response is None else "http_error",
        "response": None,
        "error": request_error,
    }
    if response is None:
        return record

    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        record["outcome"] = "invalid_json"
        record["error"] = f"JSON decode error: {exc}"
        return record
    finally:
        if hasattr(response, "close"):
            response.close()

    record["response"] = payload
    if 200 <= response.status_code < 300:
        api_status = payload.get("status") if isinstance(payload, Mapping) else None
        if api_status == 1:
            record["outcome"] = "ok"
            record["error"] = None
        else:
            record["outcome"] = "api_not_found"
            record["error"] = f"Open Beauty Facts API status {api_status!r}"
    elif record["error"] is None:
        record["error"] = f"HTTP {response.status_code}"
    return record


def collect_product_snapshot(
    barcodes: Sequence[str],
    *,
    base_url: str = DEFAULT_BASE_URL,
    session: requests.Session | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    max_retries: int = 2,
    input_metadata: Mapping[str, Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] = random.random,
) -> dict[str, Any]:
    """Collect a serial, small product sample suitable for an auditable snapshot."""
    if not barcodes:
        raise CollectionError("At least one barcode is required")
    if len(barcodes) > DEFAULT_MAX_PRODUCTS:
        raise CollectionError(
            f"Live API samples are capped at {DEFAULT_MAX_PRODUCTS} products; use the explicit bulk export for larger work."
        )

    canonical_base = normalise_base_url(base_url)
    client = session if session is not None else requests.Session()
    # Intentionally a normal for-loop: no concurrency and no API crawling behavior.
    records = [
        fetch_product(
            barcode,
            base_url=canonical_base,
            session=client,
            user_agent=user_agent,
            timeout=timeout,
            max_retries=max_retries,
            sleep=sleep,
            random_fn=random_fn,
        )
        for barcode in barcodes
    ]
    return {
        "snapshot_schema_version": 1,
        "collection_type": "api_product_sample",
        "metadata": {
            "collected_at_utc": utc_now(),
            "source_base_url": canonical_base,
            "source_product_endpoint_template": canonical_base + PRODUCT_PATH_TEMPLATE,
            "request_policy": {
                "serial": True,
                "live_product_cap": DEFAULT_MAX_PRODUCTS,
                "timeout_seconds": {"connect": timeout[0], "read": timeout[1]},
                "max_retries": max_retries,
                "retry_statuses": sorted(_RETRIABLE_STATUS_CODES),
                "user_agent": user_agent,
            },
            "input": dict(input_metadata or {}),
        },
        "results": records,
    }


def write_json(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write JSON atomically enough for an interrupted local CLI run."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    temporary.replace(output)
    return output


def download_bulk_csv(
    output_path: str | Path,
    *,
    bulk_url: str = DEFAULT_BULK_CSV_URL,
    session: requests.Session | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    max_retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] = random.random,
) -> dict[str, Any]:
    """Download the official cosmetics CSV only when explicitly requested.

    A sibling ``.metadata.json`` file records the source URL, status and checksum.
    This function is intentionally separate from product API sampling.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    client = session if session is not None else requests.Session()
    response, attempts, error = _get_with_retry(
        client,
        bulk_url,
        user_agent=user_agent,
        timeout=timeout,
        max_retries=max_retries,
        stream=True,
        sleep=sleep,
        random_fn=random_fn,
    )
    metadata: dict[str, Any] = {
        "collection_type": "bulk_csv_export",
        "retrieved_at_utc": utc_now(),
        "source_url": bulk_url,
        "destination": str(destination),
        "attempts": attempts,
        "status_code": getattr(response, "status_code", None),
        "error": error,
    }
    try:
        if response is None:
            raise CollectionError(f"Bulk CSV request failed: {error}")
        if not 200 <= response.status_code < 300:
            raise CollectionError(f"Bulk CSV request returned HTTP {response.status_code}")

        temporary = destination.with_name(destination.name + ".part")
        digest = hashlib.sha256()
        byte_count = 0
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
        temporary.replace(destination)
        metadata.update(
            {
                "outcome": "ok",
                "bytes_written": byte_count,
                "sha256": digest.hexdigest(),
                "response_headers": {
                    key: response.headers[key]
                    for key in ("Content-Type", "Content-Length", "Last-Modified", "ETag")
                    if key in response.headers
                },
            }
        )
        return metadata
    finally:
        if response is not None and hasattr(response, "close"):
            response.close()
        # Persist metadata even on a failure so a batch run has a truthful diagnostic.
        write_json(metadata, str(destination) + ".metadata.json")


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a small Open Beauty Facts API sample or explicitly download its cosmetics CSV export."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--barcodes", nargs="+", help="One to three barcode strings to fetch serially.")
    source.add_argument(
        "--from-labeled-csv",
        metavar="PATH",
        help="Read up to three preferred named, ingredient-bearing barcode strings from the labeled CSV.",
    )
    source.add_argument(
        "--bulk-csv",
        metavar="OUTPUT_PATH",
        help="Explicitly download the official cosmetics bulk CSV to OUTPUT_PATH; never the default action.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Snapshot JSON path for API samples.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Open Beauty Facts base URL.")
    parser.add_argument("--bulk-url", default=DEFAULT_BULK_CSV_URL, help="Bulk cosmetics CSV source URL.")
    parser.add_argument("--limit", type=_positive_int, default=DEFAULT_MAX_PRODUCTS, help="CSV sample limit (1-3).")
    parser.add_argument("--timeout", type=_positive_int, default=int(DEFAULT_TIMEOUT[1]), help="Read timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries after a timeout, 429, or selected 5xx response.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="Custom API User-Agent; use a reachable contact for production.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if args.limit > DEFAULT_MAX_PRODUCTS:
        parser.error(f"--limit cannot exceed {DEFAULT_MAX_PRODUCTS} for live API samples")

    timeout = (DEFAULT_TIMEOUT[0], float(args.timeout))
    try:
        if args.bulk_csv:
            metadata = download_bulk_csv(
                args.bulk_csv,
                bulk_url=args.bulk_url,
                user_agent=args.user_agent,
                timeout=timeout,
                max_retries=args.max_retries,
            )
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
            return 0

        if args.from_labeled_csv:
            barcodes = read_labeled_product_barcodes(args.from_labeled_csv, limit=args.limit)
            input_metadata: dict[str, Any] = {
                "type": "provided_labeled_csv",
                "path": str(Path(args.from_labeled_csv)),
                "selected_barcodes": barcodes,
                "provenance": LABELED_CSV_PROVENANCE,
            }
        else:
            barcodes = list(args.barcodes)
            if len(barcodes) > DEFAULT_MAX_PRODUCTS:
                raise CollectionError(
                    f"At most {DEFAULT_MAX_PRODUCTS} --barcodes values are allowed in a live sample."
                )
            input_metadata = {"type": "explicit_barcodes", "selected_barcodes": barcodes}

        snapshot = collect_product_snapshot(
            barcodes,
            base_url=args.base_url,
            user_agent=args.user_agent,
            timeout=timeout,
            max_retries=args.max_retries,
            input_metadata=input_metadata,
        )
        output = write_json(snapshot, args.output)
        statuses = [result["status_code"] for result in snapshot["results"]]
        print(json.dumps({"output": str(output), "http_statuses": statuses}, ensure_ascii=False))
        return 0
    except CollectionError as exc:
        print(f"collection error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
