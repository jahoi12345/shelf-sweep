"""Provider: fx. One dated, cited, disk-cached reference rate: ECB euro foreign-exchange reference rates (EUR -> USD, and cross rates through EUR).

The rate is evidence, not a price source. Two uses:
* ``eur_usd`` / ``to_usd``: REPORT a native-currency (EUR) second-market price also in USD (never summed into the USD totals);
* ``usd_rate`` / ``convert_to_usd`` / ``check_conversion``: the CONVERTED fallback line (user decision 2026-10-07): when no US seller offers the equivalent edition, a
  non-US market offer (EUR, GBP) is converted to USD at the ECB reference rate; the line keeps the native amount and currency, the rate, its publication date, the
  source URL, the retrieval time and the sha256 of the raw ECB bytes, and ``check_conversion`` recomputes the USD amount from those raw bytes.
Offline with no cached rate -> None (no USD figure, no converted line).
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import httpx

ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
SOURCE = "European Central Bank euro foreign exchange reference rates (daily)"


def parse_ecb(xml_text: str, currency: str = "USD"):
    """-> (rate, date) from the ECB daily XML, or None. Rate = units of ``currency`` per 1 EUR."""
    day = re.search(r"<Cube\s+time=['\"](\d{4}-\d{2}-\d{2})['\"]", xml_text)
    rate = re.search(r"<Cube\s+currency=['\"]" + re.escape(currency) + r"['\"]\s+rate=['\"]([0-9.]+)['\"]", xml_text)
    if not day or not rate:
        return None
    value = Decimal(rate.group(1))
    return (value, day.group(1)) if value > 0 else None


def _ecb_bytes(evidence_dir, *, online=False, cache_dir=None, max_age_hours=24, url=ECB_URL, transport=None):
    """-> (raw bytes, meta {url, retrieved_at, http_status, sha256}, kept file name under ``evidence_dir``) or None. Disk cache first (``max_age_hours``)."""
    evidence_dir = Path(evidence_dir); evidence_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir or evidence_dir); cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    meta_path, raw_path = cache / f"fx_{key}.json", cache / f"fx_{key}.xml"
    content, meta = None, None
    if meta_path.is_file() and raw_path.is_file():
        m = json.loads(meta_path.read_text())
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(m["retrieved_at"])).total_seconds() / 3600
        if m.get("url") == url and 0 <= age <= max_age_hours and hashlib.sha256(raw_path.read_bytes()).hexdigest() == m["sha256"]:
            content, meta = raw_path.read_bytes(), m
    if content is None:
        if not online:
            return None
        try:
            response = httpx.get(url, timeout=15, follow_redirects=False, headers={"User-Agent": "LibraryClaimSourceProbe/1.0"}, **({"transport": transport} if transport else {}))
            response.raise_for_status()
            if response.status_code != 200:
                return None
        except Exception:  # noqa: BLE001 - no rate: the caller reports no USD figure
            return None
        content = response.content
        meta = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "http_status": response.status_code, "sha256": hashlib.sha256(content).hexdigest()}
        raw_path.write_bytes(content); meta_path.write_text(json.dumps(meta, indent=2))
    kept = evidence_dir / f"fx_{meta['sha256']}.xml"
    kept.write_bytes(content)
    return content, meta, kept.name


def eur_usd(evidence_dir, *, online=False, cache_dir=None, max_age_hours=24, url=ECB_URL, transport=None):
    """-> fx evidence dict {rate, base, quote, rate_date, source, url, retrieved_at, sha256, raw_path} or None. Raw bytes are copied under ``evidence_dir`` (sha256-named)."""
    got = _ecb_bytes(evidence_dir, online=online, cache_dir=cache_dir, max_age_hours=max_age_hours, url=url, transport=transport)
    if not got:
        return None
    content, meta, kept = got
    parsed = parse_ecb(content.decode("utf-8", "replace"))
    if not parsed:
        return None
    rate, day = parsed
    return {"rate": float(rate), "base": "EUR", "quote": "USD", "rate_date": day, "source": SOURCE, "url": url, "retrieved_at": meta["retrieved_at"],
            "sha256": meta["sha256"], "raw_path": kept}


def to_usd(amount_eur, fx):
    """USD equivalent of a native EUR amount at the evidence rate, rounded to cents (half up). None without evidence."""
    if not fx:
        return None
    return float((Decimal(str(amount_eur)) * Decimal(str(fx["rate"]))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# ----------------------------------------------------------------------------------------------------------------- converted fallback lines (EUR, GBP -> USD)
def ecb_usd_per_unit(xml_text: str, currency: str):
    """-> (USD per 1 unit of ``currency`` as an exact Decimal ratio of the published ECB rates, the ECB rates used {code: str}, rate date) or None.
    EUR: the published USD rate. Another currency: the cross rate through EUR (USD per EUR / ``currency`` per EUR)."""
    usd = parse_ecb(xml_text, "USD")
    if not usd:
        return None
    if currency == "EUR":
        return usd[0], {"USD": str(usd[0])}, usd[1]
    other = parse_ecb(xml_text, currency)
    if not other or other[1] != usd[1]:
        return None
    return usd[0] / other[0], {"USD": str(usd[0]), currency: str(other[0])}, usd[1]


def usd_rate(currency, evidence_dir, *, online=False, cache_dir=None, max_age_hours=24, url=ECB_URL, transport=None):
    """-> fx evidence for converting ``currency`` (EUR, GBP, ...) to USD, or None: {base, quote, rate (USD per unit, 6 dp, display), ecb_rates_per_eur, method, rate_date,
    source, url, retrieved_at, sha256, raw_path}. The conversion itself always uses the exact published ECB rates (``convert_to_usd``)."""
    got = _ecb_bytes(evidence_dir, online=online, cache_dir=cache_dir, max_age_hours=max_age_hours, url=url, transport=transport)
    if not got:
        return None
    content, meta, kept = got
    parsed = ecb_usd_per_unit(content.decode("utf-8", "replace"), currency)
    if not parsed:
        return None
    rate, used, day = parsed
    method = ("ECB USD per EUR reference rate" if currency == "EUR" else f"ECB cross rate through EUR: (USD per EUR {used['USD']}) / ({currency} per EUR {used[currency]})")
    return {"base": currency, "quote": "USD", "rate": float(rate.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)), "ecb_rates_per_eur": used, "method": method,
            "rate_date": day, "source": SOURCE, "url": url, "retrieved_at": meta["retrieved_at"], "sha256": meta["sha256"], "raw_path": kept}


def convert_to_usd(amount, fx):
    """USD amount of a native ``amount`` in ``fx['base']`` at the exact published ECB rates of ``fx``, rounded to cents (half up). None without evidence."""
    if not fx or not fx.get("ecb_rates_per_eur"):
        return None
    rates = fx["ecb_rates_per_eur"]
    value = Decimal(str(amount)) * Decimal(rates["USD"])
    if fx["base"] != "EUR":
        value = value / Decimal(rates[fx["base"]])
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def check_conversion(line, raw_bytes):
    """Problems (empty = consistent) of a converted line against the raw ECB bytes it cites: hash, rate date, the published rates, and the USD amount recomputed."""
    fx = line.get("fx") or {}
    if not raw_bytes or hashlib.sha256(raw_bytes).hexdigest() != fx.get("sha256"):
        return ["fx raw evidence missing or hash mismatch"]
    problems = []
    if fx.get("url") != ECB_URL or fx.get("quote") != "USD" or fx.get("base") != line.get("original_currency"):
        problems.append("fx source/currency pair inconsistent with the line")
    parsed = ecb_usd_per_unit(raw_bytes.decode("utf-8", "replace"), str(fx.get("base")))
    if not parsed:
        return problems + ["fx raw evidence does not contain the rates"]
    _, used, day = parsed
    if used != fx.get("ecb_rates_per_eur") or day != fx.get("rate_date"):
        problems.append("fx rates/date do not match the raw ECB bytes")
    elif convert_to_usd(line.get("original_amount"), fx) != line.get("amount"):
        problems.append("converted USD amount does not equal native amount x cited ECB rate")
    return problems
