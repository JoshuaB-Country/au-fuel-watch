#!/usr/bin/env python3
"""Daily fuel price updater — calls Claude AI with web search to fetch
current Australian petrol and diesel prices, then writes data/prices.json.

Source hierarchy (most → least authoritative):
  1. AIP weekly retail prices        aip.com.au
  2. ACCC weekly monitoring          accc.gov.au
  3. FuelWatch WA (govt)             fuelwatch.wa.gov.au
  4. MotorMouth live aggregator      motormouth.com.au
  5. NRMA (NSW/ACT)                  nrma.com.au/living/transport/fuel
  6. RACQ (QLD)                      racq.com.au/cars/drive/fuel-prices
"""

import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import anthropic

SCRIPT_DIR = Path(__file__).parent
DATA_FILE  = SCRIPT_DIR.parent / "data" / "prices.json"

# Always use AEST (UTC+10) so the date matches what Australians see,
# regardless of what UTC time the GitHub Actions runner thinks it is.
AEST = timezone(timedelta(hours=10))
TODAY = datetime.now(AEST).strftime("%Y-%m-%d")


def load_previous_prices() -> dict | None:
    """Load yesterday's prices from the data file for sanity checking."""
    try:
        with open(DATA_FILE) as f:
            data = json.load(f)
        series = data.get("series", [])
        past = [s for s in series if s["date"] < TODAY]
        return past[-1] if past else None
    except Exception:
        return None


def build_prompts(prev: dict | None) -> tuple[str, str]:
    prev_context = ""
    if prev:
        prev_context = f"""
For reference, the most recent recorded prices are:
  - Petrol: {prev['petrol']} cpl  (date: {prev['date']})
  - Diesel: {prev['diesel']} cpl  (date: {prev['date']})

Your reported prices should be plausible relative to these. A change of more
than 25 cpl in a single day is unusual — if you find such a move, note it and
double-check with a second source before accepting it.
"""

    system = """\
You are a precise data-extraction assistant specialising in Australian fuel prices.
Search the web methodically, cross-reference at least two independent sources,
and return only verified data as a strict JSON object.
Round all prices to the nearest integer cent per litre (cpl)."""

    user = f"""\
Today is {TODAY}. Fetch the latest Australian average retail fuel prices.
{prev_context}
━━━ SEARCH INSTRUCTIONS ━━━

Work through these sources in order. Stop when you have confirmed national
averages from at least TWO independent sources:

PRIMARY (most authoritative — check these first):
  • AIP weekly retail prices:
    https://aip.com.au/industry-resources/weekly-prices
  • ACCC weekly petrol monitoring report:
    https://www.accc.gov.au/consumers/petrol-price-cycles/petrol-prices-and-your-local-area
  • FuelWatch WA (government-run, daily):
    https://www.fuelwatch.wa.gov.au

SECONDARY (use if primary sources lack today's data or for state breakdown):
  • MotorMouth live aggregator:   https://www.motormouth.com.au
  • NRMA (NSW/ACT prices):        https://www.mynrma.com.au/membership/my-nrma-app/fuel-prices
  • RACQ (QLD prices):            https://www.racq.com.au/cars-and-driving/driving/fuel-and-oil/fuel-prices
  • RAA (SA prices):              https://www.raa.com.au/travel/fuel-prices
  • Petrol Spy:                   https://petrolspy.com.au

━━━ WHAT TO FIND ━━━

1. National average ULP91 (regular unleaded petrol) — cents per litre
2. National average ULS diesel — cents per litre
3. Brent crude spot price — USD per barrel (from Reuters, Bloomberg, or
   MarketWatch: https://www.marketwatch.com/investing/future/brent%20crude)
4. State/territory breakdown for all 8 jurisdictions
5. Any significant fuel price event today (price spike, policy change, etc.)
6. Which sources you used and whether they agreed

━━━ OUTPUT FORMAT ━━━

Return ONLY a valid JSON object. No markdown fences, no explanation, no
trailing commas. Use this exact schema:

{{
  "date": "{TODAY}",
  "petrol": <integer cpl — national average ULP91>,
  "diesel": <integer cpl — national average ULS diesel>,
  "brent_usd": <number — Brent crude spot price in USD per barrel, one decimal place>,
  "note": "<one sentence about a significant price event today, or null>",
  "sources": ["<source name 1>", "<source name 2>"],
  "confidence": "high | medium | low",
  "states": {{
    "NSW": {{"petrol": <int>, "diesel": <int>}},
    "VIC": {{"petrol": <int>, "diesel": <int>}},
    "QLD": {{"petrol": <int>, "diesel": <int>}},
    "WA":  {{"petrol": <int>, "diesel": <int>}},
    "SA":  {{"petrol": <int>, "diesel": <int>}},
    "TAS": {{"petrol": <int>, "diesel": <int>}},
    "NT":  {{"petrol": <int>, "diesel": <int>}},
    "ACT": {{"petrol": <int>, "diesel": <int>}}
  }}
}}"""

    return system, user


def extract_json(text: str) -> dict:
    """Extract a JSON object from a text response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group())
    raise ValueError(f"Could not extract JSON from response:\n{text[:400]}")


def run_claude_with_search(system: str, user: str) -> dict:
    """Run Claude with web search and return structured price data."""
    client   = anthropic.Anthropic()
    messages = [{"role": "user", "content": user}]

    for iteration in range(12):
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=system,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 10,
            }],
            messages=messages,
        )

        print(f"  [iter {iteration + 1}] stop_reason={response.stop_reason}, "
              f"blocks={[b.type for b in response.content]}")

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text = "".join(
                b.text for b in response.content
                if hasattr(b, "text") and b.type == "text"
            )
            return extract_json(text)

        if response.stop_reason == "tool_use":
            tool_results = [
                {"type": "tool_result", "tool_use_id": b.id, "content": ""}
                for b in response.content if b.type == "tool_use"
            ]
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

    raise RuntimeError("Exceeded maximum iterations without an end_turn response.")


def validate_entry(entry: dict, prev: dict | None) -> None:
    """Sanity-check the fetched data against known constraints and prior prices."""
    petrol = entry.get("petrol")
    diesel = entry.get("diesel")

    # Type check
    if not isinstance(petrol, int) or not isinstance(diesel, int):
        raise ValueError(f"Non-integer prices: petrol={petrol}, diesel={diesel}")

    # Absolute range
    if not (80 <= petrol <= 500):
        raise ValueError(f"Petrol price out of plausible range: {petrol} cpl")
    if not (80 <= diesel <= 500):
        raise ValueError(f"Diesel price out of plausible range: {diesel} cpl")

    # Diesel is almost always more expensive than petrol in Australia
    if diesel < petrol - 10:
        raise ValueError(
            f"Diesel ({diesel}) is unexpectedly much cheaper than petrol ({petrol}) — "
            "check sources"
        )

    # State prices should be within ±45 cpl of the national average
    for state, prices in entry.get("states", {}).items():
        for fuel, val in prices.items():
            national = petrol if fuel == "petrol" else diesel
            if abs(val - national) > 45:
                raise ValueError(
                    f"{state} {fuel} ({val} cpl) is >45 cpl from national avg "
                    f"({national} cpl) — likely a data error"
                )

    # Day-on-day movement check
    if prev:
        petrol_move = abs(petrol - prev["petrol"])
        diesel_move = abs(diesel - prev["diesel"])
        if petrol_move > 30:
            raise ValueError(
                f"Petrol moved {petrol_move} cpl in one day "
                f"({prev['petrol']} → {petrol}) — exceeds 30 cpl threshold. "
                "Verify with a second source before accepting."
            )
        if diesel_move > 30:
            raise ValueError(
                f"Diesel moved {diesel_move} cpl in one day "
                f"({prev['diesel']} → {diesel}) — exceeds 30 cpl threshold. "
                "Verify with a second source before accepting."
            )

    # Log confidence and sources
    confidence = entry.get("confidence", "unknown")
    sources    = entry.get("sources", [])
    print(f"  Confidence: {confidence}  |  Sources: {', '.join(sources) or 'not reported'}")
    if confidence == "low":
        print("  WARNING: Claude reported low confidence — consider manual verification.")


def update_data_file(entry: dict) -> None:
    """Merge new price entry into data/prices.json."""
    with open(DATA_FILE) as f:
        data = json.load(f)

    series = data["series"]
    idx    = next((i for i, s in enumerate(series) if s["date"] == TODAY), None)

    new_point: dict = {"date": TODAY, "petrol": entry["petrol"], "diesel": entry["diesel"]}
    if entry.get("brent_usd") is not None:
        new_point["brent_usd"] = round(float(entry["brent_usd"]), 1)
    if entry.get("note"):
        new_point["note"] = entry["note"]
    if entry.get("sources"):
        new_point["sources"] = entry["sources"]

    if idx is not None:
        series[idx] = new_point
        print(f"  Updated existing entry for {TODAY}")
    else:
        series.append(new_point)
        series.sort(key=lambda s: s["date"])
        print(f"  Appended new entry for {TODAY}")

    if "states" in entry and isinstance(entry["states"], dict):
        data["states"] = entry["states"]
        print("  Updated state prices")

    data["lastUpdated"] = TODAY

    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Wrote {DATA_FILE}")


def main() -> None:
    print(f"Fetching Australian fuel prices for {TODAY}...")

    prev          = load_previous_prices()
    system, user  = build_prompts(prev)

    if prev:
        print(f"  Previous prices: petrol={prev['petrol']} cpl, "
              f"diesel={prev['diesel']} cpl ({prev['date']})")

    try:
        entry = run_claude_with_search(system, user)
        print(f"  Fetched:  petrol={entry['petrol']} cpl  |  diesel={entry['diesel']} cpl")
        validate_entry(entry, prev)
        update_data_file(entry)
        print("Done.")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
