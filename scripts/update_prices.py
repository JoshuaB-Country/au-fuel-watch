#!/usr/bin/env python3
"""Daily fuel price updater — calls Claude AI with web search to fetch
current Australian petrol and diesel prices, then writes data/prices.json."""

import json
import re
import sys
from datetime import date
from pathlib import Path

import anthropic

SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR.parent / "data" / "prices.json"
TODAY = date.today().isoformat()

SYSTEM_PROMPT = """\
You are a precise data-extraction assistant. Your job is to search the web
for current Australian fuel prices and return them as a strict JSON object.
Always prefer the most recent data you can find. Round all prices to the
nearest integer cent per litre."""

USER_PROMPT = f"""\
Today is {TODAY}. Search the web for the latest Australian average fuel prices.

Find:
1. National average petrol (ULP91) price in cents per litre
2. National average diesel (ULS) price in cents per litre
3. State/territory breakdown for NSW, VIC, QLD, WA, SA, TAS, NT, ACT
4. Any significant fuel price news or events today

Good sources: ACCC weekly petrol monitoring report, FuelWatch WA
(fuelwatch.wa.gov.au), NRMA (nrma.com.au), RACQ, AAA Fuel Price Report,
GasBuddy Australia.

Return ONLY a valid JSON object — no markdown fences, no explanation text,
no trailing commas. Use this exact schema:

{{
  "date": "{TODAY}",
  "petrol": <integer cents per litre>,
  "diesel": <integer cents per litre>,
  "note": "<one-sentence note about a significant price event today, or null>",
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


def extract_json(text: str) -> dict:
    """Extract a JSON object from a text response."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group())
    raise ValueError(f"Could not extract JSON from response:\n{text[:400]}")


def run_claude_with_search() -> dict:
    """Run Claude with web search and return structured price data."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": USER_PROMPT}]

    for iteration in range(10):
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 8,
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
            # For built-in web_search, tool results are provided by the server.
            # Pass empty tool_result placeholders so the conversation can continue.
            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": "",
                }
                for b in response.content
                if b.type == "tool_use"
            ]
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

    raise RuntimeError("Exceeded maximum iterations without an end_turn response.")


def validate_entry(entry: dict) -> None:
    """Basic sanity-check on the fetched data."""
    petrol = entry.get("petrol")
    diesel = entry.get("diesel")
    if not isinstance(petrol, int) or not isinstance(diesel, int):
        raise ValueError(f"Non-integer prices: petrol={petrol}, diesel={diesel}")
    if not (80 <= petrol <= 500):
        raise ValueError(f"Petrol price out of plausible range: {petrol}")
    if not (80 <= diesel <= 500):
        raise ValueError(f"Diesel price out of plausible range: {diesel}")


def update_data_file(entry: dict) -> None:
    """Merge new price entry into data/prices.json."""
    with open(DATA_FILE) as f:
        data = json.load(f)

    # Update or append series entry
    series = data["series"]
    idx = next((i for i, s in enumerate(series) if s["date"] == TODAY), None)

    new_point = {"date": TODAY, "petrol": entry["petrol"], "diesel": entry["diesel"]}
    if entry.get("note"):
        new_point["note"] = entry["note"]

    if idx is not None:
        series[idx] = new_point
        print(f"  Updated existing series entry for {TODAY}")
    else:
        series.append(new_point)
        series.sort(key=lambda s: s["date"])
        print(f"  Appended new series entry for {TODAY}")

    # Update state breakdown
    if "states" in entry and isinstance(entry["states"], dict):
        data["states"] = entry["states"]
        print("  Updated state prices")

    data["lastUpdated"] = TODAY

    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Wrote {DATA_FILE}")


def main() -> None:
    print(f"Fetching Australian fuel prices for {TODAY}...")

    try:
        entry = run_claude_with_search()
        print(f"  Petrol: {entry['petrol']} cpl  |  Diesel: {entry['diesel']} cpl")
        validate_entry(entry)
        update_data_file(entry)
        print("Done.")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
