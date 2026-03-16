"""
FIA Formula 2 Multi-Season Race Data Scraper
=============================================
Scrapes race results, sprint results, qualifying, and driver/team standings
from fiaformula2.com across all available seasons (2017–present).


WHY PLAYWRIGHT (NOT BEAUTIFULSOUP OR SELENIUM)
----------------------------------------------
fiaformula2.com is a JavaScript-rendered Single Page Application (SPA). If you
fetch it with requests + BeautifulSoup you get a near-empty HTML shell — the
results tables are injected by JS after page load, so static scraping sees
nothing useful.

Playwright launches a real Chromium browser, waits for the JS to finish
rendering, then extracts the data. Selenium could also do this, but Playwright
is preferred because:
  - Auto-waiting: it waits for elements to appear before extracting, eliminating
    the need for manual time.sleep() calls around every interaction.
  - Cleaner Python API and actively maintained by Microsoft.
  - Faster and more reliable for modern SPAs.


HOW THE SITE IS STRUCTURED (URL PARAMETERS)
--------------------------------------------
fiaformula2.com exposes all data via clean, predictable URL parameters.
This means we bypass any UI season selector entirely and drive everything
through direct URL construction — no clicking dropdowns required.

Key URL patterns:
    /Calendar?seasonid=NNN         — all race round links for a season
    /Results?raceid=NNN            — results for a specific race weekend
    /Results?seasonid=NNN          — results landing page for a season (fallback)
    /Standings/Driver?seasonId=NNN — driver championship standings
    /Standings/Team?seasonId=NNN   — team championship standings

On each Results page, session types (Feature Race, Sprint, Qualifying) are
presented as tab buttons. The script clicks each tab and extracts the table.


SEASON ID MAP
-------------
These season IDs were discovered by inspecting live URLs on the site:

    Year  |  seasonid
    ------+-----------
    2017  |  174
    2018  |  175
    2019  |  176
    2020  |  177
    2021  |  178
    2022  |  179
    2023  |  180
    2024  |  181
    2025  |  182
    2026  |  183

IDs are sequential integers. If a future season (2027+) is not found, try
incrementing from 183 — the pattern has been consistent since 2017.


DATA RANGE
----------
The scraper can pull data for every season from 2017 to the current season.
This covers the FIA Formula 2 Championship in its current format (rebranded
from GP2 Series in 2017). Data availability per season:

    2017–present  Feature race results (podium, laps, time/gap, points)
    2017–present  Sprint race results
    2017–present  Qualifying results
    2017–present  Driver championship standings (end-of-season or live)
    2017–present  Team championship standings

Historical seasons (2017–prior year) are complete. The current season will
only contain rounds that have already taken place.

Note: column structure varies slightly across seasons (e.g. fastest lap column
added in later years). Extra columns are stored as col_5 through col_9 — inspect
your CSV output and rename them to match the actual content.


API INTERCEPT (BONUS OPTIMISATION)
------------------------------------
On first run the script listens to all network traffic on the Results page.
Modern SPAs often call hidden internal REST/JSON endpoints to fetch their data.
If any are found, they are saved to f2_api_intercepts.json.

Why this matters: if the site fetches results via a JSON API, you can call
those URLs directly with requests.get() — no Playwright, no browser, just fast
lightweight HTTP calls. Check the intercepts file after your first run; if it
contains recognisable race data you may be able to simplify the scraper
significantly.


INSTALL DEPENDENCIES (run once)
--------------------------------
    pip install playwright pandas
    playwright install chromium


USAGE
-----
    # Scrape all seasons (2017–2026):
    python f2_scraper.py

    # Scrape specific seasons only:
    python f2_scraper.py --seasons 2024 2025

    # Scrape a single season and watch the browser (useful for debugging):
    python f2_scraper.py --seasons 2025 --headless false

    # Scrape only feature race results and standings (skip sprint/qualifying):
    python f2_scraper.py --sessions race standings

    # Recommended first run — single season smoke test before pulling everything:
    python f2_scraper.py --seasons 2024 --headless false


OUTPUT FILES (CSV, written to ./f2_data/)
-----------------------------------------
    f2_race_results.csv      — Feature race results, all scraped seasons
    f2_sprint_results.csv    — Sprint race results, all scraped seasons
    f2_qualifying.csv        — Qualifying results, all scraped seasons
    f2_driver_standings.csv  — Driver championship standings, all scraped seasons
    f2_team_standings.csv    — Team championship standings, all scraped seasons
    f2_api_intercepts.json   — Raw JSON captured from any hidden API endpoints

Each CSV includes a `season` column so all seasons can be stored in a single
file and filtered/joined easily. Data is appended season-by-season as the
script runs, so a crash mid-run does not lose data from completed seasons.


TROUBLESHOOTING
---------------
    Rows come back empty:
        Set --headless false to watch the browser. The site may have updated
        its HTML structure and the CSS selectors need adjusting.

    No rounds found for a season:
        The Calendar page may render differently for that year. Check the
        Results page directly in a browser using the seasonid URL parameter.

    Timeout errors:
        The site may be slow or rate-limiting requests. Try increasing
        REQUEST_GAP and PAGE_TIMEOUT at the top of the script.

    A future season is missing:
        Try adding the next sequential season ID to the SEASONS dict at the
        top of the script (e.g. 2027: 184) and re-run.
"""

import argparse
import json
import time
import csv
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ── Season map ────────────────────────────────────────────────────────────────

SEASONS = {
    2017: 174,
    2018: 175,
    2019: 176,
    2020: 177,
    2021: 178,
    2022: 179,
    2023: 180,
    2024: 181,
    2025: 182,
    2026: 183,
}

BASE_URL     = "https://www.fiaformula2.com"
OUTPUT_DIR   = Path("f2_data")
PAGE_TIMEOUT = 30_000   # ms to wait for elements
SLOW_MO      = 80       # ms between Playwright actions (polite pacing)
REQUEST_GAP  = 1.2      # seconds between page loads (be a good citizen)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="FIA Formula 2 multi-season scraper")
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=list(SEASONS.keys()),
        metavar="YEAR",
        help="Season year(s) to scrape, e.g. --seasons 2024 2025. Default: all."
    )
    parser.add_argument(
        "--headless", type=lambda x: x.lower() != "false", default=True,
        help="Run browser headless (default: true). Pass 'false' to watch."
    )
    parser.add_argument(
        "--sessions", nargs="+",
        choices=["race", "sprint", "qualifying", "standings"],
        default=["race", "sprint", "qualifying", "standings"],
        help="Which data to scrape. Default: all."
    )
    return parser.parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    return text.replace("\xa0", " ").replace("\u200b", "").strip()


def save_csv(rows: list[dict], filename: str, mode: str = "a") -> None:
    """Append rows to a CSV file (creates with header if new)."""
    if not rows:
        return
    path = OUTPUT_DIR / filename
    file_exists = path.exists()
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists or mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def wait_for_table(page, timeout: int = PAGE_TIMEOUT) -> bool:
    """Wait for a results table to appear. Returns True if found."""
    try:
        page.wait_for_selector("table tbody tr", timeout=timeout)
        return True
    except PlaywrightTimeout:
        return False


def extract_table(page, season: int, round_label: str, session: str) -> list[dict]:
    """
    Extract all rows from the visible results table on the current page.
    Column order on fiaformula2.com: Pos | No | Driver | Team | ... | Points
    We capture all cells generically and map by position.
    """
    rows = []
    trs = page.query_selector_all("table tbody tr")

    for tr in trs:
        cells = tr.query_selector_all("td")
        if len(cells) < 3:
            continue

        texts = [clean(c.inner_text()) for c in cells]

        row = {
            "season":   season,
            "session":  session,
            "round":    round_label,
            "position": texts[0] if len(texts) > 0 else "",
            "number":   texts[1] if len(texts) > 1 else "",
            "driver":   texts[2] if len(texts) > 2 else "",
            "team":     texts[3] if len(texts) > 3 else "",
        }

        # Capture extra columns generically (laps, time, gap, fastest lap, points…)
        # Column count varies by session type and season year
        extra_labels = ["col_5", "col_6", "col_7", "col_8", "col_9"]
        for i, label in enumerate(extra_labels):
            idx = 4 + i
            if idx < len(texts):
                row[label] = texts[idx]

        rows.append(row)

    return rows

# ── Round discovery ───────────────────────────────────────────────────────────

def get_rounds_for_season(page, season: int, season_id: int) -> list[dict]:
    """
    Load the Calendar page for a season and collect all race round links.
    Race IDs are embedded in result links as ?raceid=NNN.
    Returns: [{"race_id": 1092, "label": "Round 1 Melbourne"}]
    """
    url = f"{BASE_URL}/Calendar?seasonid={season_id}"
    print(f"    Loading calendar: {url}")
    page.goto(url, wait_until="networkidle")
    time.sleep(0.8)

    race_links = page.eval_on_selector_all(
        "a[href*='raceid']",
        "els => els.map(el => ({ href: el.href, text: el.innerText.trim() }))"
    )

    seen_ids = set()
    rounds = []
    for link in race_links:
        href = link["href"]
        if "raceid=" not in href:
            continue
        try:
            race_id = int(href.split("raceid=")[1].split("&")[0])
        except (ValueError, IndexError):
            continue
        if race_id in seen_ids:
            continue
        seen_ids.add(race_id)
        text = clean(link["text"])
        rounds.append({"race_id": race_id, "label": text or f"raceid-{race_id}"})

    # Fallback: try the Results page directly
    if not rounds:
        url2 = f"{BASE_URL}/Results?seasonid={season_id}"
        print(f"    No race links on Calendar — trying Results page: {url2}")
        page.goto(url2, wait_until="networkidle")
        time.sleep(0.8)
        race_links = page.eval_on_selector_all(
            "a[href*='raceid']",
            "els => els.map(el => ({ href: el.href, text: el.innerText.trim() }))"
        )
        for link in race_links:
            href = link["href"]
            if "raceid=" not in href:
                continue
            try:
                race_id = int(href.split("raceid=")[1].split("&")[0])
            except (ValueError, IndexError):
                continue
            if race_id in seen_ids:
                continue
            seen_ids.add(race_id)
            text = clean(link["text"])
            rounds.append({"race_id": race_id, "label": text or f"raceid-{race_id}"})

    print(f"    Found {len(rounds)} race rounds.")
    return sorted(rounds, key=lambda r: r["race_id"])

# ── Session scrapers ──────────────────────────────────────────────────────────

SESSION_KEYWORDS = {
    "race":       ["feature race", "feature", "race"],
    "sprint":     ["sprint race", "sprint"],
    "qualifying": ["qualifying", "quali"],
    "practice":   ["practice", "free practice"],
}

def scrape_race_round(page, season: int, round_info: dict, sessions_wanted: list[str]) -> dict:
    """
    Load a race result page and extract data for each requested session type.
    Returns {"race": [...], "sprint": [...], "qualifying": [...]}
    """
    race_id = round_info["race_id"]
    label   = round_info["label"]
    url     = f"{BASE_URL}/Results?raceid={race_id}"
    results = {s: [] for s in sessions_wanted}

    page.goto(url, wait_until="networkidle")
    time.sleep(0.5)

    # Discover session tabs (button/link selectors that vary across site versions)
    tab_els = page.query_selector_all(
        "button.btn, .tab-button, [role='tab'], .session-nav a, "
        ".result-nav button, .results-navigation a, .btn-group button, "
        "ul.nav li a, .tabs a, .nav-tabs a, .nav-tabs button"
    )

    if tab_els:
        for tab in tab_els:
            tab_text = clean(tab.inner_text()).lower()
            matched_session = None
            for session, keywords in SESSION_KEYWORDS.items():
                if session not in sessions_wanted:
                    continue
                if any(kw in tab_text for kw in keywords):
                    matched_session = session
                    break

            if not matched_session:
                continue

            try:
                tab.click()
                page.wait_for_selector("table tbody tr", timeout=10_000)
                time.sleep(0.3)
                rows = extract_table(page, season, label, matched_session)
                results[matched_session].extend(rows)
                print(f"      {matched_session.capitalize()}: {len(rows)} rows")
            except PlaywrightTimeout:
                print(f"      [!] Timeout after clicking '{tab_text}' tab")
            except Exception as e:
                print(f"      [!] Tab error '{tab_text}': {e}")
    else:
        # No tabs — grab whatever table is visible (assume feature race)
        if "race" in sessions_wanted and wait_for_table(page, timeout=12_000):
            rows = extract_table(page, season, label, "race")
            results["race"].extend(rows)
            print(f"      Race (no tabs): {len(rows)} rows")
        else:
            print(f"      [!] No tabs and no table at {url}")

    return results

# ── Standings ─────────────────────────────────────────────────────────────────

def scrape_standings(page, season: int, season_id: int) -> tuple[list[dict], list[dict]]:
    """Scrape driver and team standings for a given season."""
    driver_rows = []
    team_rows   = []

    for standing_type, container in [("driver", driver_rows), ("team", team_rows)]:
        url = f"{BASE_URL}/Standings/{standing_type.capitalize()}?seasonId={season_id}"
        page.goto(url, wait_until="networkidle")
        time.sleep(0.5)

        if not wait_for_table(page, timeout=15_000):
            print(f"      [!] No {standing_type} standings table found for {season}")
            continue

        trs = page.query_selector_all("table tbody tr")
        for tr in trs:
            cells = tr.query_selector_all("td")
            if len(cells) < 2:
                continue
            texts = [clean(c.inner_text()) for c in cells]
            row = {
                "season":   season,
                "position": texts[0] if len(texts) > 0 else "",
                "name":     texts[1] if len(texts) > 1 else "",
                "points":   texts[-1],
            }
            if standing_type == "driver" and len(texts) > 2:
                row["team"] = texts[2]
            container.append(row)

        print(f"      {standing_type.capitalize()} standings: {len(container)} entries")

    return driver_rows, team_rows

# ── API intercept ─────────────────────────────────────────────────────────────

def intercept_api_calls(page, season_id: int) -> list[dict]:
    """
    Listen for XHR/fetch JSON responses on the Results page.
    If the site calls hidden REST endpoints, you may be able to hit those URLs
    directly with requests.get() — no browser needed.
    """
    captured = []
    SKIP_KEYWORDS = (
        "analytics", "gtm", "sentry", "fonts", "cdn", "cloudfront",
        "google", "facebook", "twitter", "doubleclick", "segment",
        "hotjar", "intercom",
    )

    def handle_response(response):
        ct = response.headers.get("content-type", "")
        if "json" in ct:
            url = response.url
            if not any(k in url for k in SKIP_KEYWORDS):
                try:
                    data = response.json()
                    captured.append({"url": url, "data": data})
                except Exception:
                    pass

    page.on("response", handle_response)
    page.goto(f"{BASE_URL}/Results?seasonid={season_id}", wait_until="networkidle")
    time.sleep(2)
    page.remove_listener("response", handle_response)
    return captured

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    invalid = [y for y in args.seasons if y not in SEASONS]
    if invalid:
        print(f"[!] Unknown season year(s): {invalid}")
        print(f"    Available: {sorted(SEASONS.keys())}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Clear stale output files for a clean run
    for fname in [
        "f2_race_results.csv", "f2_sprint_results.csv", "f2_qualifying.csv",
        "f2_driver_standings.csv", "f2_team_standings.csv",
    ]:
        (OUTPUT_DIR / fname).unlink(missing_ok=True)

    print("=" * 60)
    print("  FIA Formula 2 Multi-Season Scraper")
    print(f"  Seasons : {sorted(args.seasons)}")
    print(f"  Sessions: {args.sessions}")
    print(f"  Headless: {args.headless}")
    print(f"  Output  : ./{OUTPUT_DIR}/")
    print("=" * 60)

    all_api_intercepts = {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=args.headless,
            slow_mo=SLOW_MO,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()

        for year in sorted(args.seasons):
            season_id = SEASONS[year]
            print(f"\n{'─' * 60}")
            print(f"  Season {year}  (seasonid={season_id})")
            print(f"{'─' * 60}")

            # API check (once, on first season)
            if year == sorted(args.seasons)[0]:
                print("\n  [API check] Listening for hidden JSON endpoints …")
                intercepts = intercept_api_calls(page, season_id)
                if intercepts:
                    all_api_intercepts[year] = intercepts
                    print(f"  → Captured {len(intercepts)} JSON endpoint(s).")
                    print("    See f2_api_intercepts.json — direct HTTP calls may work!")
                else:
                    print("  → No useful JSON endpoints found; scraping HTML tables.")

            # Standings
            if "standings" in args.sessions:
                print(f"\n  [Standings]")
                driver_rows, team_rows = scrape_standings(page, year, season_id)
                save_csv(driver_rows, "f2_driver_standings.csv")
                save_csv(team_rows,   "f2_team_standings.csv")

            # Race results
            results_sessions = [s for s in args.sessions if s != "standings"]
            if not results_sessions:
                continue

            print(f"\n  [Rounds]")
            rounds = get_rounds_for_season(page, year, season_id)
            if not rounds:
                print(f"  [!] No rounds found for {year} — skipping results.")
                continue

            print(f"\n  [Results] Scraping {len(rounds)} round(s) …")
            season_race   = []
            season_sprint = []
            season_quali  = []

            for i, rnd in enumerate(rounds, 1):
                print(f"\n    Round {i}/{len(rounds)}: {rnd['label']}  (raceid={rnd['race_id']})")
                round_data = scrape_race_round(page, year, rnd, results_sessions)
                season_race.extend(round_data.get("race", []))
                season_sprint.extend(round_data.get("sprint", []))
                season_quali.extend(round_data.get("qualifying", []))
                time.sleep(REQUEST_GAP)

            save_csv(season_race,   "f2_race_results.csv")
            save_csv(season_sprint, "f2_sprint_results.csv")
            save_csv(season_quali,  "f2_qualifying.csv")

            print(
                f"\n  Season {year} totals → "
                f"Race: {len(season_race)} | "
                f"Sprint: {len(season_sprint)} | "
                f"Qualifying: {len(season_quali)}"
            )

        browser.close()

    # Save API intercepts if any found
    if all_api_intercepts:
        api_path = OUTPUT_DIR / "f2_api_intercepts.json"
        with open(api_path, "w", encoding="utf-8") as f:
            json.dump(all_api_intercepts, f, indent=2, default=str)
        print(f"\n  [✓] API intercepts → {api_path}")

    print("\n" + "=" * 60)
    print("  ✅  Done! Output files in ./f2_data/:")
    for fname in sorted(OUTPUT_DIR.iterdir()):
        size = fname.stat().st_size
        print(f"      {fname.name}  ({size:,} bytes)")
    print("=" * 60)
    print("""
TIPS:
  • Set --headless false to watch the browser if rows come back empty.
  • Check f2_api_intercepts.json — if results data is in there, you
    can call those URLs directly with requests.get() (no Playwright).
  • col_5–col_9 are generic column names. Inspect your CSV to rename
    them (typically: laps, time/gap, fastest lap, points).
  • Run --seasons 2024 first as a quick smoke test before the full run.
""")


if __name__ == "__main__":
    main()
