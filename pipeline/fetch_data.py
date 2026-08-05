"""
One-off data fetch: pulls pbp/weekly_off/weekly_def/roster parquet for
seasons not already cached in data/, matching the existing file naming
(pbp_{season}.parquet etc.) so the rest of the pipeline needs no changes.

Per PROJECT_BRIEF.md's documented install gotcha: nfl_data_py pins
pandas<2.0,numpy<1.0 but works fine at runtime against modern versions
(confirmed here at pandas 3.0.5 / numpy 2.5.1) -- installed with --no-deps
plus requests/appdirs separately, not by relaxing the pin.
"""

import sys

import nfl_data_py as nfl

DATA_DIR = "../data"


def fetch_season(season, data_dir=DATA_DIR):
    print(f"fetching {season}...")

    pbp = nfl.import_pbp_data([season])
    pbp.to_parquet(f"{data_dir}/pbp_{season}.parquet")
    print(f"  pbp: {len(pbp)} rows")

    weekly_off = nfl.import_weekly_data([season])
    weekly_off.to_parquet(f"{data_dir}/weekly_off_{season}.parquet")
    print(f"  weekly_off: {len(weekly_off)} rows")

    weekly_def = nfl.import_weekly_pfr("def", [season])
    weekly_def.to_parquet(f"{data_dir}/weekly_def_{season}.parquet")
    print(f"  weekly_def: {len(weekly_def)} rows")

    roster = nfl.import_seasonal_rosters([season])
    roster.to_parquet(f"{data_dir}/roster_{season}.parquet")
    print(f"  roster: {len(roster)} rows")


if __name__ == "__main__":
    seasons = [int(s) for s in sys.argv[1:]]
    if not seasons:
        print("usage: python fetch_data.py <season> [<season> ...]")
        sys.exit(1)
    for season in seasons:
        fetch_season(season)
