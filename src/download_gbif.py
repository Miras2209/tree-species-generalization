"""Download research-grade iNaturalist observations through the GBIF API.

Source data (training, validation, in-distribution test): Europe and North America.
Shifted test data: Central Asia (Kazakhstan, Kyrgyzstan, Uzbekistan, Tajikistan,
Turkmenistan).

Usage:
    python -m src.download_gbif --part source
    python -m src.download_gbif --part central_asia
    python -m src.download_gbif --images      # downloads and resizes the photos
"""
import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

API = "https://api.gbif.org/v1"
PAGE = 300


def taxon_key(name):
    r = requests.get(f"{API}/species/match", params={"name": name, "rank": "SPECIES"}, timeout=30)
    r.raise_for_status()
    return r.json()["usageKey"]


def fetch_occurrences(name, region_params, limit):
    """Yield GBIF occurrence records with at least one openly licensed image."""
    key = taxon_key(name)
    offset, n = 0, 0
    while n < limit:
        params = {"datasetKey": C.INAT_DATASET_KEY, "taxonKey": key, "mediaType": "StillImage",
                  "limit": PAGE, "offset": offset, **region_params}
        r = requests.get(f"{API}/occurrence/search", params=params, timeout=60)
        r.raise_for_status()
        page = r.json()
        for rec in page["results"]:
            yield rec
            n += 1
            if n >= limit:
                return
        if page["endOfRecords"]:
            return
        offset += PAGE
        time.sleep(0.2)


def records_to_rows(rec, species, part):
    rows = []
    media = [m for m in rec.get("media", []) if m.get("type") == "StillImage"
             and any(l in str(m.get("license", "")) for l in C.OPEN_LICENSES)]
    for i, m in enumerate(media[: C.MAX_IMAGES_PER_OBS]):
        url = m["identifier"].replace("/original.", "/large.")
        rows.append({
            "obs_id": rec["gbifID"], "image_id": f'{rec["gbifID"]}_{i}', "url": url,
            "species": species, "genus": C.GENUS[species], "family": C.FAMILY[species],
            "observer": rec.get("recordedBy", "unknown"), "country": rec.get("countryCode"),
            "lat": rec.get("decimalLatitude"), "lon": rec.get("decimalLongitude"),
            "date": rec.get("eventDate"), "month": rec.get("month"), "part": part,
        })
    return rows


def collect(part):
    rows = []
    for sp in C.SPECIES_NAMES:
        if part == "source":
            for cont in C.SOURCE_CONTINENTS:
                limit = C.MAX_OBS_PER_SPECIES // len(C.SOURCE_CONTINENTS)
                for rec in fetch_occurrences(sp, {"continent": cont}, limit):
                    rows += records_to_rows(rec, sp, part)
        else:
            for cc in C.CENTRAL_ASIA:
                for rec in fetch_occurrences(sp, {"country": cc}, 10_000):
                    rows += records_to_rows(rec, sp, part)
        print(f"{part}: {sp}: {sum(r['species'] == sp for r in rows)} images", flush=True)
    df = pd.DataFrame(rows).dropna(subset=["lat", "lon", "month"])
    df["month"] = df["month"].astype(int)
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.DATA_DIR / f"meta_{part}.csv", index=False)
    return df


def download_one(row):
    path = C.IMG_DIR / f"{row.image_id}.jpg"
    if path.exists():
        return True
    try:
        r = requests.get(row.url, timeout=60)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        w, h = img.size
        s = C.SHORT_SIDE / min(w, h)
        if s < 1:
            img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
        img.save(path, quality=90)
        return True
    except Exception:
        return False


def download_images():
    C.IMG_DIR.mkdir(parents=True, exist_ok=True)
    for part in ("source", "central_asia"):
        df = pd.read_csv(C.DATA_DIR / f"meta_{part}.csv")
        with ThreadPoolExecutor(16) as ex:
            ok = list(ex.map(download_one, df.itertuples()))
        df["downloaded"] = ok
        df.to_csv(C.DATA_DIR / f"meta_{part}.csv", index=False)
        print(f"{part}: {sum(ok)}/{len(ok)} images downloaded")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["source", "central_asia"])
    ap.add_argument("--images", action="store_true")
    a = ap.parse_args()
    if a.part:
        collect(a.part)
    if a.images:
        download_images()
