"""Download real photographs of QR codes in the wild from Wikimedia Commons.

Only freely licensed images are used, and every image's author, licence and source page
are recorded in data/web/raw/sources.json for attribution. An image is kept when our
preprocessor finds *and decodes* a QR whose side is at least --min-qr-side pixels, so
the kept set is directly usable as genuine posters.

The run also measures how often the QR stage succeeds on raw real-world photos
(data/web/raw/detection_stats.json). That part of the pipeline is tested on real data.
"""

import argparse
import json
import re
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

from qrguard.config import DATA_DIR
from qrguard.preprocess import find_qr

API = "https://commons.wikimedia.org/w/api.php"
UA = "QRGuard-student-project/1.0 (Amrita Vishwa Vidyapeetham coursework)"
ROOT_CATEGORIES = ["Category:Photographs of QR codes", "Category:QR codes by country",
                   "Category:QR codes by city", "Category:QR code scanning"]


def api(params):
    q = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def walk_categories(roots, max_depth):
    seen, frontier = set(roots), [(c, 0) for c in roots]
    while frontier:
        cat, d = frontier.pop(0)
        yield cat
        if d >= max_depth:
            continue
        cont = {}
        while True:
            r = api({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmtype": "subcat",
                     "cmlimit": 500, **cont})
            for m in r["query"]["categorymembers"]:
                if m["title"] not in seen:
                    seen.add(m["title"])
                    frontier.append((m["title"], d + 1))
            if "continue" not in r:
                break
            cont = r["continue"]


def files_in(cat, width):
    cont = {}
    while True:
        r = api({"action": "query", "generator": "categorymembers", "gcmtitle": cat, "gcmtype": "file",
                 "gcmlimit": 100, "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
                 "iiurlwidth": width, **cont})
        for p in r.get("query", {}).get("pages", {}).values():
            if p.get("imageinfo"):
                yield p["title"], p["imageinfo"][0]
        if "continue" not in r:
            break
        cont = r["continue"]


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


def download(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-images", type=int, default=400, help="photos to examine")
    ap.add_argument("--keep", type=int, default=40, help="stop after this many usable photos")
    ap.add_argument("--min-qr-side", type=int, default=150)
    ap.add_argument("--width", type=int, default=1600, help="thumbnail width requested from Commons")
    ap.add_argument("--depth", type=int, default=2, help="sub-category depth")
    args = ap.parse_args()

    out = DATA_DIR / "web" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    sources, stats = [], {"examined": 0, "qr_found": 0, "qr_decoded": 0, "kept": 0, "download_errors": 0}
    seen_files = set()
    try:
        for cat in walk_categories(ROOT_CATEGORIES, args.depth):
            for title, info in files_in(cat, args.width):
                if title in seen_files or info.get("mime") not in ("image/jpeg", "image/png"):
                    continue
                seen_files.add(title)
                meta = info.get("extmetadata", {})
                lic = strip_html(meta.get("LicenseShortName", {}).get("value"))
                if not lic or "fair use" in lic.lower():
                    continue
                try:
                    img = download(info.get("thumburl") or info["url"])
                except Exception:
                    stats["download_errors"] += 1
                    continue
                if img is None:
                    continue
                stats["examined"] += 1
                corners, payload = find_qr(img)
                if corners is not None:
                    stats["qr_found"] += 1
                    stats["qr_decoded"] += bool(payload)
                    side = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
                    if payload and side >= args.min_qr_side:
                        name = f"web_{stats['kept']:03d}.jpg"
                        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        sources.append({
                            "file": name, "title": title, "page": info.get("descriptionurl"),
                            "author": strip_html(meta.get("Artist", {}).get("value")), "license": lic,
                            "category": cat, "qr_side_px": round(side), "payload": payload,
                        })
                        stats["kept"] += 1
                        print(f"[{stats['kept']:3d}] {title}  (QR {side:.0f}px, {lic})")
                if stats["kept"] >= args.keep or stats["examined"] >= args.max_images:
                    raise StopIteration
                time.sleep(0.2)          # be polite to the Commons servers
    except StopIteration:
        pass

    with open(out / "sources.json", "w") as f:
        json.dump(sources, f, indent=2, ensure_ascii=False)
    stats["qr_found_rate"] = stats["qr_found"] / max(stats["examined"], 1)
    stats["qr_decoded_rate"] = stats["qr_decoded"] / max(stats["examined"], 1)
    with open(out / "detection_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
