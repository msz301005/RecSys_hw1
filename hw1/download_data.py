"""Download original UCSD Amazon Reviews 2018 files; never commit the data."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

SOURCES = {
    "Video_Games_5.json.gz": "http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz",
    "meta_Video_Games.json.gz": "http://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz",
}
EXPECTED_SHA256 = {
    "Video_Games_5.json.gz": "68df0c4ceb50f23f9136518ff6fd31fe07b379e2774dbf80c7250f4f1fbaa896",
    "meta_Video_Games.json.gz": "45637fb16e705393faa39c50f75790b33b2665c3675a1a23a84dd47a22ad684d",
}


def download_all(directory="data"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for filename, url in SOURCES.items():
        dest = directory / filename
        if not dest.exists():
            print(f"Downloading {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "RecSysHomework/1.0 (academic research)"})
            with urllib.request.urlopen(request, timeout=120) as response, dest.with_suffix(dest.suffix + ".part").open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            dest.with_suffix(dest.suffix + ".part").replace(dest)
        with dest.open("rb") as f:
            magic = f.read(2)
        if magic != b"\x1f\x8b":
            raise ValueError(f"{dest} is not a gzip archive (possibly an HTML response)")
        with dest.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != EXPECTED_SHA256[filename]:
            raise ValueError(f"SHA-256 mismatch for {filename}: expected {EXPECTED_SHA256[filename]}, got {digest}")
        manifest[filename] = {"url": url, "bytes": dest.stat().st_size, "sha256": digest}
        print(filename, manifest[filename], flush=True)
    (directory / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data")
    download_all(parser.parse_args().directory)
