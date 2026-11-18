"""
Download AIT-LDS-v2.1 *no-pcaps* archives from Zenodo, streaming-safe.

Fixes in this version:
  * `--dest` is resolved against the REPOSITORY ROOT, not the current
    working directory. Running the script from any sub-folder now puts
    the data in <repo>/datasets/ait/ regardless of CWD.
  * The resolved destination is printed before any I/O happens, so a
    wrong path is visible immediately.
  * (carried over) streaming download/hash/extract, resume via .part,
    --silos, --skip-existing, --force.
"""

import argparse
import gc
import hashlib
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total=None, initial=0, desc="", **_):
            self.total = total or 0
            self.n = initial
            self.desc = desc
        def __enter__(self): return self
        def __exit__(self, *_): sys.stdout.write("\n")
        def update(self, k):
            self.n += k
            if self.total:
                pct = 100.0 * self.n / self.total
                sys.stdout.write(f"\r  {self.desc}: {pct:5.1f}%")
                sys.stdout.flush()


# ---------------------------------------------------------------------------
# Repo-root resolution — the fix for the "wrong folder" bug
# ---------------------------------------------------------------------------
def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` looking for a repo marker. Falls back to `start`."""
    markers = ("README.md", "requirements.txt", "pyproject.toml", ".git")
    for parent in [start, *start.parents]:
        if any((parent / m).exists() for m in markers):
            return parent
    return start

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:                # e.g. inside Jupyter
    SCRIPT_DIR = Path.cwd()

REPO_ROOT    = _find_repo_root(SCRIPT_DIR)
DEFAULT_DEST = REPO_ROOT / "datasets" / "ait"


# ---------------------------------------------------------------------------
# Zenodo record 19483937
# ---------------------------------------------------------------------------
CHECKSUMS = {
    "fox_no-pcaps.zip":             "27b09dd2876ec08ee99a9b53e8718c92",
    "harrison_no-pcaps.zip":        "4e837053ce4c5496227e0a3b65048c1c",
    "shaw_no-pcaps.zip":            "f890a1f52bc9f893b44254ad86118e39",
    "wardbeck_no-pcaps.zip":        "14eeacd83571a83615c2cce1f6a6eacf",
    "wheeler_no-pcaps.zip":         "774a146275a22707ee78bcb4574aca23",
    "wilson_no-pcaps.zip":          "98285593473dc05d0a6899f7cdefcd2d",
    "russellmitchell_no-pcaps.zip": "abb43a46c70939ad9f98669e3510d2b7",
    "santos_no-pcaps.zip":          "cb484a2c55ea03c63c556f0531fe8367",
}
BASE_URL   = "https://zenodo.org/api/records/19483937/files/{key}/content"
CHUNK      = 1 << 20
USER_AGENT = "ait-lds-dl/1.1 (+https://zenodo.org/records/19483937)"


def download_with_resume(url: str, dest: Path, chunk_size: int = CHUNK) -> None:
    part    = dest.with_suffix(dest.suffix + ".part")
    already = part.stat().st_size if part.exists() else 0
    req = Request(url, headers={"User-Agent": USER_AGENT})
    if already:
        req.add_header("Range", f"bytes={already}-")
    with urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0)) + already
        mode  = "ab" if already else "wb"
        with open(part, mode) as f, tqdm(
            total=total, initial=already, unit="B", unit_scale=True,
            desc=f"  dl {dest.name}",
        ) as bar:
            while True:
                buf = resp.read(chunk_size)
                if not buf:
                    break
                f.write(buf)
                bar.update(len(buf))
    part.rename(dest)


def md5_streaming(path: Path, chunk_size: int = CHUNK) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f, tqdm(
        total=path.stat().st_size, unit="B", unit_scale=True,
        desc=f"  md5 {path.name}",
    ) as bar:
        for buf in iter(lambda: f.read(chunk_size), b""):
            h.update(buf)
            bar.update(len(buf))
    return h.hexdigest()


def extract_zip_streaming(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for name in tqdm(zf.namelist(), desc="  extract", unit="file"):
            zf.extract(name, dest_dir)


def download_ait_no_pcaps(
    dest_root: str | Path | None = None,
    silos: list[str] | None = None,
    keep_zips: bool = False,
    verify: bool = True,
    skip_existing: bool = True,
) -> Path:
    """Download and extract the AIT-LDS-v2.1 no-pcaps archives."""
    # ---- the fix: default to the repo-root path, never CWD ----
    if dest_root is None:
        dest_root = DEFAULT_DEST
    dest_root = Path(dest_root).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    # Loud, upfront banner so you catch wrong paths before any I/O
    print(f"Working directory : {Path.cwd()}")
    print(f"Repo root         : {REPO_ROOT}")
    print(f"Destination       : {dest_root}")

    selected = silos or [k.replace("_no-pcaps.zip", "") for k in CHECKSUMS]
    selected = [s.strip() for s in selected if s.strip()]
    unknown  = [s for s in selected if f"{s}_no-pcaps.zip" not in CHECKSUMS]
    if unknown:
        raise ValueError(
            f"Unknown silo(s): {unknown}. "
            f"Valid: {[k.replace('_no-pcaps.zip','') for k in CHECKSUMS]}"
        )

    for silo in selected:
        key, expected = f"{silo}_no-pcaps.zip", CHECKSUMS[f"{silo}_no-pcaps.zip"]
        silo_dir, zip_path = dest_root / silo, dest_root / f"{silo}_no-pcaps.zip"

        print(f"\n=== {silo} ===")

        if skip_existing and silo_dir.is_dir() and (silo_dir / "gather").is_dir():
            print(f"  → already present at {silo_dir}, skipping")
            continue

        if not zip_path.exists():
            print(f"  → downloading {key}")
            download_with_resume(BASE_URL.format(key=key), zip_path)
        else:
            print("  → zip already on disk, reusing")

        if verify:
            got = md5_streaming(zip_path)
            if got != expected:
                zip_path.unlink(missing_ok=True)
                raise ValueError(
                    f"MD5 mismatch for {key}: expected {expected}, got {got}. "
                    f"Corrupt zip deleted — re-run to retry."
                )
            print("  ✓ MD5 OK")

        silo_dir.mkdir(exist_ok=True)
        print(f"  → extracting to {silo_dir}")
        extract_zip_streaming(zip_path, silo_dir)

        if not keep_zips:
            zip_path.unlink(missing_ok=True)
        del zip_path
        gc.collect()

    print(f"\nAll done. Datasets at: {dest_root}")
    return dest_root


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", default=None,
                   help="Override destination root. Defaults to <repo>/datasets/ait, "
                        "found by walking up from this script to the repo root.")
    p.add_argument("--silos", nargs="+", default=None,
                   help="Download only these silos, e.g. --silos fox harrison")
    p.add_argument("--keep-zips", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the folder already exists")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    download_ait_no_pcaps(
        dest_root=a.dest,               # None → DEFAULT_DEST (repo-rooted)
        silos=a.silos,
        keep_zips=a.keep_zips,
        verify=not a.no_verify,
        skip_existing=not a.force,
    )