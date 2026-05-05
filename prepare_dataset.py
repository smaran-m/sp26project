"""Download CIFAR-10, pack into the zip format pfgmpp expects, build FID reference.

Idempotent: re-runs skip steps whose output already exists. Safe to call from
both local and SLURM contexts.

Outputs (relative to repo root):
  pfgmpp/downloads/cifar-10-python.tar.gz
  pfgmpp/datasets/cifar10-32x32.zip
  pfgmpp/fid-refs/cifar10-32x32.npz
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"

REPO_ROOT = Path(__file__).resolve().parent
PFGMPP_DIR = REPO_ROOT / "pfgmpp"
DOWNLOAD_PATH = PFGMPP_DIR / "downloads" / "cifar-10-python.tar.gz"
DATASET_ZIP = PFGMPP_DIR / "datasets" / "cifar10-32x32.zip"
FID_REF_NPZ = PFGMPP_DIR / "fid-refs" / "cifar10-32x32.npz"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def download_cifar(force: bool) -> None:
    DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DOWNLOAD_PATH.exists() and not force:
        if md5sum(DOWNLOAD_PATH) == CIFAR_MD5:
            log(f"CIFAR-10 tarball already present at {DOWNLOAD_PATH} (md5 ok)")
            return
        log("CIFAR-10 tarball checksum mismatch — re-downloading")
    log(f"Downloading CIFAR-10 from {CIFAR_URL}")
    urllib.request.urlretrieve(CIFAR_URL, DOWNLOAD_PATH)
    got = md5sum(DOWNLOAD_PATH)
    if got != CIFAR_MD5:
        raise RuntimeError(f"CIFAR-10 md5 mismatch: got {got}, expected {CIFAR_MD5}")
    log(f"Downloaded and verified ({DOWNLOAD_PATH.stat().st_size / 1e6:.1f} MB)")


def build_zip(force: bool) -> None:
    if DATASET_ZIP.exists() and not force:
        log(f"Dataset zip already present at {DATASET_ZIP}")
        return
    DATASET_ZIP.parent.mkdir(parents=True, exist_ok=True)
    log(f"Building dataset zip via pfgmpp/dataset_tool.py -> {DATASET_ZIP}")
    cmd = [
        sys.executable,
        "dataset_tool.py",
        f"--source={DOWNLOAD_PATH.relative_to(PFGMPP_DIR).as_posix()}",
        f"--dest={DATASET_ZIP.relative_to(PFGMPP_DIR).as_posix()}",
    ]
    subprocess.run(cmd, cwd=PFGMPP_DIR, check=True)


def validate_zip() -> None:
    log(f"Validating {DATASET_ZIP}")
    with zipfile.ZipFile(DATASET_ZIP) as zf:
        names = zf.namelist()
    n_images = sum(1 for n in names if n.lower().endswith(".png"))
    has_meta = any(n.endswith("dataset.json") for n in names)
    log(f"  {n_images} png files, dataset.json={'yes' if has_meta else 'no'}")
    if n_images < 50000:
        raise RuntimeError(f"expected >= 50000 images, got {n_images}")
    if not has_meta:
        raise RuntimeError("dataset.json missing from zip")


def build_fid_ref(force: bool) -> None:
    if FID_REF_NPZ.exists() and not force:
        log(f"FID reference already present at {FID_REF_NPZ}")
        return
    FID_REF_NPZ.parent.mkdir(parents=True, exist_ok=True)
    log(f"Building FID reference statistics -> {FID_REF_NPZ}")
    # `fid.py ref` uses torch.distributed; needs torchrun even with one rank.
    cmd = [
        sys.executable,
        "fid.py", "ref",
        f"--data={DATASET_ZIP.relative_to(PFGMPP_DIR).as_posix()}",
        f"--dest={FID_REF_NPZ.relative_to(PFGMPP_DIR).as_posix()}",
    ]
    subprocess.run(cmd, cwd=PFGMPP_DIR, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="rebuild every artifact even if it already exists")
    ap.add_argument("--skip-fid-ref", action="store_true",
                    help="skip the FID reference step (needs GPU/torch). Useful for "
                         "preparing data on a CPU node before GPU jobs.")
    args = ap.parse_args()

    log(f"REPO_ROOT = {REPO_ROOT}")
    if not PFGMPP_DIR.is_dir():
        raise SystemExit(f"missing upstream repo at {PFGMPP_DIR}")

    download_cifar(args.force)
    build_zip(args.force)
    validate_zip()
    if args.skip_fid_ref:
        log("Skipping FID reference (--skip-fid-ref).")
    else:
        build_fid_ref(args.force)
    log("Done.")


if __name__ == "__main__":
    main()
