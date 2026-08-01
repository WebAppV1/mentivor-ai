"""
Mentivor AI — Dataset Import & Validation Tool (Phase 3)

Scans a raw intake folder of character images, filters out corrupt or
too-small files, detects near-duplicates via perceptual hashing, and
copies approved images into the curated dataset folder with sequential
naming and metadata.

This tool checks TECHNICAL quality only (corrupt files, resolution,
duplicates). It does NOT judge visual correctness (wrong hair, wrong
outfit, malformed hands, character drift) — that requires manual review
per the project's dataset curation rules.

Usage (PowerShell):
    python training\\dataset_tools\\import_and_validate.py `
        --character mentivor `
        --min-resolution 768

Run from the mentivor-ai project root.
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import imagehash

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
HASH_DISTANCE_THRESHOLD = 5  # lower = stricter duplicate detection


def compute_file_hash(path: Path) -> str:
    """SHA-256 of raw file bytes, for exact-duplicate detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_image(path: Path, min_resolution: int):
    """
    Returns (ok: bool, reason: str, image: PIL.Image | None, size: tuple | None)
    """
    try:
        img = Image.open(path)
        img.verify()  # checks for corruption
        img = Image.open(path)  # reopen after verify() invalidates the handle
        img.load()
    except Exception as e:
        return False, f"corrupt_or_unreadable: {e}", None, None

    width, height = img.size
    if min(width, height) < min_resolution:
        return False, f"below_min_resolution ({width}x{height})", None, (width, height)

    return True, "ok", img, (width, height)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", required=True, help="character id, e.g. mentivor")
    parser.add_argument("--min-resolution", type=int, default=768,
                         help="minimum edge (px) required to accept an image")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "datasets" / "characters" / args.character / "raw"
    images_dir = project_root / "datasets" / "characters" / args.character / "images"
    metadata_path = project_root / "datasets" / "characters" / args.character / "metadata.json"

    if not raw_dir.exists():
        print(f"ERROR: raw folder not found: {raw_dir}")
        print("Create it and drop your source images there first.")
        return

    images_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )

    if not candidates:
        print(f"No images found in {raw_dir}")
        return

    print(f"Found {len(candidates)} candidate file(s) in {raw_dir}\n")

    seen_file_hashes = set()
    seen_perceptual_hashes = []  # list of (imagehash, original_filename)
    accepted = []
    rejected = []

    for path in candidates:
        file_hash = compute_file_hash(path)
        if file_hash in seen_file_hashes:
            rejected.append({"file": path.name, "reason": "exact_duplicate_bytes"})
            print(f"  REJECT  {path.name:40s} exact duplicate")
            continue
        seen_file_hashes.add(file_hash)

        ok, reason, img, resolution = validate_image(path, args.min_resolution)
        if not ok:
            rejected.append({"file": path.name, "reason": reason})
            print(f"  REJECT  {path.name:40s} {reason}")
            continue

        phash = imagehash.phash(img)
        is_near_dupe = False
        for existing_hash, existing_name in seen_perceptual_hashes:
            if phash - existing_hash <= HASH_DISTANCE_THRESHOLD:
                rejected.append({
                    "file": path.name,
                    "reason": f"near_duplicate_of:{existing_name}"
                })
                print(f"  REJECT  {path.name:40s} near-duplicate of {existing_name}")
                is_near_dupe = True
                break

        if is_near_dupe:
            continue

        seen_perceptual_hashes.append((phash, path.name))
        accepted.append({
            "original_filename": path.name,
            "resolution": {"width": resolution[0], "height": resolution[1]},
            "file_hash": file_hash,
            "perceptual_hash": str(phash),
        })
        print(f"  ACCEPT  {path.name:40s} {resolution[0]}x{resolution[1]}")

    # Copy accepted images into curated folder with sequential naming
    metadata_entries = []
    for i, entry in enumerate(accepted, start=1):
        src = raw_dir / entry["original_filename"]
        ext = src.suffix.lower()
        new_name = f"{args.character}_{i:04d}{ext}"
        dst = images_dir / new_name
        shutil.copy2(src, dst)

        metadata_entries.append({
            "filename": new_name,
            "original_filename": entry["original_filename"],
            "resolution": entry["resolution"],
            "file_hash": entry["file_hash"],
            "perceptual_hash": entry["perceptual_hash"],
            "status": "accepted_pending_manual_review",
            "imported_at": datetime.now(timezone.utc).isoformat(),
        })

    metadata = {
        "character_id": args.character,
        "import_run": datetime.now(timezone.utc).isoformat(),
        "source_folder": str(raw_dir),
        "total_candidates": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "rejected": rejected,
        "accepted": metadata_entries,
        "note": (
            "Technical validation only (corrupt/resolution/duplicate). "
            "Visual correctness (hair, outfit, anatomy, style match) "
            "requires manual review before these images are used for training."
        ),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n--- Summary ---")
    print(f"Candidates:  {len(candidates)}")
    print(f"Accepted:    {len(accepted)}  -> copied to {images_dir}")
    print(f"Rejected:    {len(rejected)}")
    print(f"Metadata written to: {metadata_path}")
    print(f"\nNEXT STEP: manually review the {len(accepted)} accepted images in")
    print(f"{images_dir} and remove any with wrong hair/outfit/anatomy/style")
    print("before proceeding to captioning (Phase 4).")


if __name__ == "__main__":
    main()