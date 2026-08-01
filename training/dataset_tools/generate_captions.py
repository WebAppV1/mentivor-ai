"""
Mentivor AI — Caption File Generator (Phase 4)

Reads the filled-in caption_template.json and produces one .txt caption
file per image, using sparse captioning: trigger token + variable scene
attributes only. Fixed identity traits (hair, face, default outfit) are
intentionally omitted so the LoRA binds them to the trigger token.

Usage:
    python training/dataset_tools/generate_captions.py --character mentivor
"""

import argparse
import json
from pathlib import Path

TRIGGER_TOKEN = "mntvrchar"


def build_caption(entry: dict) -> str:
    parts = [TRIGGER_TOKEN]
    for field in ["shot", "view_angle", "pose_action", "environment", "expression"]:
        val = entry.get(field, "").strip()
        if val:
            parts.append(val)

    outfit = entry.get("outfit_variant", "").strip()
    if outfit:
        parts.append(outfit)

    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    char_dir = project_root / "datasets" / "characters" / args.character
    template_path = char_dir / "caption_template.json"
    captions_dir = char_dir / "captions"

    if not template_path.exists():
        print(f"ERROR: {template_path} not found. Run generate_caption_template.py first.")
        return

    with open(template_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    captions_dir.mkdir(parents=True, exist_ok=True)

    empty_fields_warning = []
    written = 0

    for entry in entries:
        filename = entry["filename"]
        stem = Path(filename).stem
        caption = build_caption(entry)
        txt_path = captions_dir / f"{stem}.txt"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption)

        written += 1
        print(f"  {filename:25s} -> {caption}")

        if caption.strip() == TRIGGER_TOKEN:
            empty_fields_warning.append(filename)

    print(f"\n{written} caption file(s) written to {captions_dir}")

    if empty_fields_warning:
        print(f"\nWARNING: {len(empty_fields_warning)} image(s) have NO fields filled in")
        print("(caption is just the trigger token). Did you forget to fill in the template?")
        for f in empty_fields_warning:
            print(f"  - {f}")


if __name__ == "__main__":
    main()