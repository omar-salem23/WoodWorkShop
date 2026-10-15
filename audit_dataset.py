#!/usr/bin/env python3
"""Check image/label pairs and normalized YOLO boxes before training."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dataset root containing train/valid/test")
    parser.add_argument("--classes", type=int, default=3)
    args = parser.parse_args()
    root = args.dataset.resolve()
    problems = 0
    first_split_by_hash: dict[str, tuple[str, str]] = {}
    cross_split_duplicates: list[tuple[str, str, str]] = []

    for split in ("train", "valid", "test"):
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            print(f"{split}: MISSING images or labels directory")
            problems += 1
            continue
        images = {p.stem: p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        labels = {p.stem: p for p in labels_dir.glob("*.txt")}
        missing_labels = sorted(images.keys() - labels.keys())
        orphan_labels = sorted(labels.keys() - images.keys())
        counts: Counter[int] = Counter()
        bad_rows: list[str] = []
        for stem, label_path in labels.items():
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                try:
                    if len(fields) != 5:
                        raise ValueError("expected class x_center y_center width height")
                    class_id = int(fields[0])
                    box = [float(x) for x in fields[1:]]
                    if not 0 <= class_id < args.classes:
                        raise ValueError(f"class id {class_id} outside [0, {args.classes - 1}]")
                    if any(not 0 <= value <= 1 for value in box) or box[2] <= 0 or box[3] <= 0:
                        raise ValueError("box values must be normalized and width/height positive")
                    counts[class_id] += 1
                except ValueError as exc:
                    bad_rows.append(f"{label_path.name}:{line_no}: {exc}")

        for image_path in images.values():
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            previous = first_split_by_hash.get(digest)
            if previous and previous[0] != split:
                cross_split_duplicates.append((previous[0], previous[1], f"{split}/{image_path.name}"))
            else:
                first_split_by_hash.setdefault(digest, (split, image_path.name))

        print(
            f"{split}: images={len(images)} labels={len(labels)} "
            f"boxes={dict(sorted(counts.items()))} missing_labels={len(missing_labels)} "
            f"orphan_labels={len(orphan_labels)} invalid_rows={len(bad_rows)}"
        )
        for label in missing_labels[:5]:
            print(f"  missing label: {label}")
        for label in orphan_labels[:5]:
            print(f"  label without image: {label}")
        for row in bad_rows[:10]:
            print(f"  invalid: {row}")
        if orphan_labels or bad_rows:
            problems += len(orphan_labels) + len(bad_rows)

    if problems:
        print(f"Dataset audit found {problems} problem(s).")
        return 1
    if cross_split_duplicates:
        print(f"Exact duplicate images appear across splits: {len(cross_split_duplicates)}")
        for first_split, first_name, duplicate in cross_split_duplicates[:10]:
            print(f"  {first_split}/{first_name} duplicates {duplicate}")
        print("Resolve cross-split duplicates before treating validation/test scores as independent.")
        return 1
    print("Dataset audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
