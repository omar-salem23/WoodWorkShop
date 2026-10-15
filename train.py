#!/usr/bin/env python3
"""Train and evaluate a three-class YOLO object detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
import torch
from ultralytics import YOLO


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained YOLO weights")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--retry-epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cpu", help="cpu, or GPU index such as 0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--background-conf", type=float, default=0.25)
    parser.add_argument("--max-background-fpr", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=Path("runs/woodwork"))
    parser.add_argument("--min-map50", type=float, default=0.65)
    parser.add_argument("--min-map", type=float, default=0.35)
    parser.add_argument("--min-class-map", type=float, default=0.20)
    return parser.parse_args()


def prepare_data_yaml(source: Path, output: Path) -> list[str]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must contain a mapping")

    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    if not isinstance(names, list) or len(names) != 3 or not all(isinstance(n, str) for n in names):
        raise ValueError(f"Expected exactly three class names in dataset YAML; found {names!r}")

    root_value = Path(data.get("path", "."))
    root = root_value if root_value.is_absolute() else source.parent / root_value
    data["path"] = str(root.resolve())
    if "val" not in data and "valid" in data:
        data["val"] = data.pop("valid")

    for split in ("train", "val", "test"):
        if split not in data:
            raise ValueError(f"Dataset YAML is missing the required {split!r} split")
        split_path = Path(data[split])
        if not split_path.is_absolute():
            requested_path = split_path
            split_path = root / requested_path
            if not split_path.exists():
                split_folder = "valid" if split == "val" else split
                fallback = source.parent / split_folder / "images"
                if fallback.exists():
                    split_path = fallback
        if not split_path.exists():
            raise FileNotFoundError(f"{split} images directory not found: {split_path}")
        data[split] = str(split_path.resolve())

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print("Classes (label ID order):", ", ".join(f"{i}={name}" for i, name in enumerate(names)))
    return names


def evaluate(model_path: Path, data_yaml: Path, split: str, args: argparse.Namespace, name: str) -> dict:
    model = YOLO(str(model_path))
    result = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        save_json=True,
        plots=True,
        project=str(args.output),
        name=name,
        exist_ok=True,
    )
    config = yaml.safe_load(data_yaml.read_text())
    images_dir = Path(config['val' if split == 'val' else 'test'])
    background_ids = set()
    for image in images_dir.iterdir():
        if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            continue
        label = images_dir.parent / 'labels' / (image.stem + '.txt')
        if not label.exists() or not label.read_text().strip():
            background_ids.add(str(int(image.stem)) if image.stem.isnumeric() else image.stem)
    prediction_file = Path(result.save_dir) / 'predictions.json'
    predictions = json.loads(prediction_file.read_text()) if prediction_file.exists() else []
    false_boxes = [p for p in predictions if str(p['image_id']) in background_ids and p['score'] >= args.background_conf]
    false_images = {str(p['image_id']) for p in false_boxes}
    return {
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "per_class_map50_95": [float(x) for x in result.maps],
        "background": {
            "confidence": args.background_conf,
            "images": len(background_ids),
            "images_with_false_detections": len(false_images),
            "false_boxes": len(false_boxes),
            "image_false_positive_rate": len(false_images) / len(background_ids) if background_ids else None,
        },
    }


def main() -> None:
    args = get_args()
    torch.set_num_threads(args.threads)
    args.output = args.output.resolve()
    yaml_for_run = args.output / "dataset.yaml"
    class_names = prepare_data_yaml(args.data, yaml_for_run)

    common = {
        "data": str(yaml_for_run),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": str(args.output),
        "patience": 12,
        "seed": 42,
        "deterministic": True,
        "cache": False,
        "amp": args.device != "cpu",
        "close_mosaic": 10,
        "plots": True,
        "verbose": True,
    }

    first = YOLO(args.model)
    first.train(name="initial", epochs=args.epochs, **common)
    first_best = Path(first.trainer.best)
    first_metrics = evaluate(first_best, yaml_for_run, "val", args, "initial_val")
    print("Initial validation:", json.dumps(first_metrics))

    selected_weights = first_best
    selected_metrics = first_metrics
    needs_more_training = (
        first_metrics["map50"] < args.min_map50
        or first_metrics["map50_95"] < args.min_map
        or min(first_metrics["per_class_map50_95"], default=0.0) < args.min_class_map
        or (first_metrics['background']['image_false_positive_rate'] or 0.0) > args.max_background_fpr
    )
    if needs_more_training:
        print("Validation is below the configured target; continuing from the best checkpoint.")
        retry = YOLO(str(first_best))
        retry.train(name="retry", epochs=args.retry_epochs, **common)
        retry_best = Path(retry.trainer.best)
        retry_metrics = evaluate(retry_best, yaml_for_run, "val", args, "retry_val")
        print("Retry validation:", json.dumps(retry_metrics))
        first_score = (
            min(first_metrics["per_class_map50_95"], default=0.0),
            first_metrics["map50_95"],
            first_metrics["map50"],
            -(first_metrics['background']['image_false_positive_rate'] or 0.0),
        )
        retry_score = (
            min(retry_metrics["per_class_map50_95"], default=0.0),
            retry_metrics["map50_95"],
            retry_metrics["map50"],
            -(retry_metrics['background']['image_false_positive_rate'] or 0.0),
        )
        if retry_score >= first_score:
            selected_weights, selected_metrics = retry_best, retry_metrics

    # Keep the held-out test split for one final report; never use it to choose
    # a checkpoint or tune the retry.
    test_metrics = evaluate(selected_weights, yaml_for_run, "test", args, "final_test")
    summary = {
        "classes": class_names,
        "weights": str(selected_weights),
        "validation": selected_metrics,
        "test": test_metrics,
        "targets": {
            "map50": args.min_map50,
            "map50_95": args.min_map,
            "per_class_map50_95": args.min_class_map,
            "max_background_image_false_positive_rate": args.max_background_fpr,
        },
        "meets_validation_targets": (
            selected_metrics["map50"] >= args.min_map50
            and selected_metrics["map50_95"] >= args.min_map
            and min(selected_metrics["per_class_map50_95"], default=0.0) >= args.min_class_map
            and (selected_metrics['background']['image_false_positive_rate'] or 0.0) <= args.max_background_fpr
        ),
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
