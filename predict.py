#!/usr/bin/env python3
"""Run the trained workshop detector on an image or a folder of images."""
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Image file or directory of images')
    parser.add_argument('--model', type=Path, default=ROOT/'models/best.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/predictions')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--screwdriver-edge-margin', type=float, default=2.0,
        help='Drop screwdriver boxes within this percent of an image edge; set 0 to disable')
    parser.add_argument('--device', default='cpu', help='cpu, 0, or auto')
    args = parser.parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f'Input not found: {args.source}')
    if not args.model.is_file():
        raise FileNotFoundError(f'Model not found: {args.model}')
    model = YOLO(str(args.model))
    device = None if args.device == 'auto' else args.device
    output_dir = args.output.resolve()
    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    results = model.predict(source=str(args.source), conf=args.conf,
        device=device, save=False)
    for result in results:
        if args.screwdriver_edge_margin > 0 and result.boxes is not None:
            height, width = result.orig_shape
            margin = args.screwdriver_edge_margin / 100
            kept = []
            for index, box in enumerate(result.boxes):
                if int(box.cls.item()) == 1:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    edge_distance = min(x1 / width, y1 / height,
                        (width - x2) / width, (height - y2) / height)
                    if edge_distance <= margin:
                        continue
                kept.append(index)
            result.update(boxes=result.boxes.data[kept])
        counts = {}
        for box in result.boxes:
            class_id = int(box.cls.item())
            name = model.names[class_id]
            counts[name] = counts.get(name, 0) + 1
        result.save(filename=str(results_dir / Path(result.path).name))
        print(f'{Path(result.path).name}: {counts or "no detections"}')
    print(f'Annotated images saved in: {results_dir}')


if __name__ == '__main__':
    main()
